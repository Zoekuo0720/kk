import streamlit as st
import pandas as pd
import numpy as np
import pickle
import jieba
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib as mpl
from matplotlib import font_manager # 用於更精確地管理 Matplotlib 字體
import os # 用於檢查檔案是否存在
from scipy.sparse import csr_matrix # 用於載入稀疏矩陣
import io # 用於處理上傳的檔案
from docx import Document # 用於讀取 Word 文件

# --- Matplotlib 中文字體設定 ---
# 嘗試載入微軟正黑體 (msjh.ttc)，優先從專案目錄尋找
# 請確保 'msjh.ttc' 字體文件已複製到您的 Streamlit 應用程式所在的目錄
font_path_for_mpl = "msjh.ttc" 

if os.path.exists(font_path_for_mpl):
    try:
        # 添加字體到 Matplotlib 的字體管理器
        font_manager.fontManager.addfont(font_path_for_mpl)
        # 獲取字體在 Matplotlib 中的名稱 (通常對於 msjh.ttc 就是 'Microsoft JhengHei')
        prop = font_manager.FontProperties(fname=font_path_for_mpl)
        mpl_font_name = prop.get_name()
        plt.rcParams['font.sans-serif'] = [mpl_font_name] # 設定 Matplotlib 預設字體
        st.sidebar.success(f"Matplotlib 字體 '{mpl_font_name}' 載入成功。")
    except Exception as e:
        st.sidebar.warning(f"警告：無法載入專案目錄下的字體 '{font_path_for_mpl}' 到 Matplotlib。錯誤：{e}")
        # 如果載入失敗，回退到嘗試使用系統中已安裝的字體名稱
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        st.sidebar.info("Matplotlib 將嘗試使用系統中已安裝的 'Microsoft JhengHei' 字體。")
else:
    st.sidebar.warning(f"警告：專案目錄下找不到字體文件 '{font_path_for_mpl}'。")
    st.sidebar.info("Matplotlib 將嘗試使用系統中已安裝的 'Microsoft JhengHei' 字體。")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 回退到系統字體名稱

plt.rcParams['axes.unicode_minus'] = False # 解決 Matplotlib 負號顯示為方塊的問題


# --- 載入模型和數據 ---
@st.cache_resource # 使用 Streamlit 的快取裝飾器，避免每次運行都重新載入
def load_resources():
    """載入預訓練的 TF-IDF Vectorizer、分類模型、以及測試集數據。"""
    try:
        # 確保這些 .pkl 和 .xlsx 檔案與您的 Streamlit 腳本在同一個目錄下
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('mnb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('X_test.pkl', 'rb') as f:
            X_test_loaded = pickle.load(f) # 載入保存的測試集特徵矩陣
        with open('y_test.pkl', 'rb') as f:
            y_test_loaded = pickle.load(f) # 載入保存的測試集真實標籤
        
        # 載入原始評論資料，用於生成總體文字雲和獲取所有類別標籤
        # 假設您的評論資料檔案為 '評論資料.xlsx'
        df_reviews = pd.read_excel("評論資料.xlsx")
        
        # 確保 '分類標籤' 和 '評論內容' 列存在
        if '分類標籤' not in df_reviews.columns:
            st.error("評論資料中缺少 '分類標籤' 欄位。請檢查 '評論資料.xlsx' 檔案。")
            return None, None, None, None, None, None
        if '評論內容' not in df_reviews.columns:
            st.error("評論資料中缺少 '評論內容' 欄位。請檢查 '評論資料.xlsx' 檔案。")
            return None, None, None, None, None, None

        # 載入停用詞清單 (與 main_project.py 保持一致)
        # 確保 'stopwords.txt' 檔案與您的 Streamlit 腳本在同一個目錄下
        def load_stopwords_internal(path='stopwords.txt'):
            with open(path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f if line.strip()])
        stopwords = load_stopwords_internal()

        # 自訂斷詞函數 (與 main_project.py 保持一致)
        def preprocess_text_for_wordcloud(text):
            text = str(text).strip()
            if not text:
                return ""
            words = jieba.cut(text)
            return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

        df_reviews['processed_review'] = df_reviews['評論內容'].apply(preprocess_text_for_wordcloud)
        
        return vectorizer, model, X_test_loaded, y_test_loaded, df_reviews, stopwords
    except FileNotFoundError as e:
        st.error(f"錯誤：找不到必要的檔案。請確保 'tfidf_vectorizer.pkl', 'mnb_model.pkl', 'X_test.pkl', 'y_test.pkl', '評論資料.xlsx' 和 'stopwords.txt' 都在應用程式的相同目錄下。詳細錯誤：{e}")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"載入資源時發生錯誤：{e}")
        return None, None, None, None, None, None

vectorizer, model, X_test_loaded, y_test_loaded, df_reviews, stopwords = load_resources()

# 檢查資源是否成功載入
if vectorizer is None or model is None or X_test_loaded is None or y_test_loaded is None or df_reviews is None:
    st.stop() # 如果載入失敗，則停止應用程式

# 獲取模型訓練時的類別列表
class_labels = model.classes_

# 自訂斷詞函數 (用於使用者輸入，與訓練時保持一致)
def preprocess_text_for_prediction(text):
    text = str(text).strip()
    if not text:
        return ""
    words = jieba.cut(text)
    return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

# --- 輔助函數：從 Word 文件中提取文本 ---
def get_docx_text(file):
    document = Document(file)
    full_text = []
    for para in document.paragraphs:
        if para.text.strip(): # 只添加非空段落
            full_text.append(para.text.strip())
    return "\n".join(full_text)

# --- 初始化 session state ---
if 'classified_df_for_display' not in st.session_state:
    st.session_state.classified_df_for_display = None


# --- Streamlit 應用程式介面 ---
st.set_page_config(layout="wide", page_title="評論分析儀表板")

st.title("員工評論智能分析儀表板")
st.markdown("本儀表板運用先進的機器學習模型，旨在智能分析員工評論，自動識別潛在問題主題，並以直觀的視覺化方式呈現模型性能與關鍵洞察，助力企業優化管理決策。")

st.markdown("---") # 分隔線增加視覺區隔

# --- 負評分類區塊 ---
st.header("1. 負評即時分類")
st.markdown("您可以選擇輸入單條評論進行即時分類，或上傳批量文件進行自動分析。")

# 使用 Streamlit 的 tabs 來區分單條評論和批量文件分析
tab1, tab2 = st.tabs(["單條評論分析", "批量文件分析"])

with tab1: # 這個區塊的內容會在點擊「單條評論分析」時顯示
    st.subheader("單條評論快速分析")
    user_input = st.text_area("請輸入您想分析的員工負評或建議：", "報到流程很混亂，文件準備不齊全，窗口也不明確。", height=100)

    if st.button("分析負評", key="single_analysis_button"): # 添加 key 避免按鈕衝突
        if user_input:
            # 預處理使用者輸入
            processed_input = preprocess_text_for_prediction(user_input)
            # 轉換為 TF-IDF 向量
            input_vector = vectorizer.transform([processed_input])
            # 進行預測
            prediction = model.predict(input_vector)
            predicted_category = prediction[0]

            st.success(f"**預測的負評主題是：** `{predicted_category}`")
        else:
            st.warning("請輸入評論內容。")

with tab2: # 這個區塊的內容會在點擊「批量文件分析」時顯示
    st.subheader("批量文件自動分類")
    st.markdown("請上傳包含員工評論的 Excel 檔案 (.xlsx) 或 Word 檔案 (.docx)。")
    
    # 檔案上傳器在這裡！
    uploaded_file = st.file_uploader("選擇檔案", type=["xlsx", "docx"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        df_uploaded = None
        comment_column = None

        try:
            if file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": # .xlsx
                df_uploaded = pd.read_excel(uploaded_file)
                st.info("Excel 檔案上傳成功！請確認下方數據預覽並選擇評論欄位。")
                st.dataframe(df_uploaded.head()) # 顯示前幾行讓用戶確認

                # 讓用戶選擇包含評論的欄位
                column_options = df_uploaded.columns.tolist()
                comment_column = st.selectbox("請選擇包含評論內容的欄位：", column_options, index=column_options.index('評論內容') if '評論內容' in column_options else 0, key="excel_column_select")
            
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": # .docx
                st.info("Word 檔案上傳成功！系統將提取文件所有文本內容進行分析。")
                doc_text = get_docx_text(uploaded_file)
                if doc_text:
                    # 將整個 Word 文件內容作為一條評論處理
                    df_uploaded = pd.DataFrame({'評論內容': [doc_text]})
                    comment_column = '評論內容'
                    st.text_area("Word 文件內容預覽 (前500字):", doc_text[:500] + "..." if len(doc_text) > 500 else doc_text, height=150)
                else:
                    st.warning("上傳的 Word 文件中沒有可讀取的文本內容。")
                    df_uploaded = None

            else:
                st.error("不支援的檔案類型。請上傳 .xlsx 或 .docx 檔案。")
                df_uploaded = None

            if df_uploaded is not None and comment_column is not None:
                if st.button("開始批量分類", key="batch_analysis_button"): # 添加 key 避免按鈕衝突
                    with st.spinner("正在分析評論，請稍候..."):
                        # 準備用於儲存結果的列表
                        predictions = []
                        original_texts = [] # 儲存原始文本
                        processed_texts = [] # 儲存處理後的文本

                        # 遍歷選定欄位的每一條評論
                        for index, row in df_uploaded.iterrows():
                            comment_text = str(row[comment_column]) # 確保是字串
                            original_texts.append(comment_text)

                            processed_comment = preprocess_text_for_prediction(comment_text)
                            
                            if processed_comment: # 避免處理空評論
                                input_vector = vectorizer.transform([processed_comment])
                                prediction = model.predict(input_vector)[0]
                            else:
                                prediction = "無法分類 (內容空缺)" # 或其他標示
                            
                            predictions.append(prediction)
                            processed_texts.append(processed_comment) # 保存處理後的文本

                        # 將預測結果添加到新的 DataFrame 副本中
                        df_results = pd.DataFrame({
                            '原始評論內容': original_texts,
                            '處理後評論內容': processed_texts, # 可選：顯示處理後的文本
                            '預測負評主題': predictions
                        })
                        
                        # 將分類結果儲存到 session state，供其他區塊使用
                        st.session_state.classified_df_for_display = df_results

                        st.success("批量分類完成！請查看下方結果並可選擇下載。")
                        st.dataframe(df_results) # 顯示分類結果

                        # 提供下載分類結果的選項
                        @st.cache_data # 快取數據，避免每次互動都重新生成
                        def convert_df_to_excel(df):
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False, sheet_name='Classified Reviews')
                            processed_data = output.getvalue()
                            return processed_data

                        excel_data = convert_df_to_excel(df_results)
                        st.download_button(
                            label="下載分類結果 (Excel)",
                            data=excel_data,
                            file_name="classified_reviews.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            elif df_uploaded is None:
                pass # 檔案類型不支援或Word文件為空，不做進一步處理
            else:
                st.warning("請選擇包含評論內容的欄位，或確保 Word 文件有內容。")

        except Exception as e:
            st.error(f"讀取或處理檔案時發生錯誤：{e}")
            st.info("請確認您上傳的是有效的 Excel 或 Word 檔案，且包含預期的評論欄位。")

st.markdown("---") # 分隔線增加視覺區隔

# --- 模型評估區塊 ---
st.header("2. 模型性能評估")
st.markdown("以下圖表展示了模型在**測試集**上的表現，幫助您了解模型的分類能力與潛在改進空間。")

# 使用 st.expander 讓介面更整潔
with st.expander("點擊查看分類報告 (Classification Report)"):
    st.subheader("2.1 分類報告")
    st.markdown("分類報告提供了模型在每個類別上的精準率、召回率和 F1-分數，以及整體表現。")
    y_pred_test = model.predict(X_test_loaded)
    report = classification_report(y_test_loaded, y_pred_test, target_names=class_labels, output_dict=True)
    
    # 將報告轉換為 DataFrame 以便 Streamlit 顯示
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format("{:.2f}"))
    st.markdown("""
    * **Precision (精確率):** 模型預測為某類別的樣本中，真正屬於該類別的比例。
    * **Recall (召回率):** 所有真正屬於某類別的樣本中，模型成功預測出是該類別的比例。
    * **F1-Score (F1 分數):** 精確率和召回率的調和平均值，綜合衡量兩者。
    * **Support (支持數):** 測試集中每個類別的真實樣本數量。
    """)

with st.expander("點擊查看混淆矩陣 (Confusion Matrix)"):
    st.subheader("2.2 混淆矩陣")
    st.markdown("混淆矩陣直觀展示了模型在各類別上的分類正確與錯誤情況，對角線為正確分類，非對角線為誤分類。")
    y_pred_test = model.predict(X_test_loaded)
    
    fig, ax = plt.subplots(figsize=(12, 10)) # 調整圖表大小以確保標籤清晰
    cm = confusion_matrix(y_test_loaded, y_pred_test, labels=class_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel('預測類別 (Predicted Label)')
    ax.set_ylabel('真實類別 (True Label)')
    ax.set_title('混淆矩陣')
    plt.xticks(rotation=45, ha='right') # 旋轉X軸標籤以避免重疊
    plt.yticks(rotation=0) # 保持Y軸標籤水平
    plt.tight_layout() # 自動調整佈局以防止標籤截斷
    st.pyplot(fig)
    st.markdown("混淆矩陣的對角線數值表示正確分類的樣本數，非對角線數值表示錯誤分類的樣本數。")

st.markdown("---") # 分隔線增加視覺區隔

# --- 負評關鍵詞與主題洞察區塊 ---
st.header("3. 負評關鍵詞與主題洞察")
st.markdown("透過文字雲，您可以直觀地了解各負評主題中最具代表性的詞語和詞組，幫助您快速掌握問題核心。")

# 使用 selectbox 讓用戶選擇要顯示的文字雲
selected_category = st.selectbox(
    "請選擇您想查看文字雲的負評主題：",
    options=df_reviews['分類標籤'].unique().tolist()
)

if selected_category:
    st.subheader(f"{selected_category} 文字雲")
    
    # 過濾出該類別的評論
    # 這裡仍然使用原始載入的 df_reviews 來生成文字雲，因為文字雲是基於所有已有的數據來展示關鍵詞，
    # 而非單次上傳的數據。如果需要基於上傳數據的文字雲，則需要更複雜的邏輯來累積或替換。
    category_reviews_processed = df_reviews[df_reviews['分類標籤'] == selected_category]['processed_review']
    
    # 合併所有處理過的評論文本
    text_for_wordcloud = " ".join(category_reviews_processed.dropna())

    if text_for_wordcloud:
        # 設定 WordCloud 字體路徑
        font_path_for_wordcloud = "msjh.ttc" 
        
        # 檢查字體文件是否存在，如果不存在則給出警告
        if not os.path.exists(font_path_for_wordcloud):
            st.warning(f"警告：找不到字體文件 '{font_path_for_wordcloud}'。文字雲中文可能顯示為方塊。請確保字體文件存在於專案目錄。")
            font_path_for_wordcloud = None # 如果找不到，WordCloud 會使用預設字體

        wordcloud = WordCloud(
            font_path=font_path_for_wordcloud, # 指定字體路徑
            width=800, 
            height=400, 
            background_color='white', 
            collocations=False, # 避免重複詞組
            max_words=50 # 最多顯示50個詞
        ).generate(text_for_wordcloud)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5)) # 調整文字雲圖表大小
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.write("此類別暫無足夠評論生成文字雲。")

st.markdown("---") # 分隔線增加視覺區隔

# --- 評論主題分佈概覽區塊 (新增) ---
st.header("4. 評論主題分佈概覽") # 將標題從 5 改為 4
st.markdown("此直條圖展示了評論數據中各負評主題的數量分佈，幫助您快速了解各類問題的頻繁程度。")

# 判斷使用哪個數據源來生成圖表
if st.session_state.classified_df_for_display is not None:
    # 使用批量分類後的數據
    st.info("當前圖表顯示的是您上傳檔案並分類後的評論主題分佈。")
    source_df = st.session_state.classified_df_for_display
    category_column = '預測負評主題'
else:
    # 使用應用程式啟動時載入的原始數據
    st.info("當前圖表顯示的是預設評論數據的各主題分佈。請上傳檔案進行批量分類以查看您的數據分佈。")
    source_df = df_reviews
    category_column = '分類標籤'

# 計算各分類標籤的數量
category_counts = source_df[category_column].value_counts().sort_values(ascending=False)

fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax_dist, palette='Blues_d')
ax_dist.set_title('各評論主題數量分佈') # 標題更通用
ax_dist.set_xlabel('評論主題') # 標籤更通用
ax_dist.set_ylabel('評論數')
plt.xticks(rotation=45, ha='right') # 旋轉X軸標籤以避免重疊
plt.tight_layout()
st.pyplot(fig_dist)


st.markdown("---")
st.write("© 分類互動模型")
