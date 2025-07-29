
<流程如下>
1️⃣ | 載入套件         匯入需要用到的 Python 套件(jiebe wordcloud)                     
2️⃣ | 讀取資料         從 Excel 檔案載入評論資料(pandas, openpyxl)                      
3️⃣ | Jieba 斷詞       將中文評論切割為詞彙供後續處理 jieba.cut()                      
4️⃣ | TF-IDF 向量化    將文字轉為可供機器學習使用的向量                      
5️⃣ | 切分資料集        分為訓練資料與測試資料                           
6️⃣ | 模型建立與訓練    使用 Naive Bayes 訓練分類模型                 
7️⃣ | 預測與評估        顯示準確率與分類報告（precision、recall、f1-score）    
  

# 第一部分：載入套件與基本設定
# ==============================================================================
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from collections import Counter # 用於計算類別分佈
from sklearn.metrics import ConfusionMatrixDisplay # 用於繪製混淆矩陣
import re # 用於可能的文本清洗
import pickle 
from scipy.sparse import csr_matrix # 用於處理稀疏矩陣的保存

# 設定 Matplotlib 字體以正確顯示中文
plt.rcParams['font.family'] = 'Microsoft JhengHei' # 使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# 第二部分：載入資料與預處理
# ==============================================================================
# 載入停用詞清單
def load_stopwords(path='stopwords.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if line.strip()])

stopwords = load_stopwords()

# 載入評論資料 Excel 檔案
df = pd.read_excel("評論資料.xlsx")

# 自訂斷詞函數（含停用詞過濾）
def jieba_cut(text):
    text = str(text).strip()
    if not text: # 如果文本是空的，直接返回空字串
        return ""
    # 這裡可以加入更複雜的清理，例如移除數字或特殊符號
    # text = re.sub(r'[0-9]', '', text) # 移除數字
    # text = re.sub(r'[^\w\s]', '', text) # 移除標點符號 (依需求而定，可能影響N-gram語義)

    words = jieba.cut(text)
    # 過濾停用詞，並移除切割後可能出現的單一空白字串
    return " ".join([w for w in words if w not in stopwords and w.strip() != ''])

# 對「評論內容」欄位進行斷詞並建立新欄位
df["斷詞"] = df["評論內容"].apply(jieba_cut)

# 顯示前幾筆資料，檢查斷詞結果
print("--- 前 5 筆評論內容及斷詞結果 ---")
print(df[["評論內容", "斷詞"]].head())

# 第三部分：特徵工程 (TF-IDF N-gram)
# ==============================================================================
# 初始化 TF-IDF 特徵器，設定 ngram_range=(1, 3) 以包含單詞、二元詞組和三元詞組
tfidf = TfidfVectorizer(ngram_range=(1, 3))

# 使用斷詞後的文本來建構 TF-IDF 特徵矩陣
X = tfidf.fit_transform(df["斷詞"])
y = df["分類標籤"] # 定義目標變數（分類標籤）

# 檢查 TF-IDF 特徵器提取的詞彙
print("\n--- TF-IDF 提取的前 50 個特徵詞（含 N-gram）---")
print(tfidf.get_feature_names_out()[:50])
print(f"TF-IDF 特徵總數：{len(tfidf.get_feature_names_out())}")

# 第四部分：數據切分與不均衡處理 (SMOTE)
# ==============================================================================
# 切分訓練與測試資料（80% / 20%），random_state 確保每次切分結果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 顯示訓練集原始類別分佈，看看每個分類的評論數量
print("\n--- 訓練集原始類別分佈 (SMOTE 之前) ---")
print(Counter(y_train))

# 使用 SMOTE 進行過採樣，讓數量少的分類評論數量變多，達到平衡
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 顯示 SMOTE 後的訓練集類別分佈，看看數量是否已經均衡
print("\n--- SMOTE 後訓練集類別分佈 (SMOTE 之後) ---")
print(Counter(y_train_smote))

# 第五部分：建立與訓練分類模型
# ==============================================================================
# 建立 Naive Bayes 分類模型 (MultinomialNB 適合文本分類)
model = MultinomialNB()

# 使用 SMOTE 過採樣後的數據來訓練模型
# 讓模型在學習時能更公平地對待每個類別
model.fit(X_train_smote, y_train_smote)

# 第六部分：模型預測與評估
# ==============================================================================
# 使用測試集數據進行預測
y_pred = model.predict(X_test)

# 顯示模型的分類準確度
print("\n--- 模型分類準確度 ---")
print("分類準確度：", accuracy_score(y_test, y_pred))

# 顯示詳細的分類報告 (包含精確度、召回率、F1分數)
print("\n--- 模型分類報告 ---")
print(classification_report(y_test, y_pred))

# 第七部分：結果視覺化
# ==============================================================================

# 1. 各分類標籤數量長條圖 (顯示原始數據分佈)
plt.figure(figsize=(10, 6))
df['分類標籤'].value_counts().plot(kind='bar', color='skyblue')
plt.title('各分類標籤原始評論數量', fontsize=16)
plt.xlabel("分類標籤", fontsize=12)
plt.ylabel("評論數量", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10) # 旋轉 x 軸標籤，避免重疊
plt.tight_layout() 
plt.show()

# 2. 混淆矩陣圖 (顯示模型預測的詳細情況)
# ConfusionMatrixDisplay 直接顯示混淆矩陣
plt.figure(figsize=(10, 8)) # 調整圖形大小
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, xticks_rotation='vertical', cmap=plt.cm.Blues)
plt.title("分類預測混淆矩陣", fontsize=16)
plt.xlabel("預測標籤", fontsize=12)
plt.ylabel("真實標籤", fontsize=12)
plt.tight_layout()
plt.show()

# 3. 總體 TF-IDF 文字雲
# 從 TF-IDF 矩陣中獲取每個詞彙的總體分數
# X 是整個資料集的 TF-IDF 矩陣 (包含訓練和測試資料合併的原始數據)
# 確保這裡使用的是 tfidf 這個 TfidfVectorizer 物件
word_scores = X.sum(axis=0).A1
words = tfidf.get_feature_names_out()
tfidf_dict = dict(zip(words, word_scores))
font_path = "msjh.ttc" # Windows 系統常見的正黑體

wordcloud = WordCloud(
    font_path=font_path,
    background_color="white",
    width=1000, # 可以調整圖片寬度
    height=600, # 可以調整圖片高度
    max_words=200, # 調整顯示的最大詞彙數量
    margin=2, # 詞彙之間的邊距
    prefer_horizontal=0.9 # 更傾向水平排列詞彙
)
wordcloud.generate_from_frequencies(tfidf_dict)

plt.figure(figsize=(12, 8)) # 調整圖片顯示大小
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off") # 不顯示坐標軸
plt.title("總體評論關鍵字 TF-IDF 文字雲", fontsize=18) # 添加標題
plt.show()

# 保存 TF-IDF 特徵器
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("\nTF-IDF 特徵器已保存為 tfidf_vectorizer.pkl")

# 保存訓練好的模型
with open('mnb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("分類模型已保存為 mnb_model.pkl")

# 保存測試集的特徵矩陣 (X_test) 和真實標籤 (y_test)
# X_test 是稀疏矩陣，可以直接保存
with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
print("測試集特徵矩陣 X_test 已保存為 X_test.pkl")

with open('y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
print("測試集真實標籤 y_test 已保存為 y_test.pkl")
