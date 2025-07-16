
<流程如下>
1️⃣ | 載入套件         匯入需要用到的 Python 套件(jiebe wordcloud)                     
2️⃣ | 讀取資料         從 Excel 檔案載入評論資料(pandas, openpyxl)                      
3️⃣ | Jieba 斷詞       將中文評論切割為詞彙供後續處理 jieba.cut()                      
4️⃣ | TF-IDF 向量化    將文字轉為可供機器學習使用的向量                      
5️⃣ | 切分資料集        分為訓練資料與測試資料                           
6️⃣ | 模型建立與訓練    使用 Naive Bayes 訓練分類模型                 
7️⃣ | 預測與評估        顯示準確率與分類報告（precision、recall、f1-score）    
  
step 1 安裝套件  
<在cmd終端機執行>  pip install pandas jieba scikit-learn matplotlib wordcloud openpyxl  
step 2 建立停用詞檔案  檔案須以.txt為結尾，並儲存在vs code專案檔案夾內  
step 3 整合程式碼至.py檔案中  
#載入套件
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 載入停用詞清單（放在程式同資料夾）
def load_stopwords(path='stopwords.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f if line.strip()])

stopwords = load_stopwords()

df = pd.read_excel("整合後所有負評資料.xlsx")
#  自訂斷詞函數（含停用詞過濾）
def jieba_cut(text):
    words = jieba.cut(str(text))
    return " ".join([w for w in words if w not in stopwords])

# 對「評論內容」進行斷詞並建立新欄位 
df["斷詞"] = df["評論內容"].apply(jieba_cut)
# 顯示前幾筆資料檢查
print(df[["評論內容", "斷詞"]].head())

# TF-IDF 特徵建構
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["斷詞"])
y = df["分類標籤"]

# 切分訓練與測試資料（80% / 20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練 Naive Bayes 模型 
model = MultinomialNB()
model.fit(X_train, y_train)

# 預測與分析結果
y_pred = model.predict(X_test)
print("\n分類準確度：", accuracy_score(y_test, y_pred))
print("\n分類報告：\n", classification_report(y_test, y_pred))

# 文字雲圖 
word_scores = X.sum(axis=0).A1
words = tfidf.get_feature_names_out()
tfidf_dict = dict(zip(words, word_scores))

wordcloud = WordCloud(font_path="msjh.ttc", background_color="white", width=800, height=400)
wordcloud.generate_from_frequencies(tfidf_dict)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("TF-IDF 文字雲：評論關鍵字")
plt.show()

#矩陣圖
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 使用微軟正黑體
df['分類標籤'].value_counts().plot(kind='bar', figsize=(10,5), title='各分類標籤數量')
plt.xlabel("分類")
plt.ylabel("評論數")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#混淆矩陣圖
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, xticks_rotation='vertical')
plt.title("分類預測混淆矩陣")
plt.tight_layout()
plt.show()

