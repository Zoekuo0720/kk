
<流程如下>
1️⃣ | 載入套件         匯入需要用到的 Python 套件(jiebe wordcloud)                     
2️⃣ | 讀取資料         從 Excel 檔案載入評論資料(pandas, openpyxl)                      
3️⃣ | Jieba 斷詞       將中文評論切割為詞彙供後續處理 jieba.cut()                      
4️⃣ | TF-IDF 向量化    將文字轉為可供機器學習使用的向量                      
5️⃣ | 切分資料集        分為訓練資料與測試資料                           
6️⃣ | 模型建立與訓練    使用 Naive Bayes 訓練分類模型                 
7️⃣ | 預測與評估        顯示準確率與分類報告（precision、recall、f1-score） 

pandas: 處理 Excel 表格與資料結構（DataFrame）
jieba: 中文斷詞
TfidfVectorizer: 將斷詞轉成數字（TF-IDF 向量）
MultinomialNB: 使用 Naive Bayes 模型分類
train_test_split: 切成訓練集與測試集
classification_report: 印出分類指標報告
accuracy_score: 顯示整體準確率
<<step 1>>匯入套件
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
<<step 2>>讀取資料與斷詞處理
df = pd.read_excel("整合後所有負評資料.xlsx")
df['斷詞'] = df['評論內容'].apply(lambda x: list(jieba.cut(x)))
<<step 3>>轉為TF-IDF向量格式
def jieba_cut(text):
    return " ".join(jieba.cut(str(text)))
df["斷詞"] = df["評論內容"].apply(jieba_cut)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["斷詞"])
<<step 4>>建立模型並訓練
y = df["分類標籤"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
<<step 5>>預測與評估
y_pred = model.predict(X_test)
print("\n分類準確度：", accuracy_score(y_test, y_pred))
print("\n分類報告：\n", classification_report(y_test, y_pred))
