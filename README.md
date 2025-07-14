
<流程如下>
1️⃣ | 載入套件         匯入需要用到的 Python 套件(jiebe wordcloud)                     
2️⃣ | 讀取資料         從 Excel 檔案載入評論資料(pandas, openpyxl)                      
3️⃣ | Jieba 斷詞       將中文評論切割為詞彙供後續處理 jieba.cut()                      
4️⃣ | TF-IDF 向量化    將文字轉為可供機器學習使用的向量                      
5️⃣ | 切分資料集        分為訓練資料與測試資料                           
6️⃣ | 模型建立與訓練    使用 Naive Bayes 訓練分類模型                 
7️⃣ | 預測與評估        顯示準確率與分類報告（precision、recall、f1-score） 

