#1 安裝套件（jieba、wordcloud 等）	pip install
#2	匯入程式庫與讀取 Excel 檔案     	pandas, openpyxl
#3	分詞與停用詞處理	                jieba.cut()
#4	統計高頻詞並轉為                  DataFrame	collections.Counter
#5	建立中文字體文字雲 WordCloud 圖片	WordCloud(font_path=...)
#6	顯示圖表                        	matplotlib.pyplot

1️⃣ | 載入套件         匯入需要用到的 Python 套件                     
2️⃣ | 讀取資料         從 Excel 檔案載入評論資料                      
3️⃣ | Jieba 斷詞       將中文評論切割為詞彙供後續處理                      
4️⃣ | TF-IDF 向量化    將文字轉為可供機器學習使用的向量                      
5️⃣ | 切分資料集        分為訓練資料與測試資料                           
6️⃣ | 模型建立與訓練    使用 Naive Bayes 訓練分類模型                 
7️⃣ | 預測與評估        顯示準確率與分類報告（precision、recall、f1-score） 

