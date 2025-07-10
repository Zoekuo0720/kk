1 安裝套件（jieba、wordcloud 等）	pip install
2	匯入程式庫與讀取 Excel 檔案     	pandas, openpyxl
3	分詞與停用詞處理	                jieba.cut()
4	統計高頻詞並轉為                  DataFrame	collections.Counter
5	建立中文字體文字雲 WordCloud 圖片	WordCloud(font_path=...)
6	顯示圖表                        	matplotlib.pyplot
