1 安裝套件（jieba、wordcloud 等）	pip install
2	匯入程式庫與讀取 Excel 檔案     	pandas, openpyxl
3	分詞與停用詞處理	                jieba.cut()
4	統計高頻詞並轉為                  DataFrame	collections.Counter
5	建立中文字體文字雲 WordCloud 圖片	WordCloud(font_path=...)
6	顯示圖表                        	matplotlib.pyplot
# STEP 1：安裝套件
!pip install jieba wordcloud openpyxl

# STEP 2：匯入必要套件
import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# STEP 3：讀取 Excel 檔案
file_path = '/content/檔案名稱.xlsx'  
df = pd.read_excel(file_path)

# STEP 4：擷取評論欄位並處理缺失值
comments = df['評論內容'].dropna().tolist()

# STEP 5：定義停用詞(可自行增減)
stopwords = set([
    '就是', '這樣', '結果', '還要', '可以', '不是', '沒有', '如果', '一下', '都是', '還是',
    '東西', '自己', '公司', '人事', '有點', '真的', '覺得', '很多', '比較', '應該',
    '全部', '只能', '部分', '整個', '那麼', '非常', '可能', '一下子', '現在', '需要'
])

# STEP 6：進行中文斷詞並移除停用詞與短詞
text = ' '.join(comments)
words = jieba.cut(text)
filtered_words = [word for word in words if word.strip() not in stopwords and len(word.strip()) > 1]
seg_text = ' '.join(filtered_words)

# STEP 7：檢查分詞結果
print("分詞後內容長度：", len(seg_text))
print("前100字：", seg_text[:100])

# STEP 8：統計高頻關鍵詞
word_freq = Counter(filtered_words)
freq_df = pd.DataFrame(word_freq.items(), columns=['關鍵詞', '出現次數']).sort_values(by='出現次數', ascending=False)
print(freq_df.head(20))  # 顯示前 20 名關鍵詞

# STEP 9：設定中文字型路徑
font_path = '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'

# STEP 10：產生文字雲(可自行設定圖片格式)
wordcloud = WordCloud(
    font_path=font_path,
    width=800,
    height=400,
    background_color='white',
    max_words=200
).generate(seg_text)

# STEP 11：顯示文字雲圖
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
