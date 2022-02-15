import pandas as pd
import re
from nltk.tokenize import MWETokenizer,word_tokenize
import unicodedata
import codecs
from sklearn.model_selection import train_test_split

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

def import_data():
    tag = pd.read_excel('bow_tag.xlsx')
    mapping_label = pd.read_excel('product_category.xlsx')
    crawl_data = pd.read_csv('crawl_shopee_output.csv')
    crawl_category = pd.merge(crawl_data,mapping_label, left_on="2ndname", right_on="sub_category")
    #transform category
    crawl_df = crawl_category[["category","product_name"]]
    tag_data = tag[["category","product_name"]]
    #concat
    data_raw = pd.concat([crawl_df, tag_data], ignore_index=True)
    return data_raw
  
  
#Preprocess
# Remove emoji
def remove_emoji(string):
    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002500-\U00002BEF"  # chinese char
                    u"\U00002702-\U000027B0"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    u"\U0001f926-\U0001f937"
                    u"\U00010000-\U0010ffff"
                    u"\u2640-\u2642"
                    u"\u2600-\u2B55"
                    u"\u200d"
                    u"\u23cf"
                    u"\u23e9"
                    u"\u231a"
                    u"\ufe0f"  # dingbats
                    u"\u3030"
                    "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

# Normalize về chuẩn dấu câu tiếng việt
normalizer = {'òa': 'oà',
              'óa': 'oá',
              'ỏa': 'oả',
              'õa': 'oã',
              'ọa': 'oạ',
              'òe': 'oè',
              'óe': 'oé',
              'ỏe': 'oẻ',
              'õe': 'oẽ',
              'ọe': 'oẹ',
              'ùy': 'uỳ',
              'úy': 'uý',
              'ủy': 'uỷ',
              'ũy': 'uỹ',
              'ụy': 'uỵ',
              'Ủy': 'Uỷ'}
multiple_punctuation_pattern = re.compile(r"([\"\.\?\!\,\:\;\-])(?:[\"\.\?\!\,\:\;\-]){1,}")
word_tokenizer = MWETokenizer(separator='')
def normalize_text(text):
      for absurd, normal in normalizer.items():
        text = text.replace(absurd, normal)
        return text
#stopword
def create_stopwordlist():
  f = codecs.open('stop_words.text', encoding='utf-8')
  data = []
  for i, line in enumerate(f):
      line = repr(line)
      line = line[1:len(line)-5]
      data.append(line)
  return  "|".join(data)
stopWords = create_stopwordlist()

#preprocess
def preprocess(text):
  if type(text) != str:
    return ""
  else:
    text = unicodedata.normalize("NFC", text)
    text = remove_emoji(text)
    text = multiple_punctuation_pattern.sub(r" \g<1> ", text)
    text = normalize_text(text.lower())
    text = re.sub(stopWords,"", text)
    text = re.sub('(\()[\s\w]*(\))', '', text)
    text = re.sub('(\[)[\s\w\%\d]+(\])', '', text)
    text = re.sub('[^\w]+', ' ', text)
    text = word_tokenizer.tokenize(word_tokenize(text))
    return ' '.join(text)
 

def tag_other(row):
    if row["product_name"] == "" or row["product_name"] ==" ":
        return 0
    return 1

data_raw = import_data()
data_raw["product_name"] = data_raw["product_name"].apply(preprocess)
data_raw["is_text"] = data_raw.apply(lambda row: tag_other(row), axis = 1)
# chỉ lấy khi còn product_name & bỏ duplicate
data = data_raw.loc[data_raw["is_text"] == 1]
df = data.drop_duplicates()
print("raw:",data_raw.shape)
print("before:",data.shape)
print("after:",df.shape)
train, test = train_test_split(df, test_size=0.2,random_state=1234)
train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)
