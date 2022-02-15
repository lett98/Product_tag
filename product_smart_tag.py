#input: product_name from ghtk_database
#output: product_tag & threshold
#connect to ghtk_db by trino
#import packages
from email import charset
import pandas as pd
import re
from nltk.tokenize import MWETokenizer,word_tokenize
import unicodedata
import codecs
from fasttext import load_model
from sqlalchemy.engine import create_engine
from pandas.io import sql
from sqlalchemy import create_engine
import pymysql


def get_input(query):
    engine = create_engine("trino://user:password@trino.ghtk.vn:443/hive")
    connection = engine.connect()
    return pd.read_sql(query, connection)


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
  for line in enumerate(f):
      line = repr(line)
      line = line[1:len(line)-5]
      data.append(line)
  return data
stopWords = create_stopwordlist()

def remove_stopwords(text):
  for w in stopWords:
    pattern = r'\b'+w+r'\b'
    text = re.sub(pattern, '', text)
  return text

#preprocess
def preprocess(text):
  if type(text) != str:
    return ""
  else:
    text = unicodedata.normalize("NFC", text)
    text = remove_emoji(text)
    text = multiple_punctuation_pattern.sub(r" \g<1> ", text)
    text = normalize_text(text.lower())
    text = remove_stopwords(text.lower())
    text = re.sub('(\()[\s\w]*(\))', '', text)
    text = re.sub('(\[)[\s\w\%\d]+(\])', '', text)
    text = re.sub('[^\w]+', ' ', text)
    text = word_tokenizer.tokenize(word_tokenize(text))
    return ' '.join(text)
    
# Dictionary to store the result
results = []
def process(df):
    # load model from local
    fast_model = load_model("fasttext_model.bin")
    for i in range(0,len(df)):
        product_name = df['product_name'].loc[i]
        text = preprocess(product_name)
        if text == "" or text ==" " or re.sub('\d+','', text) == "":
            threshold = -1
            label_threshold = ""
        else:
            predict_result = fast_model.predict(text)
            threshold = predict_result[1][0]
            label_threshold= predict_result[0][0]
        results.append([df["shop_code"][i], df["pkg_order"][i],product_name,label_threshold,threshold])
    return results

def converse_output(row):
    if row["threshold"] == -1:
        return "Không xác định"
    elif row["threshold"] >= 0.7:##threshold boundary
        return row["label_desc"]
    return "Hàng khác"  
  
def execute_values(data):
    con = pymysql.connect(host="host", user="user", passwd="password", db="database")
    cursor = con.cursor()

    df = pd.DataFrame(data)
    val_to_insert = df.values.tolist()

    cursor.executemany("INSERT INTO `product-smart-tag` (`shop_code`, `pkg_order`, `product_name`,`category_id`,`category`,`predict_category`,`threshold`) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    val_to_insert)

    con.commit()
    con.close()
    print('Sucessfully written to Database!!!')



if __name__ == '__main__':
    #input
    query = """Select * FROM ghtk_product_name"""
    input = get_input(query)
    #process
    predict_result = process(input)
    predict_output = pd.DataFrame(predict_result, columns=["shop_code","pkg_order","product_name","category", "threshold"])
    #description
    label_desc = pd.read_excel('label_desc.xlsx')
    sub_result = pd.merge(predict_output,label_desc, on = "category", how= "left")
    final_result = sub_result[["shop_code","pkg_order","product_name","category","threshold","label_desc"]]
    final_result["predict_category"] = final_result.apply(lambda row: converse_output(row), axis = 1)
    #output
    output = final_result[["shop_code","pkg_order","product_name","category","label_desc","predict_category","threshold"]]
    execute_values(output)
    print("SUCCESS")
