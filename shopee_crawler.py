import requests
import json
import pandas as pd
import time
import random

OUTPUT_FILE = "crawl_shopee_output.csv"

def get_product_id():
    # lick shopee
    with open('shopee_link.json', encoding='utf-8') as f:
        kv = json.load(f)
    return kv
  
def get_data():
    #get 1000 samples for each product
    limit_number = 100 # number of product
    pages = [ '?page={}'.format(i) for i in range(1,10)]# 10 pages
    url = 'https://shopee.vn/api/v4/search/search_items?by=relevancy&limit={}&match_id={}&newest=60&order=desc&page_type=search&scenario=PAGE_OTHERS&version=2'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
    }

    kv = get_product_id()
    k_v = {k: v for k,v in kv.items() if k in list(kv.keys())}
    for kk,vv in k_v.items():
        for name, url_link in kv[kk].items():
            print(kk, '/', name)
            links = []
            links.append(url_link)

            match_id = url_link.split('.')[-1]

            url_request = url.format(limit_number, match_id)
            for page in pages:
                links.append(url_link + page)
                
            result_name = []
            for link in links:
                headers['Referer'] = link
                r = requests.get(url_request, headers=headers)
                data = r.json()
                for i in range(0,len(data['items'])): ##100 product
                    if i <= 100:
                        result_name.append((kk, name, data['items'][i]["item_basic"]["name"]))
                    else:
                        break
                time.sleep(random.randrange(5,20))
            return result_name

data = get_data()
df = pd.DataFrame(data, columns=['1stname', '2ndname', 'product_name'])
final_data = df[["1stname","product_name"]].drop_duplicates()
final_data.to_csv(OUTPUT_FILE, index=False)
print("Done!!!")
