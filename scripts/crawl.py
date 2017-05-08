#coding:utf-8   
import sys
import requests
import json
import time
import random
import datetime
import threading
import multiprocessing

import redis


# the main page refreh api 
#Url = "http://www.toutiao.com/api/pc/feed/?category=__all__&utm_source=toutiao"
#Url = "http://www.toutiao.com/api/pc/feed/?category=news_hot&utm_source=toutiao&widen=1"

Url = "http://www.toutiao.com/api/pc/feed/?category=%s&utm_source=toutiao&widen=1&max_behot_time=%s&max_behot_time=%s&tadrequire=true&as=A135C940DF6BE90&cp=590FEB5EB9E01E1"
r = redis.StrictRedis(host='localhost', port=6379, db=0)

if len(sys.argv) > 1:
    start_date = sys.argv[1]
    start_timestamp = datetime.datetime.strptime(start_date,"%Y%m%d").strftime("%s")
else:
    start_timestamp = 0

cats = ["news_tech","news_society","news_entertainment","news_sports","news_car",
        "news_finance","news_game","news_world","news_military","news_history",
        "news_fashion","__all__"
        ]
cats2 = ["news_baby","news_food","news_health","news_story","news_travel"]
cats3 = ["news_home"]
cats = cats3

def process(cat,timestamp):
    toutiao_data = requests.get(Url%(cat,timestamp,timestamp)).text
    data = json.loads(toutiao_data)

    if data.get("message") == "false":
        return 

    articles = data.get("data")
    #timestamp = (datetime.datetime.fromtimestamp(int(timestamp)) - datetime.timedelta(minutes=15)).strftime("%s")
    timestamp = data.get("next").get("max_behot_time")

    for article in articles:
        if not article.get("group_id") or not article.get("title") or not article.get("abstract"):
            continue
        key = article.get("group_id")
        val = json.dumps(article)
        #print key,article.get("title")
        r.set(key,val)
    return timestamp


def worker(cat,timestamp):
    count = 0 
    while(count < 20000):
        print "Thread:[%s] num#[%d] t:[%s]"%(cat,count,timestamp)
        gap = random.randint(7,10)
        time.sleep(gap)
        timestamp = process(cat,timestamp)
        count += 1
    

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=len(cats))
    for cat in cats:
        pool.apply_async(worker, (cat, start_timestamp))
    pool.close()
    pool.join()
    print "------all jobs finished!------"
