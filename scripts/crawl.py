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


URL = "http://www.toutiao.com/api/pc/feed/?category=%s&utm_source=toutiao&widen=1&max_behot_time=%s&max_behot_time=%s&tadrequire=true&as=A135C940DF6BE90&cp=590FEB5EB9E01E1"
r = redis.StrictRedis(host='localhost', port=6379, db=0)

#sleep time in secs
sleep_time = 5
call_nums = 2000

if len(sys.argv) > 1:
    start_date = sys.argv[1]
    start_timestamp = datetime.datetime.strptime(start_date,"%Y%m%d").strftime("%s")
else:
    start_timestamp = 0

cats = ["news_tech","news_society","news_entertainment","news_sports","news_car",
        "news_finance","news_game","news_world","news_military","news_history",
        "news_fashion","news_baby","news_food","news_health","news_story",
        "news_travel","new_home","__all__"]

def process(cat,timestamp):
    toutiao_data = requests.get(URL%(cat,timestamp,timestamp).text
    data = json.loads(toutiao_data)

    if data.get("message") == "false":
        return 

    articles = data.get("data")
    timestamp = data.get("next").get("max_behot_time")

    for article in articles:
        # remove the invalid record
        if not article.get("group_id") or not article.get("title") or not article.get("abstract"):
            continue
        key = article.get("group_id")
        val = json.dumps(article)
        #print key,article.get("title")
        r.set(key,val)
    return timestamp


def worker(cat,timestamp):
    count = 0 
    while(count < call_nums):
        print "Thread:[%s] num#[%d] t:[%s]"%(cat,count,timestamp)
        gap = random.randint(sleep_time - 2,sleep_time + 2)
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
