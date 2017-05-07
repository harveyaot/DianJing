#coding:utf-8   
import sys
import requests
import json
import time
import random

import redis


# the main page refreh api 
#Url = "http://www.toutiao.com/api/pc/feed/?category=__all__&utm_source=toutiao"
Url = "http://www.toutiao.com/api/pc/feed/?category=news_hot&utm_source=toutiao&widen=1&max_behot_time=%s"
r = redis.StrictRedis(host='localhost', port=6379, db=0)
timestamp = "1494053191"

def process():
    global timestamp
    toutiao_data = requests.get(Url%(timestamp)).text
    data = json.loads(toutiao_data)

    if data.get("message") == "false":
        return 

    articals = data.get("data")
    timestamp = data.get("next").get("max_behot_time")

    for artical in articals:
        if not artical.get("group_id") or not artical.get("title") or not artical.get("abstract"):
            continue
        key = artical.get("group_id")
        print key,artical.get("title")
        #r.set(key,artical)

def crawl():
    count = 0 
    while(count < 20000):
        gap = random.randint(4,7)
        time.sleep(gap)
        process()
        time.sleep(gap)
        print timestamp
        count += 1
        print "crawl num #[%d]"%(count)

if __name__ == "__main__":
    crawl()
