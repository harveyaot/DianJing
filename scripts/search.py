# coding:utf-8   
# call the toutiao search api to crawl articles
import sys
import requests
import json
import time
import random
import datetime
import threading
import multiprocessing

import redis
import argparse


r = redis.StrictRedis(host='localhost', port=6379, db=1)

#sleep time in secs
sleep_time = 5

def process(key):
    toutiao_data = requests.get(URL%(key)).text
    data = json.loads(toutiao_data)

    if data.get("return_count","0") > "0":
        return 

    articles = data.get("data")

    for article in articles:
        # remove the invalid record
        if not article.get("group_id") or not article.get("title"):
            continue
        key = article.get("group_id")
        val = json.dumps(article)
        #print key,article.get("title")
        r.set(key,val)


def myworker(sub_keywords):
    for key in sub_keywords:
        print key
        time.sleep(sleep_time)
        process(key)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--end_date",
                        )

    parser.add_argument("-n","--worker_size",
                        default = 10, 
                        type = int)

    parser.add_argument("-k","--keyword_file",
                        default = "keywords.select", 
                        help = "the keywords file to use")

    args = parser.parse_args()

    if args.end_date:
        end_date = args.end_date
        end_timestamp = datetime.datetime.strptime(end_date,"%Y%m%d").strftime("%s")
    else:
        end_timestamp = 0

    URL = "http://www.toutiao.com/search_content/?offset=20&format=json&keyword=%s&autoload=true&count=100"
    pool = multiprocessing.Pool()
    keywords = [line.split('\t')[0] for line in open(args.keyword_file,'r').readlines()]
    batch = len(keywords) / args.worker_size
    for i in range(args.worker_size):
        if i == args.worker_size - 1:
            sub_keywords = keywords[i * batch : ]
        else:
            sub_keywords = keywords[i * batch : i * batch + batch]
        pool.apply_async(myworker, (sub_keywords,))
    pool.close()
    pool.join()
    print "------all jobs finished!------"
