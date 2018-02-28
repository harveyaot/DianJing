# coding utf-8
import redis
import json
import pickle
import numpy as np
import random
import jieba
import multiprocessing
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

word2idx, idx2word, allwords, corpus = None, None, {}, []
DUMP_FILE = 'data/basic_data_700k_v2.pkl'
check_sample_size = 10
TF_THRES = 5
DF_THRES = 2

r0 = redis.StrictRedis(host='localhost', port=6379, db=0)
r1 = redis.StrictRedis(host='localhost', port=6379, db=1)

id_beg = 0
id_eos = 1
id_emp = 2
id_unk = 3

r = None


class Word:
    def __init__(self, val, tf, df):
        self.val = val
        self.tf = tf
        self.df = df

    def __repr__(self):
        pass


def parse_all_crawled_data(keys, idx):
    res = []
    if idx == 0:
        conn = r0
    else:
        conn = r1
    for data in conn.mget(keys):
        data = json.loads(data)
        key = data.get("group_id")
        title = data.get("title", "").replace('\t', ' ')
        abstract = data.get("abstract", "").replace('\t', ' ')
        if abstract == "":
            abstract = title
        res.append((key, title, abstract))
    return res


def cal_word_tf_df(corpus):
    words = {}
    title_abstract_pairs = []
    for doc in corpus:
        title, abstract = doc[1].lower(), doc[2].lower()
        ts_ = list(jieba.cut(title, cut_all=False))
        as_ = list(jieba.cut(abstract, cut_all=False))
        title_abstract_pairs.append((ts_, as_))
        # acumulate the term frequency
        for word in ts_ + as_:
            if not words.get(word):
                words[word] = Word(val=word, tf=1, df=0)
            else:
                words[word].tf += 1
        # acummulate the doc frequency
        for word in set(ts_ + as_):
            words[word].df += 1
    return words, title_abstract_pairs


def build_idx_for_words_tf_df(chars, tf_thres=TF_THRES, df_thres=DF_THRES):

    start_idx = id_unk + 1

    char2idx = {}
    idx2char = {}

    char2idx['<eos>'] = id_eos
    char2idx['<unk>'] = id_unk
    char2idx['<emp>'] = id_emp
    char2idx['<beg>'] = id_beg

    chars = filter(lambda char: char.tf >
                   tf_thres or char.df > df_thres, chars)
    char2idx.update(dict([(char.val, start_idx + idx)
                          for idx, char in enumerate(chars)]))
    idx2char = dict([(idx, char) for char, idx in char2idx.items()])
    return char2idx, idx2char


def prt(label, x):
    print label+':',
    for w in x:
        if w == id_emp:
            continue
        print idx2word[w].encode('utf-8'),
    print


def worker(i, keys, idx):
    print "worker [%2d] started with keys:[%d]!" % (i, len(keys))
    corpus = parse_all_crawled_data(keys, idx)
    print "worker [%2d] get docs :[%d]!" % (i, len(corpus))
    words, sub_corpus = cal_word_tf_df(corpus)
    return words, sub_corpus


def combine_results(res):
    global copurs, word2idx, idx2word
    words, sub_corpus = res[0], res[1]
    corpus.extend(sub_corpus)
    for word in words:
        if word not in allwords:
            allwords[word] = Word(val=word, tf=0, df=0)
        allwords[word].tf += words[word].tf
        allwords[word].df += words[word].df
    word2idx, idx2word = build_idx_for_words_tf_df(allwords.values())


def dump_all_results():
    datafile = open(DUMP_FILE, 'wb')
    titles, abstracts = [], []
    for ts_, as_ in corpus:
        titles.append([word2idx.get(word, id_unk) for word in ts_])
        abstracts.append([word2idx.get(word, id_unk) for word in as_])
    pickle.dump((allwords, word2idx, idx2word,
                 titles, abstracts), datafile, -1)


def check_dump():
    global word2idx, idx2word
    allwords, word2idx, idx2word, titles, abstracts = pickle.load(
        open(DUMP_FILE))
    print "allwords size is:", len(allwords)
    print "word2idx size is:", len(word2idx)
    print "titles size is:", len(titles)
    for k in range(check_sample_size):
        k = random.randint(0, len(titles) - 1)
        print "[%s]th Example" % (k)
        prt('title', titles[k])
        prt('abstract', abstracts[k])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DUMP_FILE = sys.argv[1]
        print >> sys.stderr, "[Check] Dump File of [%s]" % DUMP_FILE
    check_dump()
