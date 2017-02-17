import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from collections import Counter
import gensim, logging
import csv
import jieba
import jieba.posseg as pseg
from gensim.models import word2vec
import logging

global model
w2v_column_num = 200

def get_student_book(bookname):
    booklist = list()
    for i in bookname:
        name = i.strip('"').split()
        booklist.append(name[0])
    return ' '.join(booklist)

def jieba_cut(booklist):
    seg_list = jieba.cut(booklist)
    return ' '.join(seg_list)

def get_w2v_features(booklist):
    num = 0
    arr = np.zeros((1,w2v_column_num)).reshape(-1)
    for i in booklist.split():
        if i in model.wv.vocab:
            num += 1
            arr = np.add(arr, model[i])
        else:
            continue
    res = np.divide(arr, num)
    return res

def get_w2v_columns(num):
    name_list = list()
    for i in range(num):
        name_list.append('w2v_'+ str(i))
    return name_list

column_name = ['ID', 'BORROW_DATE', 'BOOK_NAME', 'BOOK_ID', 'EXTRA']
#borrow = pd.read_csv('../data/input/borrow_train_test.txt', names = column_name)

borrow = pd.read_csv('../data/input/borrow_train_final_test.txt', names = column_name)

st_bookname = pd.DataFrame(borrow['BOOK_NAME'].groupby(borrow['ID']).apply(get_student_book))

st_bookname.to_csv('../data/analysis/st_bookname0213.csv', index = True, encoding="utf-8")
st_bookname = pd.read_csv('../data/analysis/st_bookname0213.csv')

st_bookname['BOOK_NAME'] = st_bookname['BOOK_NAME'].apply(jieba_cut)
st_bookname['BOOK_NAME'].to_csv('../data/analysis/corpus0213.csv', index = False, header = False, encoding = 'utf-8')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('../data/analysis/corpus0213.csv')
model = word2vec.Word2Vec(sentences, size = w2v_column_num, min_count = 1)
model.save('../data/model/w2v_model0213.model')

res = st_bookname['BOOK_NAME'].apply(get_w2v_features)
w2v_column_name = get_w2v_columns(w2v_column_num)

w2v_features = pd.DataFrame.from_records(res, columns = w2v_column_name)
w2v_features['ID'] = st_bookname['ID']

w2v_features.to_csv('../data/input/w2v_features0213.csv', index = False)