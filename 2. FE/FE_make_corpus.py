import re
import numpy as np
import pandas as pd
import re
#import MeCab
from konlpy.tag import *
from gensim.models import Word2Vec, fasttext
import matplotlib.pyplot as plt
from eunjeon import Mecab
#shell -> pip install eunjeon --user

def make_corpus_our(df):
    result_list = []
    for i in range(len(df)):
        tmp = []
        
        if type(df["브랜드"].iloc[i]) != float:
            tmp.extend([df["브랜드"].iloc[i]])
        if type(df["상품명다시"].iloc[i]) != float:
            tmp_p = re.sub('[-=.#)/?:($}+]','',df["상품명다시"].iloc[i])
            tmp.extend(tmp_p.split())
        if type(df["단위"].iloc[i]) != float:
            try :
                tmp_p = re.sub('[-=.#)/?:($}+]','',df["단위"].iloc[i])
                tmp.extend(tmp_p.split())
            except :
                tmp.extend(str(df["단위"].iloc[i]))
        
        if len(tmp) != 0:
            result_list.append(tmp)
    return result_list

def make_corpus_M(df):
    tag_N = ["NNG", "NNP", "NNB", " NNBC", "NR", "NP","SL","SN"]
    tagger = Mecab()
    corpus = []
    #토크나이징
    for i in range(len(df)):
        try:
            corpus.append(['/'.join(p) for p in tagger.pos(df['NEW상품명'].iloc[i]) if p[1] in tag_N])
        except :
            pass
    return corpus