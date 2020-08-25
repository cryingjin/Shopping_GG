import numpy as np
import pandas as pd
#import MeCab
from konlpy.tag import *
from gensim.models import Word2Vec, fasttext
import matplotlib.pyplot as plt
from eunjeon import Mecab
#shell -> pip install eunjeon --user

def make_corpus_M(df_for_corpus):
    corpus = []
    tag_N = ["NNG", "NNP", "NNB", " NNBC", "NR", "NP","SL","SN"]
    #토크나이징
    tagger = Mecab()


    for i in range(len(df_for_corpus)):
        corpus.append(['/'.join(p) for p in tagger.pos(df_for_corpus['NEW상품명'].loc[i]) if p[1] in tag_N])
        
    return corpus

def zero_pad_from_2Darray_R(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length)


def product_name_embedding(df, w2v_m = "skip", dim = 40, win = 3,min_cnt = 2):
    
    corpus = make_corpus_M(df)

    if w2v_m == "skip" :
        Skip_Gram_model = Word2Vec(corpus, size=dim, window=win, min_count=min_cnt, workers=1, iter=500, sg=1)
        words = Skip_Gram_model.wv.index2word #one-hot encoding알아서 해줌 
        vectors = Skip_Gram_model.wv.vectors
        
    else : 
        CBOW_model = Word2Vec(corpus, size=dim, window=win, min_count=min_cnt, workers=1, iter=500  , sg=0)
        words = CBOW_model.wv.index2word #one-hot encoding알아서 해줌 
        vectors = CBOW_model.wv.vectors
    
    vec_size = []
    vec_em = []
    for j in range(len(df)) : 
        vec_con = []
        test = ['/'.join(p) for p in tagger.pos(df['상품명다시'].iloc[j]) if p[1] in tag_N]
        for k in (test):
            try :
                vec_con.extend(vectors[words.index(k)]) #w2v으로 임베딩된 토큰만 concate
            except : 
                pass
        vec_em.append(vec_con) # # of data(list) -> (dim*토큰개수,)vec
        vec_size.append(len(vec_con))

    """

     for j in range(len(df)) : 
        vec_con = []
        vec_em = []
        test = ['/'.join(p) for p in tagger.pos(df['상품명다시'].iloc[j]) if p[1] in tag_N]
        for k in (test):
            try :
                vec_con.append(vectors[words.index(k)]) #w2v으로 임베딩된 토큰만 concate
            except : 
                pass
        vec_em.append(vec_con) # # of data(list) -> n개 토큰(list) -> (40,)vec
        vec_size.append(len(vec_con))

    max_vec_dim = max(vec_size)*dim #embedding dim
    """

    max_vec_dim = max(vec_size) #embedding dim
    assert len(vec_em) == len(df)

    p_name_embedded = zero_pad_from_2Darray_R(vec_em, max_vev_dim)

    return p_name_embedded
