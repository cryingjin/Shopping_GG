import numpy as np
import pandas as pd
#import MeCab
from konlpy.tag import MeCab
from gensim.models import Word2Vec, fasttext
import matplotlib.pyplot as plt
#from eunjeon import Mecab
#shell -> pip install eunjeon --user

tagger = Mecab()
tag_N = ["NNG", "NNP", "NNB", " NNBC", "NR", "NP","SL","SN"]

def make_corpus_M(df_for_corpus):
    corpus = []
    #토크나이징
    for i in range(len(df_for_corpus)):
        corpus.append(['/'.join(p) for p in tagger.pos(df_for_corpus['NEW상품명'].loc[i]) if p[1] in tag_N])
    return corpus

def zero_pad_from_2Darray_R(aa, fixed_length, padding_value=0):
    rows = []
    for a in aa:
        rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
    return np.concatenate(rows, axis=0).reshape(-1, fixed_length)

def product_name_embedding_ver1(df, w2v_m = "skip", dim = 10, win = 3,min_cnt = 2):
    #using MeCab(), Just Concatenate, zero-padding (right)
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
        test = ['/'.join(p) for p in tagger.pos(df['NEW상품명'].iloc[j]) if p[1] in tag_N]
        for k in (test):

            try :
                vec_con.extend(vectors[words.index(k)]) #w2v으로 임베딩된 토큰만 concate
            except : 
                pass
        vec_em.append(vec_con) # # of data(list) -> (dim*토큰개수,)vec
        vec_size.append(len(vec_con))

    max_vec_dim = max(vec_size) #embedding dim
    assert len(vec_em) == len(df)
    feature_name = ['v' + str(i) for i in range(max_vec_dim)]


    p_name_embedded = zero_pad_from_2Darray_R(vec_em, max_vec_dim)
    
    #make dataframe
    vector_df = pd.DataFrame(p_name_embedded,columns = feature_name)
    df = pd.merge(df,vector_df,left_index = True, right_index = True)



    return df

def product_name_embedding_ver2(df, w2v_m = "skip", dim = 10, win = 3,min_cnt = 2):
    #using MeCab(), product name embedding = mean of tocken vectors
   
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
        vec_con = np.zeros((dim))
        test = ['/'.join(p) for p in tagger.pos(df['NEW상품명'].iloc[j]) if p[1] in tag_N]
        cnt = 0
        for k in test: 
            try :#w2v으로 임베딩된 토큰만
                vec_con += np.asarray(vectors[words.index(k)]) 
                cnt += 1
            except : 
                pass
        if cnt != 0:
            vec_con = vec_con/(cnt) #상품명 임베딩 = 토큰들의 임베딩 벡터 평균

        vec_con = vec_con.tolist()
        vec_em.append(vec_con) # # of data(list) -> (dim*토큰개수,)vec
        

    assert len(vec_em) == len(df)
    feature_name = ['v' + str(i) for i in range(dim)]
    #make dataframe
    vector_df = pd.DataFrame(vec_em,columns = feature_name)
    df = pd.merge(df,vector_df,left_index = True, right_index = True)

    return df