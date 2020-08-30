import numpy as np
import pandas as pd
#import MeCab
from konlpy.tag import *
from gensim.models import Word2Vec, fasttext
import matplotlib.pyplot as plt
from eunjeon import Mecab
#shell -> pip install eunjeon --user

class FE_W2V:
    def __init__(self, df):
        self.df = df
        self.tagger = Mecab()
        self.tag_N = ["NNG", "NNP", "NNB", " NNBC", "NR", "NP","SL","SN"]
    
    def make_corpus_M(self):
        corpus = []
        #토크나이징
        for i in range(len(self.df)):
            try:
                corpus.append(['/'.join(p) for p in self.tagger.pos(self.df['NEW상품명'].iloc[i]) if p[1] in self.tag_N])
            except :
                pass
        return corpus

    def zero_pad_from_2Darray_R(self, aa, fixed_length, padding_value=0):
        rows = []
        for a in aa:
            rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
        return np.concatenate(rows, axis=0).reshape(-1, fixed_length)
    
    def product_name_embedding_ver1(self, df_name = None,w2v_m = "skip", dim = 10, win = 3,min_cnt = 2):
    #using MeCab(), Just Concatenate, zero-padding (right)
        df_ver1 = df_name if df_name is not None else self.df

        #assert df_ver1 != None

        corpus = self.make_corpus_M()
        #print(corpus)
        #print(corpus)
        
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
        for j in range(len(self.df)) : 
            vec_con = []
            try : 
                test = ['/'.join(p) for p in self.tagger.pos(self.df['NEW상품명'].iloc[j]) if p[1] in self.tag_N]
                for k in (test):
                    try :
                        vec_con.extend(vectors[words.index(k)]) #w2v으로 임베딩된 토큰만 concate
                    except : 
                        pass
            except: 
                pass
            vec_em.append(vec_con) # # of data(list) -> (dim*토큰개수,)vec
            vec_size.append(len(vec_con))

        max_vec_dim = max(vec_size) #embedding dim
        assert len(vec_em) == len(self.df)
        feature_name = ['v' + str(i) for i in range(max_vec_dim)]


        p_name_embedded = self.zero_pad_from_2Darray_R(vec_em, max_vec_dim)
        
        #make dataframe
        vector_df_tmp = pd.DataFrame(p_name_embedded,columns = feature_name)
        df_name_tmp = self.df["NEW상품명"].reset_index(drop = True)
        vector_df = pd.merge(df_name_tmp,vector_df_tmp.drop_duplicates(),left_index = True, right_index = True)
        df_ver1 = pd.merge(self.df,vector_df.drop_duplicates(),on = "NEW상품명",how = "left")
   
        return df_ver1
        

    def product_name_embedding_ver2(self, df_name = None, w2v_m = "skip", dim = 10, win = 3,min_cnt = 2):
    #using MeCab(), product name embedding = mean of tocken vectors
        df_ver2 = df_name if df_name is not None else self.df
        #assert df_ver2 != None

        corpus = self.make_corpus_M()
        
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

        for j in range(len(self.df)) : 
            vec_con = np.zeros((dim))
            cnt = 0

            try :
                test = ['/'.join(p) for p in self.tagger.pos(self.df['NEW상품명'].iloc[j]) if p[1] in self.tag_N]
                for k in (test):
                    try :
                        vec_con += np.asarray(vectors[words.index(k)]) 
                        cnt += 1
                    except : 
                        pass
            except: 
                pass

            if cnt != 0:
                vec_con = vec_con/(cnt) #상품명 임베딩 = 토큰들의 임베딩 벡터 평균

            vec_em.append(vec_con) # # of data(list) -> (dim*토큰개수,)vec
            vec_size.append(len(vec_con))

        assert len(vec_em) == len(self.df)
        feature_name = ['v' + str(i) for i in range(dim)]
        #make dataframe
        vector_df_tmp = pd.DataFrame(np.asarray(vec_em),columns = feature_name)
        df_name_tmp = self.df["NEW상품명"].reset_index(drop = True)
        vector_df = pd.merge(df_name_tmp,vector_df_tmp.drop_duplicates(),left_index = True, right_index = True)
        df_ver2 = pd.merge(self.df,vector_df.drop_duplicates(),on = "NEW상품명",how = "left")

        return df_ver2

        