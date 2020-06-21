import numpy as np
from sklearn.feature_extraction import stop_words
import os
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from data_model import *
class Models :
    def __init__(self):
        self.data      = DataModel()
        self.xtrain_q  = self.data.X_train()[0]    
        self.xtrain_qa = self.data.X_train()[1]       
        self.qa        = self.data.getQA()
        self.q_models   = [ #CountVectorizer(), # bow with qa                           
                          TfidfVectorizer(ngram_range=(1,2))                          
                          ]
        self.qa_models  = [ CountVectorizer(),                    
                           TfidfVectorizer(ngram_range=(1,2)), 
                           
                          ]
        self.features_q  =  [m.fit_transform(self.xtrain_q) for m in self.q_models]
        self.features_qa =  [m.fit_transform(self.xtrain_qa) for m in self.qa_models]
    
    def get_q_models(self):
        return self.q_models
        
    def get_qa_models(self):
        return self.qa_models

    def get_q_features (self):
        return self.features_q
    
    def get_qa_features(self):
        return self.features_q
    
    def input_feature(self , q):
        q = [q]
        q = self.data.text_preprocessor(q)
        return [m.transform(q) for m in self.q_models+self.qa_models]
    
    def predict(self,q):
        q_transform  = self.input_feature(q)
        answers   = []
        for model_id, f in enumerate(self.features_q+self.features_qa):
            rank = np.array([cosine_similarity(q_transform[model_id], f[i]) for i in range(self.data.rows)])       
            answers.append(np.argmax(rank))        
        majority = Counter(answers).most_common()[0][0]
        return self.qa[majority][0]


