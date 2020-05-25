import numpy as np
from sklearn.feature_extraction import stop_words
import os
from nltk.corpus import stopwords 
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances 
from sklearn.feature_extraction.text import TfidfVectorizer
from data_model import DataModel
import numpy as np
class TF_IDF:
    def __init__(self):
        self.data = DataModel()
        self.xtrain = self.data.X_train()
        print(self.xtrain.shape)
        self.qa = self.data.getQA()
        self.model = TfidfVectorizer(ngram_range=(1,2))
        self.features = self.model.fit_transform(self.xtrain)
    def get_model(self):
        return self.model
    def get_features (self):
        return self.features
    def input_feature(self , q):
        q = [q]
        q = self.data.text_preprocessor(q)
        return self.model.transform(q)
    def predict(self,q):
      q = self.input_feature(q)
      rank = np.array([cosine_similarity(q,self.features[i]) for i in range(self.data.rows)])
      index = np.argmax(rank)  
      return self.qa[index][1]


