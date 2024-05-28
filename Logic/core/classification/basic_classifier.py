import numpy as np
import sklearn
from tqdm import tqdm

from ..word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self, n_components=50):
        self.pca = sklearn.decomposition.PCA(n_components=n_components)


    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def prediction_report(self, x, y):
        pass

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        positive_reviews = 0
        ft_model = FastText(method='skipgram')
        ft_model.prepare(None,mode="load")

        for sentence in sentences:
            emb = ft_model.get_query_embedding(sentence)   
            emb = self.pca.transform([emb])
            if self.predict([emb]) == 1:
                positive_reviews += 1
        return positive_reviews / len(sentences)
       

