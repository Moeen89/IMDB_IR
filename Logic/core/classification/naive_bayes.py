import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        self.num_classes = len(np.unique(y))
        self.classes = np.unique(y)
        self.number_of_samples, self.number_of_features = x.shape
        self.prior = np.zeros(self.num_classes)
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))

        for i, c in enumerate(self.classes):
            x_c = x[y == c]
            self.prior[i] = x_c.shape[0] / self.number_of_samples
            self.feature_probabilities[i] = (np.sum(x_c, axis=0) + self.alpha) / (np.sum(x_c) + self.alpha)

        self.log_probs = np.log(self.feature_probabilities)
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        return np.argmax(np.dot(x, self.log_probs.T) + np.log(self.prior), axis=1)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        

        


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    loader = ReviewLoader("../Logic/core/classification/IMDB Dataset.csv")
    x, y = loader.load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    cv = CountVectorizer(max_features=25000)
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)
    nb = NaiveBayes(cv)
    nb.fit(x_train.toarray(), y_train)
    print(nb.prediction_report(x_test.toarray(), y_test))

