import numpy as np
import sklearn
from sklearn.metrics import classification_report
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__(n_components=100)
        self.k = n_neighbors
        self.x = None
        self.y = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        x = self.pca.fit_transform(x)
        self.y = y
        self.x = x

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
        x = self.pca.transform(x)
        y_pred = []
        for i in tqdm(range(x.shape[0])):
            distances = np.linalg.norm(self.x - x[i], axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y[nearest_indices]
            y_pred.append(np.bincount(nearest_labels).argmax())
        return np.array(y_pred)

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


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader("../Logic/core/classification/IMDB Dataset.csv")
    loader.load_data()
    loader.get_embeddings()
    x_train, x_test, y_train, y_test = loader.split_data()
    classifier = KnnClassifier(n_neighbors=5)
    classifier.fit(x_train, y_train)
    print(classifier.prediction_report(x_test, y_test))
