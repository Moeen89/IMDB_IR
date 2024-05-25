import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path
        

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        corpus = list(Index_reader(self.file_path, Indexes.DOCUMENTS).get_index().values())
        dataframes = pd.DataFrame(corpus)
        dataframes = dataframes.drop(columns=["id", "stars"]) 
        return dataframes

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        data = self.read_data_to_df()
        # concat string of each row to one column
        data['synposis'] = data['synposis'].apply(lambda x: ' '.join(x[i] for i in range(len(x))) if x else "")
        data['summaries'] = data['summaries'].apply(lambda x: ' '.join(x[i] for i in range(len(x))) if x else "")
        data['reviews'] = data['reviews'].apply(lambda x: ' '.join(x[i][0] for i in range(len(x))) if x else "" )
        data['title'] = data['title'].apply(lambda x: x if x else "")
        data['text'] = data['synposis'] + ' ' + data['summaries'] + ' ' + data['reviews'] + ' ' + data['title']
        X = data['text'].to_numpy()
        le = LabelEncoder()
        data['genres'] = data['genres'].apply(lambda x: x[0] if x else "unknown")
        y = le.fit_transform(data['genres'])
        return X, np.array(y)


