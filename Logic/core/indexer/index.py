import time
import os
import json
import copy
from Logic.core.utility.preprocess import Preprocessor
from Logic.core.indexer.indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for document in self.preprocessed_documents:
            current_index[document['id']] = document
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for document in self.preprocessed_documents:
            stars = document['stars']
            if stars is None:
                continue
            for star in stars:
                if star not in current_index:
                    current_index[star] = {}
                if document['id'] not in current_index[star]:
                    current_index[star][document['id']] = 0
                current_index[star][document['id']] += 1
        return current_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for document in self.preprocessed_documents:
            genres = document['genres']
            if genres is None:
                continue
            for genre in genres:
                if genre not in current_index:
                    current_index[genre] = {}
                if document['id'] not in current_index[genre]:
                    current_index[genre][document['id']] = 0
                current_index[genre][document['id']] += 1
        return current_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for document in self.preprocessed_documents:
            summaries = document['summaries']
            if summaries is None:
                continue
            for summary in summaries:
                for word in summary.split():
                    if word not in current_index:
                        current_index[word] = {}
                    if document['id'] not in current_index[word]:
                        current_index[word][document['id']] = 0
                    current_index[word][document['id']] += 1    

                
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            current_index = self.index[index_type]
            return list(current_index[word].keys())
        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        self.preprocessed_documents.append(document)
        self.index[Indexes.DOCUMENTS.value][document['id']] = document
        for star in document['stars']:
            if star not in self.index[Indexes.STARS.value]:
                self.index[Indexes.STARS.value][star] = {}
            if document['id'] not in self.index[Indexes.STARS.value][star]:
                self.index[Indexes.STARS.value][star][document['id']] = 0
            self.index[Indexes.STARS.value][star][document['id']] += 1

        for genre in document['genres']:
            if genre not in self.index[Indexes.GENRES.value]:
                self.index[Indexes.GENRES.value][genre] = {}
            if document['id'] not in self.index[Indexes.GENRES.value][genre]:
                self.index[Indexes.GENRES.value][genre][document['id']] = 0
            self.index[Indexes.GENRES.value][genre][document['id']] += 1

        for summary in document['summaries']:
            if summary not in self.index[Indexes.SUMMARIES.value]:
                self.index[Indexes.SUMMARIES.value][summary] = {}
            if document['id'] not in self.index[Indexes.SUMMARIES.value][summary]:
                self.index[Indexes.SUMMARIES.value][summary][document['id']] = 0
            self.index[Indexes.SUMMARIES.value][summary][document['id']] += 1

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """
        if document_id in self.index[Indexes.DOCUMENTS.value]:
            del self.index[Indexes.DOCUMENTS.value][document_id]
        for star in self.index[Indexes.STARS.value]:
            if document_id in self.index[Indexes.STARS.value][star]:
                del self.index[Indexes.STARS.value][star][document_id]
        for genre in self.index[Indexes.GENRES.value]:
            if document_id in self.index[Indexes.GENRES.value][genre]:
                del self.index[Indexes.GENRES.value][genre][document_id]
        for summary in self.index[Indexes.SUMMARIES.value]:
            if document_id in self.index[Indexes.SUMMARIES.value][summary]:
                del self.index[Indexes.SUMMARIES.value][summary][document_id]

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(
                set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(
                set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(
                set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(
                set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(
                set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        json_object = json.dumps(self.index[index_name])
        with open(path, 'w') as f:
            f.write(json_object)

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        with open(path, 'r') as f:
            loaded_index = json.load(f)
        self.index[path[6:-11]] = loaded_index

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time <= brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False


# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
#
# with open('../IMDB_crawled.json', 'r') as f:
#     json_data = f.read()
# crawled_movies = json.loads(json_data)
# for movie in crawled_movies:
#     for item in ['stars', 'genres', 'summaries']:
#         movie[item] = Preprocessor(movie[item]).preprocess()
#
# index = Index(crawled_movies)
# for item in ['documents', 'stars', 'genres', 'summaries']:
#     print(item, ": ")
#     index.check_if_indexing_is_good(item)
# for item in ['documents', 'stars', 'genres', 'summaries']:
#     path = 'index/' + item + '_index.json'
#     index.store_index(path, item)
#
# for item in ['documents', 'stars', 'genres', 'summaries']:
#     path = 'index/' + item + '_index.json'
#     index.load_index(path)
#     print(index.check_if_index_loaded_correctly(item, index.index[item]))
