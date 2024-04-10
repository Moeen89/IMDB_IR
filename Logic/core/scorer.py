import numpy as np


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            if term not in self.index:
                return 0
            df = len(self.index.get(term, {}))
            if df == 0:
                idf = 0
            else:
                idf = np.log(self.N / df)
        self.idf[term] = idf
        return idf

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """

        tfs = {}
        for term in query:
            tfs[term] = tfs.get(term, 0) + 1
        return tfs

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        tf_query = self.get_query_tfs(query)
        list_of_documents = self.get_list_of_documents(query)
        scores = {}
        for document in list_of_documents:
            scores[document] = self.get_vector_space_model_score(query, tf_query, document, method[:3], method[4:])
        return scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        query_vector = []
        document_vector = []
        query_words = list(set(query))
        for i, word in enumerate(query_words):
            if word in self.index.keys():
                if query_method[0] == 'n':
                    query_vector.append(query_tfs[word])
                elif query_method[0] == 'l':
                    query_vector.append(1 + np.log(query_tfs[word]))
                if document_method[0] == 'n':
                    document_vector.append(self.index[word].get(document_id, 0))
                elif document_method[0] == 'l':
                    tf_tmp = self.index[word].get(document_id, 0)
                    if tf_tmp == 0:
                        document_vector.append(0)
                    else:
                        document_vector.append(1 + np.log(tf_tmp))
                if query_method[1] == 't':
                    query_vector[-1] *= self.get_idf(word)
                if document_method[1] == 't':
                    document_vector[-1] *= self.get_idf(word)
        if query_method[2] == 'c':
            query_vector = np.array(query_vector)
            query_vector = query_vector / np.linalg.norm(query_vector)

        if document_method[2] == 'c':
            document_vector = np.array(document_vector)
            document_vector = document_vector / np.linalg.norm(document_vector)

        return np.dot(np.array(query_vector), np.array(document_vector))

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        documnets = self.get_list_of_documents(query)
        scores = {}
        for document in documnets:
            scores[document] = self.get_okapi_bm25_score(query, document, average_document_field_length,
                                                         document_lengths)
        return scores

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        score = 0
        query_words = list(set(query))
        k1 = 1.5
        b = 0.75
        for i, word in enumerate(query_words):
            if word in self.index.keys():
                tf = self.index[word].get(document_id, 0)
                idf = self.get_idf(word)
                document_length = document_lengths[document_id]
                score += idf * (
                        (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * document_length / average_document_field_length)))
        return score
