import numpy as np
import itertools
import random
import json


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.shingles = None

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = None
        words = document.split()
        shingles = set()
        for i in range(len(words) - k+1):
            shingle = ''
            for j in range(k):
                shingle += words[i + j]
                shingle += ' '
            shingle = shingle.strip()
            shingles.add(shingle)
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        shingles_doc = []
        for doc in self.documents:
            shingles_doc.append(self.shingle_document(doc))
        self.shingles = list(set.union(*shingles_doc))
        characteristic_matrix = np.ndarray(shape=(len(self.shingles),len(self.documents)),dtype=np.int8)
        for i in range(len(self.documents)):
            for j in range(len(self.shingles)):
                if self.shingles[j] in shingles_doc[i]:
                    characteristic_matrix[j,i] = 1
                else:
                    characteristic_matrix[j,i] = 0
        return characteristic_matrix            


    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        sketch_len = self.num_hashes
        hash_vals = [random.randint(0, 999999) for i in range(sketch_len)]
        min_hash = np.ndarray(shape=(sketch_len,len(self.documents)), dtype=np.int64)
        for i in range(len(self.documents)):
            for j in range(sketch_len):
                shingle = self.shingle_document(self.documents[i])
                for k in shingle:
                    min_hash[j,i] = min(min_hash[j,i],hash(k+str(hash_vals[j])))
        return min_hash            
                
            
        

    def lsh_buckets(self, signature, bands=200, rows_per_band=5):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}
        for i in range(bands):
            start_row = i * signature.shape[0] // bands
            end_row = start_row + rows_per_band
            for j in range(signature.shape[1]):
                band = signature[start_row:end_row,j]
                hash_band = hash(str(band))
                if hash_band not in buckets:
                    buckets[hash_band] = []
                buckets[hash_band].append(j)
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        characteristic_matrix = self.build_characteristic_matrix()
        signature_matrix = self.min_hash_signature()
        buckets = self.lsh_buckets(signature_matrix)
        self.jaccard_similarity_test(buckets, self.documents)


    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        union = len(first_set.union(second_set))
        if union == 0:
            return 0
        intersection = len(first_set.intersection(second_set))
        return intersection / union
        


    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

# documents = []
#
# with open('../IMDB_crawled.json', 'r') as f:
#     crawled_json = f.read()
# crawled = json.loads(crawled_json)
# for movie in crawled:
#     sums = ''
#     if movie['summaries'] == None:
#         continue
#     for summary in movie['summaries']:
#         sums += summary + ' '
#     documents.append(sums.strip())
# with open('LSHFakeData.json', 'r') as f:
#     dups = f.read()
# lsh_fake = json.loads(dups)
# for movie in lsh_fake:
#     sums = ''
#     for summary in movie['summaries']:
#         sums += summary + ' '
#     documents.append(sums.strip())
#
# minhash = MinHashLSH(documents, 2000)
# minhash.perform_lsh()