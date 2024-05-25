class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        for i in range(len(word) - k + 1):
            shingles.add(word[i:i + k])

        return shingles

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        union = len(first_set.union(second_set))
        if union == 0:
            return 0
        intersection = len(first_set.intersection(second_set))
        return intersection / union

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()
        for document in all_documents:
            for field in ["id", "title", "first_page_summary"]:
                if all_documents[document][field] is not None:
                    for word in all_documents[document][field].split():
                        if word not in all_shingled_words:
                            all_shingled_words[word] = self.shingle_word(word)
                        if word not in word_counter:
                            word_counter[word] = 0
                        word_counter[word] += 1
            for field in ["directors", "writers", "stars","summaries"]:
                if all_documents[document][field] is not None:
                    for f in all_documents[document][field]:
                        for word in f.split():
                            if word not in all_shingled_words:
                                all_shingled_words[word] = self.shingle_word(word)
                            if word not in word_counter:
                                word_counter[word] = 0
                            word_counter[word] += 1

        return all_shingled_words, word_counter

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        for candidate in self.all_shingled_words:
            jaccard = self.jaccard_score(self.shingle_word(word), self.all_shingled_words[candidate])
            top5_candidates.append((candidate, jaccard))
        top5_candidates.sort(key=lambda x: x[1], reverse=True)    
        return top5_candidates[0:5]

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""

        for word in query.split():
            if word in self.all_shingled_words:
                final_result += word + " "
            else:
                top5_candidates = self.find_nearest_words(word)
                final_result += top5_candidates[0][0] + " "
        final_result = final_result.strip()
        return final_result
