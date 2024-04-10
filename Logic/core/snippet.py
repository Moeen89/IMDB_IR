class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """
        stop_words = ['this', 'that', 'about', 'whom', 'being', 'where', 'why', 'had', 'should', 'each']
        for stop_word in stop_words:
            query = query.replace(stop_word, "")
        return query

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""

        query = self.remove_stop_words_from_query(query)
        if query is None:
            return "", []
        words = doc.split()
        query_words = list(set(query.split()))
        not_exist_words = list(set(query_words).difference(set(words)))
        locations = []
        for i in range(len(words)):
            if words[i] in query_words:
                locations.append(i)
        last_end = 0
        for location in locations:
            start = location - self.number_of_words_on_each_side
            end = location + self.number_of_words_on_each_side
            if start < 0:
                start = 0
            if end > len(words):
                end = len(words)
            if last_end > start:
                start = last_end

            snippet = " ".join(words[start:end])

            snippet = snippet.replace(words[location], "***" + words[location] + "***", )
            if last_end + 1 < start:
                snippet = "... " + snippet
            final_snippet += snippet + " "
            last_end = end

        return final_snippet, not_exist_words
