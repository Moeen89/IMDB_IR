from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            self.hubs.append(movie["id"])
            self.graph.add_node(movie["id"])
            for star in movie["stars"]:
                if star not in self.authorities:
                    self.authorities.append(star)
                self.graph.add_node(star)
                self.graph.add_edge(movie["id"], star)

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        extended_hubs = []
        for movie in corpus:
            if movie["id"] not in self.hubs:
                if movie["stars"]:
                    for star in movie["stars"]:
                        if star in self.authorities:
                            extended_hubs.append(movie)
                            break
        for movie in extended_hubs:
            self.graph.add_node(movie["id"])
            if movie["stars"]:
                for star in movie["stars"]:
                    if star in self.authorities:
                        self.graph.add_edge(movie["id"], star)
        for hub in extended_hubs:
            self.hubs.append(hub["id"])

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        a_s = []
        h_s = []
        hub_scores = {}
        authority_scores = {}
        for node in self.hubs:
            hub_scores[node] = 1
        for node in self.authorities:
            authority_scores[node] = 1

        for i in range(num_iteration):
            new_hub_score = {}
            for hub in self.hubs:
                new_hub_score[hub] = 0
                for authority in self.graph.get_successors(hub):
                    new_hub_score[hub] += authority_scores[authority]
            new_hub_score = {k: v / sum(new_hub_score.values()) for k, v in new_hub_score.items()}
            new_authority_score = {}
            for authority in self.authorities:
                new_authority_score[authority] = 0
                for hub in self.graph.get_predecessors(authority):
                    new_authority_score[authority] += hub_scores[hub]
            new_authority_score = {k: v / sum(new_authority_score.values()) for k, v in new_authority_score.items()}
            hub_scores = new_hub_score
            authority_scores = new_authority_score

        a_s = sorted(authority_scores, key=authority_scores.get, reverse=True)[:max_result]
        h_s = sorted(hub_scores, key=hub_scores.get, reverse=True)[:max_result]
        return a_s, h_s


if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    path = "\index"
    corpus = list(Index_reader(path, Indexes.DOCUMENTS).get_index().values())
    root_set = list(corpus)[:20]
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
