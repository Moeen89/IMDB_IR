import numpy as np
import os

import wandb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
if __name__ == '__main__':
    #wandb.login(key="")
    ft_model = FastText(method='skipgram')
    ft_model.prepare(None, mode="load")
    ft_data_loader = FastTextDataLoader("/index")
    X, y = ft_data_loader.create_train_data()
    X = X[0:100]
    y = y[0:100]
    X_emb = np.array([ft_model.get_query_embedding(text) for text in tqdm(X)])

    # 1. Dimension Reduction
    # TODO: Perform Principal Component Analysis (PCA):
    #     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
    #     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
    #     - Draw plots to visualize the results.
    dimred = DimensionReduction()
    X = dimred.pca_reduce_dimension(X_emb, 50)
    dimred.wandb_plot_explained_variance_by_components(X_emb, "IMDB", "2")

    # TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
    #     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
    #     - Use the output vectors from this step to draw the diagram.
    x_tsne = dimred.convert_to_2d_tsne(X)
    dimred.wandb_plot_2d_tsne(X, "IMDB", "2")

    # 2. Clustering
    ## K-Means Clustering
    # TODO: Implement the K-means clustering algorithm from scratch.
    # TODO: Create document clusters using K-Means.
    # TODO: Run the algorithm with several different values of k.
    # TODO: For each run:
    #     - Determine the genre of each cluster based on the number of documents in each cluster.
    #     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
    #     - Check the implementation and efficiency of the algorithm in clustering similar documents.
    # TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
    # TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
    centeroids = []
    cluster_assignments = []
    cu = ClusteringUtils()
    k_values = [2 * i for i in range(1, 10)]
    for k in k_values:
        cu.visualize_kmeans_clustering_wandb(X, k, "IMDB", "3", DimensionReduction())

    cu.plot_kmeans_cluster_scores(X, y, k_values, "IMDB", "4")

    ## Hierarchical Clustering
    # TODO: Perform hierarchical clustering with all different linkage methods.
    # TODO: Visualize the results.
    linkages = ["average", "ward", "complete", "single"]
    for linkage in linkages:
        cu.wandb_plot_hierarchical_clustering_dendrogram(X, "IMDB", linkage, "5")

    # 3. Evaluation
    # TODO: Using clustering metrics, evaluate how well your clustering method is performing.
    cu.visualize_elbow_method_wcss(X, [2 * i for i in range(1, 15)], "IMDB", "6")
    cm = ClusteringMetrics()
    for k in range(2, 20, 4):
        centeroids,cluster_assignments = cu.cluster_kmeans(X, k)
        label = cu.fix_labels(cluster_assignments, k, y)
        print(
            f"{k}:  ari: {cm.adjusted_rand_score(y, cluster_assignments)} , purity: {cm.purity_score(y, label)} , silhouette: {cm.silhouette_score(X, cluster_assignments)}")
