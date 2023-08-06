import numpy as np
import umap
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from Clustering.Clustering import Clustering_and_DimRed
from ExtractTopWords.ExtractTopWords import ExtractTopWords




class Topic:
    """
    class to represent a topic and all its attributes
    """

    def __init__(self, 
                 topic_name: str, 
                 documents: list[str], 
                 words: dict[str, int],
                 centroid_hd: np.ndarray = None, 
                 centroid_ld: np.ndarray = None,
                 document_embeddings_hd: np.ndarray = None,
                 document_embeddings_ld: np.ndarray = None,
                 document_embedding_similarity: np.ndarray = None,
                 umap_mapper: umap.UMAP = None,
                 top_words: dict[str, list[str]] = None,
                 top_word_scores: dict[str, list[float]] = None
                 ) -> None:
        """
        Represents a topic and all its attributes
        params: 
            topic_name: name of the topic
            documents: list of documents in the topic
            words: dict of words and their counts in the topic
            centroid_hd: centroid of the topic in high dimensional space
            centroid_ld: centroid of the topic in low dimensional space
            document_embeddings_hd: embeddings of documents in high dimensional space that belong to this topic
            document_embeddings_ld: embeddings of documents in low dimensional space that belong to this topic
            document_embedding_similarity: similarity array of document embeddings to the centroid in the low dimensional space
            umap_mapper: umap mapper object to map from high dimensional space to low dimensional space
            top_words: dictionary of top-words in the topic according to different metrics
            top_word_scores: dictionary of how representative the top-words are according to different metrics
        """
        self.topic_name = topic_name
        self.documents = documents
        self.words = words
        self.centroid_hd = centroid_hd
        self.centroid_ld = centroid_ld
        self.document_embeddings_hd = document_embeddings_hd
        self.document_embeddings_ld = document_embeddings_ld
        self.document_embedding_similarity = document_embedding_similarity
        self.umap_mapper = umap_mapper
        self.top_words = top_words
        self.top_word_scores = top_word_scores


    def __str__(self) -> str:
        repr = f"Topic: {self.topic_name}\n"
        
        return repr
    def __repr__(self) -> str:
        repr = f"Topic: {self.topic_name}\n"
        
        return repr

@staticmethod
def extract_topics(corpus: list[str], document_embeddings: np.ndarray, clusterer: Clustering_and_DimRed, vocab_embeddings: np.ndarray, n_topwords: int = 30, topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"], compute_vocab_hyperparams: dict = {}) -> list[Topic]:
    """
    Extract the topics from the given corpus by using the clusterer object on the embeddings
    params: 
        corpus: list of documents
        document_embeddings: embeddings of the documents
        clusterer: clusterer object to cluster the documents
        vocab_embeddings: embeddings of the vocabulary
        n_topwords: number of top-words to extract from the topics
        topword_extraction_methods: list of methods to extract top-words from the topics. Can contain "tfidf" and "cosine_similarity"
        compute_vocab_hyperparams: hyperparameters for the top-word extraction methods
    returns:
        list of Topic objects
    """
    for elem in topword_extraction_methods:
        if elem not in ["tfidf", "cosine_similarity"]:
            raise ValueError("topword_extraction_methods can only contain 'tfidf' and 'cosine_similarity'")
    if topword_extraction_methods == []:
        raise ValueError("topword_extraction_methods cannot be empty")

    dim_red_embeddings, labels, umap_mapper = clusterer.cluster_and_reduce(document_embeddings)  # get dimensionality reduced embeddings, their labels and the umap mapper object

    extractor = ExtractTopWords()
    centroid_dict = extractor.extract_centroids(document_embeddings, labels)  # get the centroids of the clusters
    dim_red_centroids = umap_mapper.transform(np.array(list(centroid_dict.values())))  # map the centroids to low dimensional space
    dim_red_centroid_dict = {label: centroid for label, centroid in zip(centroid_dict.keys(), dim_red_centroids)}

    vocab = extractor.compute_corpus_vocab(corpus, **compute_vocab_hyperparams)  # compute the vocabulary of the corpus
    if "tfidf" in topword_extraction_methods:
        tfidf_topwords, tfidf_dict = extractor.extract_topwords_tfidf(corpus, vocab, labels, n_topwords)
    if "cosine_similarity" in topword_extraction_methods:
        cosine_topwords, cosine_dict = extractor.extract_topwords_centroid_similarity(vocab, vocab_embeddings, corpus, labels, dim_red_centroid_dict, umap_mapper, n_topwords, reduce_vocab_embeddings = True, reduce_centroid_embeddings = False)
                                                                                     
    topics = []
    for i, label in enumerate(np.unique(labels)):
        if label == -1: # dont include outliers
            continue
        print(label)
        topic_name = f"{label}"
        documents = [doc for j, doc in enumerate(corpus) if labels[j] == label]
        embeddings_hd = document_embeddings[labels == label]
        embeddings_ld = dim_red_embeddings[labels == label]
        centroid_hd = centroid_dict[label]
        centroid_ld = dim_red_centroids[label]
        
        centroid_similarity = np.dot(embeddings_ld, centroid_ld)/(np.linalg.norm(embeddings_ld, axis = 1)*np.linalg.norm(centroid_ld))
        similarity_sorting = np.argsort(centroid_similarity)[::-1]
        documents = [documents[i] for i in similarity_sorting]
        embeddings_hd = embeddings_hd[similarity_sorting]
        embeddings_ld = embeddings_ld[similarity_sorting]

        top_words = {
            "tfidf": tfidf_topwords[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_topwords[label] if "cosine_similarity" in topword_extraction_methods else None
        }
        top_word_scores = {
            "tfidf": tfidf_dict[label] if "tfidf" in topword_extraction_methods else None,
            "cosine_similarity": cosine_dict[label] if "cosine_similarity" in topword_extraction_methods else None
        }

        topic = Topic(topic_name = topic_name,
                        documents = documents,
                        words = vocab,
                        centroid_hd = centroid_hd,
                        centroid_ld = centroid_ld,
                        document_embeddings_hd = embeddings_hd,
                        document_embeddings_ld = embeddings_ld,
                        document_embedding_similarity = centroid_similarity,
                        umap_mapper = umap_mapper,
                        top_words = top_words, 
                        top_word_scores = top_word_scores
                        )
                      
        topics.append(topic)
    
    return topics







