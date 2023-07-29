import nltk
import string
import collections
from tqdm import tqdm
from typing import List
import numpy as np
import re  # import the re module
from nltk.tokenize import word_tokenize
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.insert(0, parentdir) 

from GetEmbeddings import GetEmbeddingsOpenAI

nltk.download('stopwords')
nltk.download('punkt')

class ExtractTopWords:


    def extract_centroids(self, embedddings: np.ndarray, labels: np.ndarray) -> dict:
        """
        Extract centroids of clusters 
        params:
            embeddings: np.ndarray, embeddings to cluster and reduce
            labels: np.ndarray, cluster labels. -1 means outlier
        returns: 
            dict, dictionary of cluster labels and their centroids
        """

        centroid_dict = {}
        for label in np.unique(labels):
            if label != -1:
                centroid_dict[label] = np.mean(embedddings[labels == label], axis = 0)

        return centroid_dict
    
    def compute_centroid_similarity(self, embeddings: np.ndarray, centroid_dict: dict, cluster_label: int) -> np.ndarray:
        """
        compute the similarity of the document embeddings to the centroid of the cluster via cosine similarity

        params:
            embeddings: np.ndarray, embeddings to cluster and reduce
            centroid_dict: dict, dictionary of cluster labels and their centroids
            cluster_label: int, cluster label for which to compute the similarity
        returns: 
            np.ndarray, cosine similarity of the document embeddings to the centroid of the cluster
        """
        centroid = centroid_dict[cluster_label]
        similarity = np.dot(embeddings, centroid) / (np.linalg.norm(embeddings) * np.linalg.norm(centroid))
        return similarity
    
    def get_most_similar_docs(self, corpus: list[str], embeddings: np.ndarray, labels: np.ndarray, centroid_dict: dict, cluster_label: int, top_n: int = 10) -> List[str]:
        """
        get the most similar documents to the centroid of a cluster

        params:
            corpus: list[str], list of documents
            embeddings: np.ndarray, embeddings to cluster and reduce
            labels: np.ndarray, cluster labels. -1 means outlier
            centroid_dict: dict, dictionary of cluster labels and their centroids
            cluster_label: int, cluster label for which to compute the similarity
            top_n: int, number of top documents to extract
        returns: 
            List[str], list of the most similar documents to the centroid of a cluster
        """
        similarity = self.compute_centroid_similarity(embeddings, centroid_dict, cluster_label)
        most_similar_docs = [corpus[i] for i in np.argsort(similarity)[-top_n:][::-1]]
        return most_similar_docs
    
    def compute_corpus_vocab(self, 
                        corpus: list[str],
                        remove_stopwords: bool = True, 
                        remove_punction:bool = True, 
                        min_word_length:int = 2,
                        max_word_length:int = 20, 
                        remove_short_words:bool = True, 
                        remove_numbers:bool = True, 
                        verbose:bool = True,
                        min_doc_frequency: int = 3):
        """
        compute the vocabulary of the corpus. Perform preprocessing of the corpus 

        params:
            corpus: list[str], list of documents
            remove_stopwords: bool, whether to remove stopwords
            remove_punction: bool, whether to remove punctuation
            remove_long_words: bool, whether to remove long words
            remove_short_words: bool, whether to remove short words
            remove_numbers: bool, whether to remove numbers
            verbose: bool, whether to print progress and say what is happening
            min_doc_frequency: int, minimum number of documents a word should appear in to be considered in the vocab

        returns:
            list[str], list of words in the corpus sorted alphabetically
        """
        stopwords = set(nltk.corpus.stopwords.words('english'))
        
        word_counter = collections.Counter()
        doc_frequency = collections.defaultdict(set)

        for doc_id, doc in enumerate(tqdm(corpus, disable=not verbose, desc="Processing corpus")):
            words = nltk.word_tokenize(doc)
            for word in words:
                if remove_punction and word in string.punctuation:
                    continue
                if remove_stopwords and word.lower() in stopwords:
                    continue
                if remove_numbers and re.search(r'\d', word):  # use a regular expression to check for digits
                    continue
                if not re.search('[a-zA-Z]', word):  # checks if word contains at least one alphabetic character
                    continue
                if len(word) > max_word_length or (remove_short_words and len(word) < min_word_length):
                    continue
                
                word_lower = word.lower()
                word_counter[word_lower] += 1
                doc_frequency[word_lower].add(doc_id)

        vocab = {word for word in word_counter.keys() if len(doc_frequency[word]) >= min_doc_frequency}

        # Sorting the vocabulary alphabetically
        vocab = sorted(list(vocab))
        
        return vocab
    

    def compute_words_topics(self, corpus: list[str], vocab: list[str], labels: np.ndarray) -> dict:
        """
        compute the words per topic

        params:
            corpus: list[str], list of documents
            vocab: list[str], list of words in the corpus sorted alphabetically
            labels: np.ndarray, cluster labels. -1 means outlier
        returns:
            dict, dictionary of topics and their words
        """

        # Download NLTK resources (only required once)
        nltk.download("punkt")
        vocab = set(vocab)

        words_per_topic = {label: [] for label in np.unique(labels) if label != -1}

        for doc, label in tqdm(zip(corpus, labels), desc="Computing words per topic", total=len(corpus)):
            if label != -1:
                words = word_tokenize(doc)
                for word in words:
                    if word.lower() in vocab:
                        words_per_topic[label].append(word.lower())

        return words_per_topic
                    
    def embed_vocab_openAI(self, api_key: str, vocab: list[str], embedder:GetEmbeddingsOpenAI = None) -> np.ndarray:
        """
        embed the vocabulary using the OpenAi embedding API

        params:
            api_key: str, openai api key
            vocab: list[str], list of words in the corpus sorted alphabetically
            embedder: GetEmbeddingsOpenAI, embedding object
        returns:
            np.ndarray, embeddings of the vocabulary
        """
        vocab = sorted(list(set(vocab)))
        if embedder is None: 
            embedder = GetEmbeddingsOpenAI.GetEmbeddingsOpenAI(api_key)
        result = embedder.get_embeddings(vocab)

        print(result)

        res_dict = {}
        for word, emb in zip(vocab, result["embeddings"]):
            print(word)
            res_dict[word] = emb
        return res_dict
    
    def compute_bow_representation(self, document: str, vocab: list[str]) -> np.ndarray:
        """
        compute the bag-of-words representation of a document

        params:
            document: str, document to compute the bag-of-words representation of
            vocab: list[str], list of words in the corpus sorted alphabetically
        returns:
            np.ndarray, bag-of-words representation of the document
        """
        bow = np.zeros(len(vocab))
        words = word_tokenize(document)
        vocab_set = set(vocab)
        for word in words:
            if word.lower() in vocab_set:
                bow[vocab.index(word.lower())] += 1
        return bow

    
    def extract_topics_tfidf(self, corpus: list[str], vocab: list[str], labels: np.ndarray, top_n_words: int = 10) -> dict:
        """
        UNDER CONSTRUCTION 
        extract the top-words for each topic using a class-based tf-idf score

        params: 
            corpus: list[str], list of documents
            vocab: list[str], list of words in the corpus sorted alphabetically
            labels: np.ndarray, cluster labels. -1 means outlier
            top_n_words: int, number of top words to extract per topic
        returns: 
            dict, dictionary of topics and their top words
        """
        # compute for each cluster how often each word occurs in the cluster
        word_topic_mat = np.zeros((len(vocab), len((np.unique(labels))) - 1))
        for i, doc in tqdm(enumerate(corpus), desc="Computing word-topic matrix", total=len(corpus)):
            if labels[i] != -1:
                bow = self.compute_bow_representation(doc, vocab)
                word_topic_mat[:, labels[i]] += bow

        tf = word_topic_mat / np.sum(word_topic_mat, axis=0)
        idf = np.log(1 + (word_topic_mat.shape[1] / np.sum(word_topic_mat > 0, axis=1)))

        print(tf.shape, idf.shape)

        tfidf = tf * idf[:, np.newaxis]

        # set tfidf to zero if tf is nan (happens if word does not occur in any document or topic does not have any words)
        tfidf[np.isnan(tf)] = 0

        # extract top words for each topic
        top_words = {}
        for i, topic in enumerate(np.unique(labels)):
            if topic != -1:
                top_words[topic] = [vocab[word_idx] for word_idx in np.argsort(tfidf[:, i - 1])[-top_n_words:][::-1]]


        return top_words, tfidf
    
    def extract_top_words_centroid_similarity(self, vocab: list[str], vocab_embedding_dict: dict, cluster_labels:np.ndarray, centroid_dict: dict, top_n_words: int = 10) -> dict:
        """
        UNDER CONSTRUCTION 
        Extract the top-words for each cluster by computing the cosine similarity of the words that occur in the corpus to the centroid of the cluster
        params: 
            corpus: list[str], list of documents
            vocab: list[str], list of words in the corpus sorted alphabetically
            vocab_embedding_dict: dict, dictionary of words and their embeddings
            cluster_labels: np.ndarray, cluster labels. -1 means outlier
            cluster_dict: dict, dictionary of cluster labels and their centroids
            top_n_words: int, number of top words to extract per topic
        returns:
            dict, dictionary of topics and their top words
        """
        top_words = {}
        for label in np.unique(cluster_labels):
            if label != -1:
                similarity = self.compute_centroid_similarity(np.array([vocab_embedding_dict[word] for word in vocab]), centroid_dict, label)
                top_words[label] = [vocab[word_idx] for word_idx in np.argsort(similarity)[-top_n_words:][::-1]]
        return top_words