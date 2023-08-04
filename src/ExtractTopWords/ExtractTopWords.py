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
import umap

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
                        min_word_length:int = 3,
                        max_word_length:int = 20, 
                        remove_short_words:bool = True, 
                        remove_numbers:bool = True, 
                        verbose:bool = True,
                        min_doc_frequency: int = 3,
                        min_freq: float = 0.1,
                        max_freq: float = 0.9) -> list[str]:
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
            min_freq: float, minimum frequency percentile of words to be considered in the vocabulary
            max_freq: float, maximum frequency percentile of words to be considered in the vocabulary

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
                # remove words that do not begin with an alphabetic character
                if not word[0].isalpha():
                    continue
                if len(word) > max_word_length or (remove_short_words and len(word) < min_word_length):
                    continue
                
                word_lower = word.lower()
                word_counter[word_lower] += 1
                doc_frequency[word_lower].add(doc_id)

        total_words = sum(word_counter.values())
        freq_counter = {word: count / total_words for word, count in word_counter.items()}

        # print most common words and their frequencies
        if verbose:
            print("Most common words:")
            for word, count in word_counter.most_common(10):
                print(f"{word}: {count}")

        freq_arr = np.array(list(freq_counter.values()))

        min_freq_value = np.quantile(freq_arr, min_freq, method="lower")
        max_freq_value = np.quantile(freq_arr, max_freq, method="higher")
        

        vocab = {}

        for word in freq_counter.keys():
            if min_freq_value <= freq_counter[word] <= max_freq_value and len(doc_frequency[word]) >= min_doc_frequency:
                vocab[word] = freq_counter[word]

        vocab = {word for word in freq_counter.keys() 
                if min_freq_value <= freq_counter[word] <= max_freq_value 
                and len(doc_frequency[word]) >= min_doc_frequency}

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
            res_dict[word] = emb
        return res_dict
    
    def compute_bow_representation(self, document: str, vocab: list[str], vocab_set: set[str]) -> np.ndarray:
        """
        compute the bag-of-words representation of a document

        params:
            document: str, document to compute the bag-of-words representation of
            vocab: list[str], list of words in the corpus sorted alphabetically
            vocab_set: set[str], set of words in the corpus sorted alphabetically
        returns:
            np.ndarray, bag-of-words representation of the document
        """
        bow = np.zeros(len(vocab))
        words = word_tokenize(document)
        if vocab_set is None:
            vocab_set = set(vocab)
        for word in words:
            if word.lower() in vocab_set:
                bow[vocab.index(word.lower())] += 1
        return bow

    
    def extract_topics_tfidf(self, corpus: list[str], vocab: list[str], labels: np.ndarray, top_n_words: int = 10) -> (dict, np.ndarray):
        """
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
        vocab_set = set(vocab)
        for i, doc in tqdm(enumerate(corpus), desc="Computing word-topic matrix", total=len(corpus)):
            if labels[i] != -1:
                bow = self.compute_bow_representation(doc, vocab, vocab_set)
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
                top_words[topic] = [vocab[word_idx] for word_idx in np.argsort(-tfidf[:, i - 1])[:top_n_words]]


        return top_words, tfidf
    
    def compute_embedding_similarity_centroids(self, vocab: list[str], vocab_embedding_dict: dict, umap_mapper: umap.UMAP, centroid_dict: dict, reduce_vocab_embeddings: bool = False, reduce_centroid_embeddings: bool = False) -> np.ndarray:
        """
        compute the cosine similarity of each word in the vocab to each centroid
        params:
            vocab: list[str], list of words in the corpus sorted alphabetically
            vocab_embedding_dict: dict, dictionary of words and their embeddings
            umap_mapper: umap.UMAP, UMAP mapper to transform new embeddings in the same way as the document embeddings
            centroid_dict: dict, dictionary of cluster labels and their centroids. -1 means outlier
            reduce_vocab_embeddings: bool, whether to reduce the vocab embeddings with the UMAP mapper
            reduce_centroid_embeddings: bool, whether to reduce the centroid embeddings with the UMAP mapper
        returns:
            np.ndarray, cosine similarity of each word in the vocab to each centroid. has shape (len(vocab), len(centroid_dict) - 1)
        """
        embedding_dim = umap_mapper.n_components
        centroid_arr = np.zeros((len(centroid_dict) - 1, embedding_dim))
        for i, centroid in enumerate(centroid_dict.values()):
            if i != -1:
                centroid_arr[i - 1] = centroid
        if reduce_centroid_embeddings:
            centroid_arr = umap_mapper.transform(centroid_arr)
        
        centroid_arr = centroid_arr / np.linalg.norm(centroid_arr, axis=1).reshape(-1,1)
        

        org_embedding_dim = list(vocab_embedding_dict.values())[0].shape[0]
        vocab_arr = np.zeros((len(vocab), org_embedding_dim))
        for i, word in enumerate(vocab):
            vocab_arr[i] = vocab_embedding_dict[word]
        if reduce_vocab_embeddings:
            vocab_arr = umap_mapper.transform(vocab_arr)

        vocab_arr = vocab_arr / np.linalg.norm(vocab_arr, axis=1).reshape(-1,1)
        
        similarity = vocab_arr @ centroid_arr.T # cosine similarity
        return similarity
    
    def extract_top_words_centroid_similarity(self, vocab: list[str], vocab_embedding_dict: dict, centroid_dict: dict, umap_mapper: umap.UMAP, top_n_words: int = 10, reduce_vocab_embeddings: bool = True, reduce_centroid_embeddings: bool = False) -> (dict, np.ndarray):
        """
        Extract the top-words for each cluster by computing the cosine similarity of the words that occur in the corpus to the centroid of the cluster
        params: 
            corpus: list[str], list of documents
            vocab: list[str], list of words in the corpus sorted alphabetically
            vocab_embedding_dict: dict, dictionary of words and their embeddings
            centroid_dict: dict, dictionary of cluster labels and their centroids. -1 means outlier
            umap_mapper: umap.UMAP, UMAP mapper to transform new embeddings in the same way as the document embeddings
            top_n_words: int, number of top words to extract per topic
            reduce_vocab_embeddings: bool, whether to reduce the vocab embeddings with the UMAP mapper
            reduce_centroid_embeddings: bool, whether to reduce the centroid embeddings with the UMAP mapper

        returns:
            dict, dictionary of topics and their top words
            np.ndarray, cosine similarity of each word in the vocab to each centroid. has shape (len(vocab), len(centroid_dict) - 1)
        """
        similarity_mat = self.compute_embedding_similarity_centroids(vocab, vocab_embedding_dict, umap_mapper, centroid_dict, reduce_vocab_embeddings, reduce_centroid_embeddings)
        top_words = {}
        for i, topic in enumerate(np.unique(list(centroid_dict.keys()))):
            if topic != -1:
                top_words[topic] = [vocab[word_idx] for word_idx in np.argsort(-similarity_mat[:, i - 1])[:top_n_words]]

        return top_words, similarity_mat