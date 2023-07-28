import nltk
import string
import collections
from tqdm import tqdm
from typing import List
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

class ExtractTopWords:


    def extract_centroids(self, embedddings: np.ndarray, labels: np.ndarray) -> dict:
        """
        Extract centroids of clusters 
        params:
            embeddings: np.ndarray, embeddings to cluster and reduce
            labels: np.ndarray, cluster labels. -1 means outlier
        """

        centroid_dict = {}
        for label in np.unique(labels):
            if label != -1:
                centroid_dict[label] = np.mean(embedddings[labels == label], axis = 0)

        return centroid_dict
    
    def compute_centroid_similarity(self, embeddings: np.ndarray, centroid_dict: dict, cluster_label: int) -> np.ndarray:
        """
        compute the similarity of the document embeddings to the centroid of the cluster

        params:
            embeddings: np.ndarray, embeddings to cluster and reduce
            centroid_dict: dict, dictionary of cluster labels and their centroids
            cluster_label: int, cluster label for which to compute the similarity
        """

        centroid = centroid_dict[cluster_label]
        similarity = np.dot(embeddings, centroid)
        return similarity
    
    def compute_corpus_vocab(self, 
                            corpus: list[str],
                            remove_stopwords: bool = True, 
                            frequent_words_cutoff_quantile: float = 0.95, 
                            infrequent_words_cutoff_quantile:float = 0.01, 
                            remove_punction:bool = True, 
                            min_word_length:int = 2,
                            max_word_length:int = 20, 
                            remove_short_words:bool = True, 
                            remove_numbers:bool = True, 
                            verbose:bool = True):
        """
        compute the vocabulary of the corpus. Perform preprocessing of the corpus 

        params:
            corpus: list[str], list of documents
            remove_stopwords: bool, whether to remove stopwords
            frequent_words_cutoff_quantile: float, quantile to use for the cutoff of frequent words
            infrequent_words_cutoff_quantile: float, quantile to use for the cutoff of infrequent words
            remove_punction: bool, whether to remove punctuation
            remove_long_words: bool, whether to remove long words
            remove_short_words: bool, whether to remove short words
            remove_numbers: bool, whether to remove numbers
            verbose: bool, whether to print progress and say what is happening

        returns:
            list[str], list of words in the corpus sorted aplhabetically
        """
        stopwords = set(nltk.corpus.stopwords.words('english'))
        
        word_counter = collections.Counter()

        for doc in tqdm(corpus, disable=not verbose, desc="Processing corpus"):
            words = nltk.word_tokenize(doc)
            for word in words:
                if remove_punction and word in string.punctuation:
                    continue
                if remove_stopwords and word.lower() in self.stopwords:
                    continue
                if remove_numbers and word.isdigit():
                    continue
                if len(word) > max_word_length or (remove_short_words and len(word) < min_word_length):
                    continue
                
                word_counter[word.lower()] += 1

        total_words = sum(word_counter.values())
        cumulative_freq = 0.0
        freq_words_cutoff = None
        infreq_words_cutoff = None

        for word, count in word_counter.most_common():
            cumulative_freq += count / total_words
            if cumulative_freq > frequent_words_cutoff_quantile and freq_words_cutoff is None:
                freq_words_cutoff = count
            if cumulative_freq > 1 - infrequent_words_cutoff_quantile and infreq_words_cutoff is None:
                infreq_words_cutoff = count

        vocab = {word for word, count in word_counter.items() if count < freq_words_cutoff and count > infreq_words_cutoff}

        # Sorting the vocabulary alphabetically
        vocab = sorted(list(vocab))
        return vocab