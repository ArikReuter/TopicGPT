import numpy as np
import os 
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from Clustering.Clustering import Clustering_and_DimRed
from ExtractTopWords.ExtractTopWords import ExtractTopWords
from TopwordEnhancement.TopwordEnhancement import TopwordEnhancement



class TopicGPT:
    """
    This is the main class for doing topic modelling with TopicGPT. It contains all the major methods.
    """

    def __init__(self, 
                 corpus: list[str],
                 openai_api_key: str,
                 n_topics: int = None,
                 document_embeddings: np.ndarray = None,
                 vocab_embeddings: dict[str, np.ndarray] = None,
                 clusterer: Clustering_and_DimRed = None,
                 ) -> None:
        
        """
        Initialize the main class for doing topic modelling with TopicGPT.

        params:
            corpus: documents to do topic modelling on. Is in the form of a list of strings where each string is a document.
            openai_api_key: your OpenAI API key. You can get this from https://beta.openai.com/account/api-keys.
            n_topics: number of topics to find. If None, then it will be automatically determined using the Hdbscan algorithm (https://pypi.org/project/hdbscan/). 
            document_embeddings: document embeddings for the corpus. If None, then it will be computed using the openAI API.
            vocab_embeddings: vocab embeddings for the corpus. Is given in a dictionary where the keys are the words and the values are the embeddings. If None, then it will be computed using the openAI API.
            clusterer: the clustering and dimensionality reduction object. The class can be found in the "Clustering/Clustering" folder. If None, a clustering object with default parameters will be used.
        """

        self.corpus = corpus
        self.openai_api_key = openai_api_key
        self.n_topics = n_topics
        self.document_embeddings = document_embeddings
        self.vocab_embeddings = vocab_embeddings
        self.clusterer = clusterer

