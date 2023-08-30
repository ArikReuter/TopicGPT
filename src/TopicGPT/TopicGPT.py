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
from GetEmbeddings.GetEmbeddingsOpenAI import GetEmbeddingsOpenAI
from TopicPrompting.TopicPrompting import TopicPrompting
from TopicRepresentation.TopicRepresentation import Topic
import TopicRepresentation.TopicRepresentation as TopicRepresentation



class TopicGPT:
    """
    This is the main class for doing topic modelling with TopicGPT. It contains all the major methods.
    """

    def __init__(self, 
                 openai_api_key: str,
                 n_topics: int = None,
                 openai_prompting_model: str = "gpt-3.5-turbo-16k",
                 max_number_of_tokens: int = 16384,
                 corpus_intruction: str = "",
                 document_embeddings: np.ndarray = None,
                 vocab_embeddings: dict[str, np.ndarray] = None,
                 embedding_model: str = "text-embedding-ada-002",
                 max_numer_of_tokens_embedding: int = 8191,
                 clusterer: Clustering_and_DimRed = None,
                 n_topwords: int = 2000,
                 n_topwords_description: int = 500,
                 topword_extraction_methods: list[str] = ["tfidf", "cosine_similarity"], 
                 compute_vocab_hyperparams: dict = {},
                 enhancer: TopwordEnhancement = None,
                 topic_prompting: TopicPrompting = None
                 ) -> None:
        
        """
        Initialize the main class for doing topic modelling with TopicGPT.

        params:
            openai_api_key: your OpenAI API key. You can get this from https://beta.openai.com/account/api-keys.
            n_topics: number of topics to find. If None, then it will be automatically determined using the Hdbscan algorithm (https://pypi.org/project/hdbscan/). 
            openai_prompting_model: Model provided by Openai to describe the topics and answer the prompts. For available models see https://platform.openai.com/docs/models. 
            max_number_of_tokens: maximum number of tokens to use for the OpenAI API.
            corpus_intruction: If further information on the given corpus are available, it can be beneficial to let the model know about it. 
            document_embeddings: document embeddings for the corpus. If None, then it will be computed using the openAI API.
            vocab_embeddings: vocab embeddings for the corpus. Is given in a dictionary where the keys are the words and the values are the embeddings. If None, then it will be computed using the openAI API.
            embedding_model: Name of the embedding model to use. See https://beta.openai.com/docs/api-reference/text-embedding for available models.
            max_numer_of_tokens_embedding: Maximum number of tokens to use for the OpenAI API when computing the embeddings.
            clusterer: the clustering and dimensionality reduction object. The class can be found in the "Clustering/Clustering" folder. If None, a clustering object with default parameters will be used. Note that it doe note make sense to provide document and vocab embeddings and an embedding object at the same time. 
            n_topwords: number of top words to extract and save for each topic. Note that fewer top words might be used later. 
            n_topwords_description: number of top words to give to the LLM in order to describe the topic. 
            topword_extraction_methods: list of methods to use for extracting top words. The available methods are "tfidf", "cosine_similarity", and "topword_enhancement". See the file ExtractTopWords/ExtractTopWords.py for more details.
            compute_vocab_hyperparams: hyperparameters for computing the vocab embeddings. See the file ExtractTopWords/ExtractTopWords.py for more details.
            enhancer: the topword enhancement object. Is used to describe the The class can be found in the "TopwordEnhancement/TopwordEnhancement.py" folder. If None, a topword enhancement object with default parameters will be used. Note that if an openai model is specified here it will overwrite the openai_prompting_model argument for topic description.
            topic_prompting: the topic prompting object. This object is used to formulate the prompts. The class can be found in the "TopicPrompting/TopicPrompting.py" folder. If None, a topic prompting object with default parameters will be used. Note that if an openai model is specified here it will overwrite the openai_prompting_model argument for topic description.
        """

        self.openai_api_key = openai_api_key
        self.n_topics = n_topics
        self.openai_prompting_model = openai_prompting_model
        self.max_number_of_tokens = max_number_of_tokens
        self.corpus_intruction = corpus_intruction
        self.document_embeddings = document_embeddings
        self.vocab_embeddings = vocab_embeddings
        self.embedding_model = embedding_model
        self.max_numer_of_tokens_embedding = max_numer_of_tokens_embedding
        self.embedder = GetEmbeddingsOpenAI(api_key = self.openai_api_key, embedding_model = self.embedding_model, max_tokens = self.max_numer_of_tokens_embedding)
        self.clusterer = clusterer
        self.n_topwords = n_topwords
        self.n_topwords_description = n_topwords_description
        self.topword_extraction_methods = topword_extraction_methods
        self.compute_vocab_hyperparams = compute_vocab_hyperparams
        self.enhancer = enhancer
        self.topic_prompting = topic_prompting	

        self.document_embeddings = None # document embeddings for the corpus

        for elem in topword_extraction_methods:
            assert elem in ["tfidf", "cosine_similarity", "topword_enhancement"], "Invalid topword extraction method. Valid methods are 'tfidf', 'cosine_similarity', and 'topword_enhancement'."

        if embedding_model is not None and document_embeddings is not None and vocab_embeddings is not None:
            raise ValueError("You cannot provide document and vocab embeddings and an embedding object at the same time.")
        
        if clusterer is None:
            self.clusterer = Clustering_and_DimRed(number_clusters_hdbscan = self.n_topics)
        
        if enhancer is None:
            self.enhancer = TopwordEnhancement(openai_key = self.openai_api_key, openai_model = self.openai_prompting_model, max_number_of_tokens = self.max_number_of_tokens)

        if topic_prompting is None:
            self.topic_prompting = TopicPrompting(openai_key = self.openai_api_key, openai_prompting_model = "gpt-3.5-turbo-16k",  max_context_length_promting = 16000, enhancer = self.enhancer, openai_embedding_model = self.embedding_model, max_number_of_tokens_embedding = self.max_numer_of_tokens_embedding)
        
        self.extractor = ExtractTopWords.ExtractTopWords()
    
    def compute_embeddings(self, corpus: list[str]) -> (np.ndarray, dict[str, np.ndarray]):
        """
        This function computes the document and vocab embeddings for the corpus. 
        params:
            corpus: List of strings to embed. Where each element in the list is a document.
        return:
            document_embeddings: document embeddings for the corpus. Has shape (len(corpus), n_embedding_dimensions).
            vocab_embeddings: vocab embeddings for the corpus. Is given in a dictionary where the keys are the words and the values are the embeddings.
        """
        
        self.document_embeddings = self.embedder.get_embeddings(corpus)

        self.vocab = self.extractor.compute_corpus_vocab(self.corpus, **self.compute_vocab_hyperparams)

        self.vocab_embeddings = self.extractor.embed_vocab_openAI(self.openai_api_key, self.vocab, embedder = self.embedder)

        return self.document_embeddings, self.vocab_embeddings
    
    def extract_topics(self, corpus) -> list[Topic]:
        """
        This function extracts the topics from the corpus. 
        params:
            corpus: List of strings to embed. Where each element in the list is a document.
        return:
            topics: list of Topic objects. 
        """

        assert self.document_embeddings is not None and self.vocab_embeddings is not None, "You need to compute the embeddings first."

        if self.vocab is None: 
            self.vocab = self.extractor.compute_corpus_vocab(self.corpus, **self.compute_vocab_hyperparams)
        
        self.topic_lis = TopicRepresentation.extract_topics_no_new_vocab_computation(
            corpus = corpus,
            vocab = self.vocab,
            document_embeddings = self.document_embeddings,
            clusterer = self.clusterer,
            vocab_embeddings = self.vocab_embeddings,
            n_topwords = self.n_topwords,
            topword_extraction_methods = self.topword_extraction_methods,
        )

        return self.topic_lis
    
    def describe_topics(self, topics) -> list[Topic]:
        """
        This function gives topics a name and describes them by using the openai api.
        params:
            topics: list of Topic objects. 
        return:
            topics: list of Topic objects. 
        """

        assert self.topic_lis is not None, "You need to extract the topics first."

        self.topic_lis = TopicRepresentation.describe_and_name_topics(
            topics = topics,
            enhancer = self.enhancer,
            topword_method= "cosine_similarity",
            n_words = self.n_topwords_description
        )

        return self.topic_lis
    
    def fit(self, corpus): 
        """
        This function computes the embeddings if necessary, extracts the topics, and describes them.
        params:
            corpus: List of strings to embed. Where each element in the list is a document.
        """
        self.corpus = corpus 

        if self.vocab_embeddings is None or self.document_embeddings is None:
            self.compute_embeddings(corpus = self.corpus)
        
        self.topic_lis = self.extract_topics(corpus = self.corpus)
        self.topic_lis = self.describe_topics(topics = self.topic_lis)

        