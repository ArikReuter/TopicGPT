import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import tiktoken
import openai
from typing import Callable
import numpy as np

basic_instruction =  "You are a helpful assistant. You are excellent at inferring topics from top-words extracted via topic-modelling. You make sure that everything you output is strictly based on the provided text."

class TopwordEnhancement:
    
    def __init__(self, openai_key: str, openai_model: str = "gpt-3.5-turbo", max_context_length = 4000, openai_model_temperature:float = 0.5, basic_model_instruction: str = basic_instruction, corpus_instruction: str = ""):
        """
        params:
            openai_key: your openai key
            openai_model: the openai model to use
            max_context_length: the maximum length of the context for the openai model
            openai_model_temperature: the softmax temperature to use for the openai model
            basic_model_instruction: the basic instruction for the model
            corpus_instruction: the instruction for the corpus. Useful if specific information on the corpus on hand is available
        """
        self.openai_key = openai_key
        self.openai_model = openai_model
        self.max_context_length = max_context_length
        self.openai_model_temperature = openai_model_temperature
        self.basic_model_instruction = basic_model_instruction
        self.corpus_instruction = corpus_instruction

    def __str__(self) -> str:
        repr = f"TopwordEnhancement(openai_model = {self.openai_model})"
        return repr

    def __repr__(self) -> str:
        repr = f"TopwordEnhancement(openai_model = {self.openai_model})"
        return repr
    
    def count_tokens_api_message(self, messages: list[dict[str]]) -> int:
        """
        Count the number of tokens in the API message
        params:
            message: the message from the API
        returns:
            number of tokens in the message
        """
        encoding = tiktoken.encoding_for_model(self.openai_model)
        n_tokens = 0
        for message in messages: 
            for key, value in message.items():
                if key == "content":
                    n_tokens += len(encoding.encode(value))
        
        return n_tokens
    
    def describe_topic_topwords_completion_object(self, 
                               topwords: list[str], 
                               n_words: int = None,
                               query_function: Callable = lambda tws: f"Please give me the common topic of those words: {tws}. Also describe the various aspects and sub-topics of the topic.") -> openai.ChatCompletion:
        """
        Describe the given topic based on its topwords by using the openai model. The given query is used together with the base query to query the model.
        params:
            topwords: list of topwords
            n_words: number of words to use for the query. If None, all words are used
            query_function: function to query the model. The function should take a list of topwords and return a string
        returns:
            A description of the topics by the model in form of an openai.ChatCompletion object
        """
        if n_words is None:
            n_words = len(topwords)
        topwords = topwords[:n_words]
        topwords = np.array(topwords)

        # if too many topwords are given, use only the first part of the topwords that fits into the context length
        tokens_cumsum = np.cumsum([len(tiktoken.encoding_for_model(self.openai_model).encode(tw + ", ")) for tw in topwords]) + len(tiktoken.encoding_for_model(self.openai_model).encode(self.basic_model_instruction + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print("Too many topwords given. Using only the first part of the topwords that fits into the context length. Number of topwords used: ", np.argmax(tokens_cumsum > self.max_context_length))
            n_words = np.argmax(tokens_cumsum > self.max_context_length)
            topwords = topwords[:n_words]



        completion = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content":  self.basic_model_instruction + " " + self.corpus_instruction},
                {"role": "user", "content": query_function(topwords)},
            ],
            temperature = self.openai_model_temperature
        )

        return completion
    
    def describe_topic_topwords_str(self, 
                               topwords: list[str], 
                               n_words: int = None,
                               query_function: Callable = lambda tws: f"Please give me the common topic of those words: {tws}. Also describe the various aspects and sub-topics of the topic.") -> str:
        """
        Describe the given topic based on its topwords by using the openai model. The given query is used together with the base query to query the model.
        params:
            topwords: list of topwords
            n_words: number of words to use for the query. If None, all words are used
            query_function: function to query the model. The function should take a list of topwords and return a string
        returns:
            A description of the topics by the model in form of a string
        """
        completion = self.describe_topic_topwords_completion_object(topwords, n_words, query_function)
        return completion.choices[0].message["content"]
    
    def generate_topic_name_str(self,
                                topwords: list[str],
                                n_words: int = None,
                                query_function: Callable = lambda tws: f"Please give me the common topic of those words: {tws}. Give me only the title of the topic and nothing else please. Make sure the title is precise and not longer than 5 words, ideally even shorter.") -> str:
        """
        Generate a topic name based on the given topwords by using the openai model. The given query is used together with the base query to query the model. Works completely analogously to describe_topic_topwords_str.
        params:
            topwords: list of topwords
            n_words: number of words to use for the query. If None, all words are used
            query_function: function to query the model. The function should take a list of topwords and return a string
        returns:
            A topic name generated by the model in form of a string
        """
        return self.describe_topic_topwords_str(topwords, n_words, query_function)


    def describe_topic_documents_completion_object(self, 
                                 documents: list[str],
                                 truncate_doc_thresh = 100,
                                 n_documents: int = None,
                                 query_function: Callable = lambda docs: f"Please give me the common topic of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.") -> openai.ChatCompletion:
        """
        Describe the given topic based on its documents by using the openai model. The given query is used together with the base query to query the model.
        params:
            documents: list of documents
            truncate_doc_thresh: threshold for the number of words in a document. If a document has more words than this threshold, it is pruned to this threshold.
            n_documents: number of documents to use for the query. If None, all documents are used
            query_function: function to query the model. The function should take a list of documents and return a string
        returns:
            A description of the topics by the model in form of an openai.ChatCompletion object
        """
        if n_documents is None:
            n_documents = len(documents)
        documents = documents[:n_documents]

        # prune documents based on number of tokens they contain 
        new_doc_lis = []
        for doc in documents:
            doc = doc.split(" ")
            if len(doc) > truncate_doc_thresh:
                doc = doc[:truncate_doc_thresh]
            new_doc_lis.append(" ".join(doc))
        documents = new_doc_lis

        # if too many documents are given, use only the first part of the documents that fits into the context length
        tokens_cumsum = np.cumsum([len(tiktoken.encoding_for_model(self.openai_model).encode(doc + ", ")) for doc in documents]) + len(tiktoken.encoding_for_model(self.openai_model).encode(self.basic_model_instruction + " " + self.corpus_instruction))
        if tokens_cumsum[-1] > self.max_context_length:
            print("Too many documents given. Using only the first part of the documents that fits into the context length. Number of documents used: ", np.argmax(tokens_cumsum > self.max_context_length))
            n_documents = np.argmax(tokens_cumsum > self.max_context_length)
            documents = documents[:n_documents]
        
        completion = openai.ChatCompletion.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": self.basic_model_instruction + " " + self.corpus_instruction},
                {"role": "user", "content": query_function(documents)},
            ],
            temperature = self.openai_model_temperature
        )

        return completion
    
    
    @staticmethod
    def sample_identity(n_docs: int) -> np.ndarray:
        """
        do not change the order of the documents
        params:
            n_docs: number of documents
        returns:
            np.arange(n_docs)
        """
        return np.arange(n_docs)
    
    @staticmethod
    def sample_uniform(n_docs: int) -> np.ndarray:
        """
        sample documents randomly
        params:
            n_docs: number of documents
        returns:
            np.random.permutation(n_docs)
        """
        return np.random.permutation(n_docs)
    @staticmethod
    def sample_poisson(n_docs: int) -> np.ndarray:
        """
        sample documents randomly according to a poisson distribution, i.e. draw more documents from the beginning of the list
        params:
            n_docs: number of documents
        returns:
            np.random.permutation(n_docs)
        """
        return np.random.poisson(1, n_docs)
    
    def describe_topic_documents_sampling_completion_object(self,
                                                            documents: list[str],
                                                            truncate_doc_thresh = 100,
                                                            n_documents: int = None,
                                                            query_function: Callable = lambda docs: f"Please give me the common topic of the sample of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.",
                                                            sampling_strategy: (Callable, str) = None,
                                                            )-> openai.ChatCompletion:
        """
        Take a list of documents belonging to a topic and describe the topic by sampling a subset of the documents and describing the topic based on the sample.
        params:
            documents: list of documents ordered by similarity to centroid of topic
            truncate_doc_thresh: threshold for the number of words in a document. If a document has more words than this threshold, it is pruned to this threshold.
            n_documents: number of documents to use for the query. If None, all documents are used
            query_function: function to query the model. The function should take a list of documents and return a string
            sampling_strategy: strategy to sample the documents. If None, the first provided documents are used. Otherwise the returned array specifies the order the documents should be considered. 
            If the sampling strategy is a string, it is interpreted as a method of the class, e.g. "sample_uniform" is interpreted as self.sample_uniform. Can also be a function
        returns:
            A description of the topics by the model in form of an openai.ChatCompletion object
        """
        if type(sampling_strategy) == str:
            if sampling_strategy == "topk":
                sampling_strategy = self.sample_identity
            if sampling_strategy=="identity":
                sampling_strategy = self.sample_identity
            elif sampling_strategy=="uniform":
                sampling_strategy = self.sample_uniform
            elif sampling_strategy=="poisson":
                sampling_strategy = self.sample_poisson
        
        new_documents = [documents[i] for i in sampling_strategy(n_documents)]

        result = self.describe_topic_documents_completion_object(new_documents, truncate_doc_thresh, n_documents, query_function)
        return result
    
    def describe_topic_document_sampling_str(self,
                                             documents: list[str],
                                             truncate_doc_thresh = 100,
                                             n_documents: int = None,
                                             query_function: Callable = lambda docs: f"Please give me the common topic of the sample of those documents: {docs}. Note that the documents are truncated if they are too long. Also describe the various aspects and sub-topics of the topic.",
                                             sampling_strategy: (Callable, str) = None,
                                             )-> str:
        """
        Take a list of documents belonging to a topic and describe the topic by sampling a subset of the documents and describing the topic based on the sample.
        params:
            documents: list of documents ordered by similarity to centroid of topic
            truncate_doc_thresh: threshold for the number of words in a document. If a document has more words than this threshold, it is pruned to this threshold.
            n_documents: number of documents to use for the query. If None, all documents are used
            query_function: function to query the model. The function should take a list of documents and return a string
            sampling_strategy: strategy to sample the documents. If None, the first provided documents are used. Otherwise the returned array specifies the order the documents should be considered. 
            If the sampling strategy is a string, it is interpreted as a method of the class, e.g. "sample_uniform" is interpreted as self.sample_uniform. Can also be a function
        returns:
            A description of the topics by the model in form of a string
        """
        completion = self.describe_topic_document_sampling_completion_object(documents, truncate_doc_thresh, n_documents, query_function, sampling_strategy)
        return completion.choices[0].message["content"]