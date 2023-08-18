import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import openai
import numpy as np
import json
import tiktoken
import openai
import re
import sklearn
from TopicRepresentation.TopicRepresentation import Topic
from TopicRepresentation.TopicRepresentation import extract_and_describe_topic_cos_sim
from TopwordEnhancement import TopwordEnhancement


basic_model_instruction = """You are a helpful assistant. 
You are excellent at inferring information about topics discovered via topic modelling using information retrieval. 
You summarize information intelligently. 
You use the functions you are provided with if applicable.
You make sure that everything you output is strictly based on the provided text. If you cite documents, give their indices. 
You always explicitly say if you don't find any useful information!
You only say that something is contained in the corpus if you are very sure about it!"""


class TopicPrompting:
    """
    This class allows to formulate prompts and queries against the identified topics to get more information about them
    """

    def __init__(self, 
                 topic_lis: list[Topic], 
                 openai_key: str, 
                 openai_prompting_model: str = "gpt-3.5-turbo", 
                 max_context_length_promting = 4000, 
                 openai_model_temperature_prompting:float = 0.5,
                 openai_embedding_model = "text-embedding-ada-002",
                 max_context_length_embedding = 8191, 
                 basic_model_instruction = basic_model_instruction,
                 corpus_instruction = "",
                 random_state = 42):
        """
        params: 
            topic_lis: list of Topic objects
            openai_key: openai key
            topic_list: list of Topic objects
            openai_prompting_model: openai model to use for prompting
            max_context_length_promting: maximum context length for the prompting model
            openai_model_temperature_prompting: temperature for the prompting model
            openai_embedding_model: openai model to use for computing embeddings for similarity search
            max_context_length_embedding: maximum context length for the embedding model
            basic_model_instruction: basic instruction for the prompting model
            corpus_instruction: instruction for the prompting model to use the corpus
            random_state: random state for reproducibility
        """
        self.topic_lis = topic_lis
        self.openai_key = openai_key
        self.openai_prompting_model = openai_prompting_model
        self.max_context_length_promting = max_context_length_promting
        self.openai_model_temperature_prompting = openai_model_temperature_prompting
        self.openai_embedding_model = openai_embedding_model
        self.max_context_length_embedding = max_context_length_embedding    
        self.basic_model_instruction = basic_model_instruction
        self.corpus_instruction = corpus_instruction
        self.random_state = random_state

        self.function_descriptions = {
                "knn_search": {
                    "name": "knn_search",
                    "description": "This function can be used to find out if a topic is about a specific subject or contains information about it. Note that it is possible that just useless documents are returned.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic_index": {
                                "type": "integer",
                                "description": "index of the topic to search in."
                            },
                            "query": {
                                "type": "string",
                                "description": "query string. Can be a single word or a sentence. Used to create an embedding and search a vector database for the k nearest neighbors."
                            },
                            "k": {
                                "type": "integer",
                                "description": "number of neighbors to return. Use more neighbors to get a more diverse and comprehensive set of results."
                            }
                        },
                        "required": ["topic_index", "query"]

                    }
                },
                "identify_topic_idx": {
                    "name": "identify_topic_idx",
                    "description": "This function can be used to identify the index of the topic that the query is most likely about. This is useful if the topic index is needed for other functions. It should NOT be used to find more detailed information on topics. Note that it is possible that the model does not find any topic that fits the query. In this case, the function returns None.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "query string. Can be a single word or a sentence. Used to find the index of the topic that is most likely about the query."
                            }
                        },
                        "required": ["query"]

                    }
                }
        }

        self.functionNames2Functions = {
            "knn_search": self.knn_search_openai,
            "identify_topic_idx": self.identify_topic_idx_openai
        }

    def knn_search(self, topic_index: int, query: str, k: int = 20, doc_cutoff_threshold: int = 1000) -> list[str]:
        """
        find the k nearest neighbors of the query in the given topic based on cosine similarity in the original embedding space
        params: 
            topic: Topic object
            query: query string
            k: number of neighbors to return
            doc_cutoff_threshold: maximum number of tokens per document. Afterwards, the document is cut off
        returns:
            list of k nearest neighbors
        """
        topic = self.topic_lis[topic_index]

        query_embedding = openai.Embedding.create(input = [query], model = self.openai_embedding_model)["data"][0]["embedding"]

        query_similarities = topic.document_embeddings_hd @ query_embedding / (np.linalg.norm(topic.document_embeddings_hd, axis = 1) * np.linalg.norm(query_embedding))

        topk_doc_indices = np.argsort(query_similarities)[::-1][:k]
        topk_docs = [topic.documents[i] for i in topk_doc_indices]

        # cut off documents that are too long
        max_number_tokens = self.max_context_length_promting - len(tiktoken.encoding_for_model(self.openai_prompting_model).encode(self.basic_model_instruction + " " + self.corpus_instruction)) - 100
        n_tokens = 0
        for i, doc in enumerate(topk_docs):
            encoded_doc = tiktoken.encoding_for_model(self.openai_prompting_model).encode(doc)
            n_tokens += len(encoded_doc[:doc_cutoff_threshold])
            if n_tokens > max_number_tokens:
                topk_docs = topk_docs[:i]
                topk_doc_indices = topk_doc_indices[:i]
                break
            if len(encoded_doc) > doc_cutoff_threshold:
                encoded_doc = encoded_doc[:doc_cutoff_threshold]
                topk_docs[i] = tiktoken.encoding_for_model(self.openai_prompting_model).decode(encoded_doc)




        return topk_docs, [int(elem) for elem in topk_doc_indices]
    
    def knn_search_openai(self, topic_index: int, query: str, k: int = 20) -> json:
        """"
        A version of the knn_search function that returns a json file to be used with the openai API
        params:
            topic_index: index of the topic to search in
            query: query string
            k: number of neighbors to return
        returns:
            json object to be used with the openai API
        """
        topk_docs, topk_doc_indices = self.knn_search(topic_index, query, k)
        json_obj = json.dumps({
            "top-k documents": topk_docs,
            "indices of top-k documents": list(topk_doc_indices)
        })
        return json_obj
    
    def prompt_knn_search(self, llm_query: str, topic_index: int = None, n_tries:int = 2) -> str:
        """
        Use the LLM to answer the llm query based on the documents belonging to the topic.  
        params: 
            llm_query: query string for the LLM
            topic_index: index of topic object. If None, the topic is inferred from the query
            n_tries: number of tries to get a valid response from the LLM
        returns:
            answer string
        """
        messages = [
            {
                "role": "system",
                "content": self.basic_model_instruction + " " + self.corpus_instruction
            },
            {
                "role": "user",
                "content": llm_query
            }
            ]
        for _ in range(n_tries):
            try: 
                response_message = openai.ChatCompletion.create(
                    model = self.openai_prompting_model,
                    messages = messages,
                    functions = [self.function_descriptions["knn_search"]],
                    function_call = "auto")["choices"][0]["message"]
                
                # Step 2: check if GPT wanted to call a function
                print(response_message)
                function_call = response_message.get("function_call")
                if function_call is not None:
                    print("GPT wants to the call the function: ", function_call)
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors

                    function_name = function_call["name"]
                    function_to_call = self.functionNames2Functions[function_name]
                    function_args = json.loads(function_call["arguments"])
                    if topic_index is not None:
                        function_args["topic_index"] = topic_index
                    function_response = function_to_call(**function_args)

                    # Step 4: send the info on the function call and function response to GPT
                    messages.append(response_message)  # extend conversation with assistant's reply
                    
                
                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response,
                        }
                    )  # extend conversation with function response

                    print(messages)
                    second_response = openai.ChatCompletion.create(
                        model=self.openai_prompting_model,
                        messages=messages,
                    )  # get a new response from GPT where it can see the function response
            except (TypeError, ValueError, openai.error.APIError, openai.error.APIConnectionError) as error:
                print("Error occured: ", error)
                print("Trying again...")
            
            return second_response
        
    def identify_topic_idx(self, query: str, n_tries = 3) -> int:
        """
        Identify the index of the topic that the query is most likely about. This is done by asking a LLM to say which topic has the description that best fits the query. 
        params:
            query: query string
            n_tries: number of tries to get a valid response from the LLM
        returns:
            index of the topic that the query is most likely about
        """

        topic_descriptions_str = ""
        for i, topic in enumerate(self.topic_lis):
            description = topic.topic_description
            description = f"""Topic index: {i}: \n {description} \n \n"""
            topic_descriptions_str += description
        
        system_prompt = f"""You are a helpful assistant."""
        
        user_prompt = f""" Please find the index of the topic that is about the following query: {query}. 
        Those are the given topics: '''{topic_descriptions_str}'''.
        Please make sure to reply ONLY with an integer number between 0 and {len(self.topic_lis) - 1}!
        Reply with -1 if you don't find any topic that fits the query!
        Always explicitly say if you don't find any useful information by replying with -1! If in doubt, say that you did not find any useful information!
        Reply in the following format: "The topic index is: <index>"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
            ]
        for _ in range(n_tries):
            try: 
                response_message = openai.ChatCompletion.create(
                model = self.openai_prompting_model,
                messages = messages
                
                )["choices"][0]["message"]

            except (TypeError, ValueError, openai.error.APIError, openai.error.APIConnectionError) as error:
                print("Error occured: ", error)
                print("Trying again...")


        
        response_text = response_message["content"]
        # find integer number in response text
        match = re.search(r'(-?\d+)', response_text)
        topic_index = int(match.group(1))
        
        if topic_index is None:
            raise ValueError("No integer number found in response text! The model gave the following response: ", response_text)
        
        if topic_index == -1: 
            return None
        else:
            return topic_index

    def identify_topic_idx_openai(self, query: str, n_tries = 3) -> json:
        """
        A version of the identify_topic_idx function that returns a json file to be used with the openai API
        params:
            query: query string
            n_tries: number of tries to get a valid response from the LLM
        returns:
            json object to be used with the openai API
        """
        topic_index = self.identify_topic_idx(query, n_tries)
        json_obj = json.dumps({
            "topic index": topic_index
        })
        return json_obj

    def general_prompt(self, prompt: str, n_tries = 2) -> str:
        """
        Prompt the LLM with a general prompt and return the response. Allow the llm to call any function defined in the class. 
        Use n_tries in case the LLM does not give a valid response.
        params:
            prompt: prompt string
            n_tries: number of tries to get a valid response from the LLM
        returns:
            response string
        """
        messages = [
            {
                "role": "system",
                "content": self.basic_model_instruction + " " + self.corpus_instruction
            },
            {
                "role": "user",
                "content": prompt
            }
            ]
        for _ in range(n_tries):
            try: 
                response_message = openai.ChatCompletion.create(
                    model = self.openai_prompting_model,
                    messages = messages,
                    functions = [self.function_descriptions[key] for key in self.function_descriptions.keys()],
                    function_call = "auto")["choices"][0]["message"]
                
                # Step 2: check if GPT wanted to call a function
                print(response_message)
                function_call = response_message.get("function_call")
                if function_call is not None:
                    print("GPT wants to the call the function: ", function_call)
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors

                    function_name = function_call["name"]
                    function_to_call = self.functionNames2Functions[function_name]
                    function_args = json.loads(function_call["arguments"])
                    function_response = function_to_call(**function_args)

                    # Step 4: send the info on the function call and function response to GPT
                    messages.append(response_message)  # extend conversation with assistant's reply
                
                    messages.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": function_response,
                        }
                    )  # extend conversation with function response

                    print(messages)
                    second_response = openai.ChatCompletion.create(
                        model=self.openai_prompting_model,
                        messages=messages,
                    )  # get a new response from GPT where it can see the function response
            except (TypeError, ValueError, openai.error.APIError, openai.error.APIConnectionError) as error:
                print("Error occured: ", error)
                print("Trying again...")
            
            return second_response

    def split_topic_new_assignments(self, topic_idx: int, new_topic_assignments: np.ndarray, vocab_embedding_dict: dict, enhancer: TopwordEnhancement, inplace = False) -> list[Topic]:
        """
        split a topic into new topics based on new topic assignments.
        params:
            topic_idx: index of the topic to split
            new_topic_assignments: new topic assignments for the documents in the topic
            vocab_embedding_dict: dictionary mapping words to their embeddings
            enhancer: TopwordEnhancement object fro naming and describing the new topics
            inplace: if True, the topic is split inplace. Otherwise, a new list of topics is created and returned
        returns:
            list of new topics
        """
        old_topic = self.topic_lis[topic_idx]

        assert len(new_topic_assignments) == len(old_topic.documents), "new_topic_assignments must have the same length as the number of documents in the topic!"

        # create new topics
        new_topics = []
        for i in range(np.unique(new_topic_assignments)):
            docs = [old_topic.documents[j] for j in range(len(old_topic.documents)) if new_topic_assignments[j] == i]
            docs_embeddings = old_topic.document_embeddings_hd[new_topic_assignments == i]
            words_raw = []
            for doc in docs:
                words_raw += doc.split(" ")
            words_raw = set(words_raw)
            words = [word for word in old_topic.words if word in words_raw]

            new_topic = extract_and_describe_topic_cos_sim(
                documents_topic = docs,
                document_embeddings_topic = docs_embeddings,
                words_topic = words,
                vocab_embeddings = vocab_embedding_dict,
                umap_mapper = old_topic.umap_mapper,
                enhancer=enhancer,
                topword_extraction_methods = ["cosine_similarity"],
                n_topwords = 2000
            )
            # TODO: also add tfidf topwords
            new_topic.topic_idx = len(self.topic_lis) + i + 1
            new_topics.append(new_topic)
        
        if inplace:
            self.topic_lis.pop(topic_idx)
            self.topic_lis += new_topics
        else:
            new_topic_lis = self.topic_lis.copy()
            new_topic_lis.pop(topic_idx)
            new_topic_lis += new_topics
            return new_topic_lis

    def split_topic_kmeans(self, topic_idx: int, n_clusters: int = 2, inplace = False) -> list[Topic]:
        """
        Split an existing topic into several subtopics using kmeans clustering. 
        params:
            topic_idx: index of the topic to split
            n_clusters: number of clusters to split the topic into
            inplace: if True, the topic is split inplace. Otherwise, a new list of topics is created and returned
        """
        old_topic = self.topic_lis[topic_idx]
        embeddings = old_topic.document_embeddings_ld  # embeddings to split into clusters

        kmeans_res = sklearn.cluster.KMeans(n_clusters = n_clusters, random_state = self.random_state).fit(embeddings)
        cluster_labels = kmeans_res.labels_

        new_topics = self.split_topic_new_assignments(topic_idx, cluster_labels, old_topic.vocab_embedding_dict, old_topic.enhancer, inplace)

        return new_topics

    
    def split_topic_keyword():
        pass 

    def combine_topics():
        pass 

    def create_new_topic_keyword():
        """
        Create a new topic based on a keyword. Remove all documents belonging to the other topics from them and add them to the new topic.
        """
        pass

# implement function for proper chatting 
# Add description to plot of topics