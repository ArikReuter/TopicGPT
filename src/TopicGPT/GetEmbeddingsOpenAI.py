import openai
from openai.embeddings_utils import get_embedding
import tiktoken
from tqdm import tqdm
import numpy as np

class GetEmbeddingsOpenAI:
    """
    This class allows to compute embeddings of text using the OpenAI API.
    """

    def __init__(self, api_key:str, embedding_model:str = "text-embedding-ada-002", tokenizer:str = None, max_tokens:int = 8191) -> None:
        """
        Constructor of the class.
        :param api_key: API key to use the OpenAI API.
        :param embedding_model: Name of the embedding model to use.
        :param tokenizer: Name of the tokenizer to use.
        :param max_tokens: Maximum number of tokens to use.

        Per default the embedding model "text-embedding-ada-002" is used with the corresponding tokenizer "cl100k_base" and a maximum number of tokens of 8191.
        """
        self.api_key = api_key
        openai.api_key = api_key
        self.embedding_model = embedding_model

        self.tokenizer_str = tokenizer
        
    
        self.max_tokens = max_tokens

    @staticmethod
    def num_tokens_from_string(string: str, encoding) -> int:
            """
            Returns the number of tokens in a text string.
            :param string: Text string to compute the number of tokens.
            :param encoding: function to encode the string into tokens.
            :return: Number of tokens in the text string.
            """
            num_tokens = len(encoding.encode(string))
            return num_tokens

    def compute_number_of_tokens(self, corpus: list[str]) -> int:
        """
        This function computes the total number of tokens needed to embed the corpus.
        :param corpus: List of strings to embed. Where each element in the list is a document.
        :return: Total number of tokens needed to embed the corpus.
        """

        if self.tokenizer_str is None:
             tokenizer = tiktoken.encoding_for_model(self.embedding_model)

        else: 
             tokenizer = tiktoken.get_encoding(self.tokenizer_str)

        num_tokens = 0
        for document in tqdm(corpus):
            num_tokens += self.num_tokens_from_string(document, tokenizer)
        
        return num_tokens
        
    def split_doc(self, text):
             """
             split a single document that is longer than the maximum number of tokens into a list of smaller documents 
                :param text: string to embed. 
                :return: List of strings to embed. Where each element in the list is a list of chunks comprising the document. 
             """
             split_text = []
             split_text.append(text[:self.max_tokens])
             for i in range(1, len(text) // self.max_tokens):
                split_text.append(text[i * self.max_tokens:(i + 1) * self.max_tokens])
             split_text.append(text[(len(text) // self.max_tokens) * self.max_tokens:])
             return split_text
    
    def split_long_docs(self, text: list[str]) -> list[list[str]]:
         """
         split all documents that are longer than the maximum number of tokens into a list of smaller documents 
            :param text: List of strings to embed. Where each element in the list is a document.    
            :return: List of list of strings to embed. Where each element in the list is a list of chunks comprising the document. 
         """
         if self.tokenizer_str is None:
              tokenizer = tiktoken.encoding_for_model(self.embedding_model)
         else:
              tokenizer = tiktoken.get_encoding(self.tokenizer_str)
    
         
         split_text = []
         for document in tqdm(text):
            if self.num_tokens_from_string(document, tokenizer) > self.max_tokens:
                split_text.append(self.split_doc(document))
            else:
                split_text.append([document])
         return split_text   
    
    def make_api_call(self, text:str):
         """
        make an API call to the OpenAI API to embed a text string
            :param text: string to embed.
            :return: API response.
        """
         response = openai.Embedding.create(input = [text], model = self.embedding_model)
         return response


    
    def get_embeddings_doc_split(self, corpus: list[list[str]], n_tries = 3) -> list[dict]:
        """
        This function computes the embeddings of a corpus for splitted documents.
        :param corpus: List of strings to embed. Where each element in the list is a document that is represented by the list of its chunks.
        :param n_tries: Number of tries to make an API call.
        :return: List of dictionaries. Where each dictionary contains the embedding of the document, the text of the document and a list of errors that occured during the embedding process.
        """
        api_res_list = [] 
        for i in tqdm(range(len(corpus))):
            chunk_lis = corpus[i]
            api_res_doc = []
            for chunk_n, chunk in enumerate(chunk_lis):

                for i in range(n_tries + 1):
                    try: 
                        api_res_doc.append(
                            {"api_res": self.make_api_call(chunk), 
                            "error": None }
                         )
                        break
                    except Exception as e:
                            print(f"Error {e} occured for chunk {chunk_n} of document {i}")
                            print(chunk)
                            print("Trying again.")
                            if i == n_tries: 
                                print("Maximum number of tries reached. Skipping chunk.")
                                api_res_doc.append(
                                    {"api_res": None, 
                                    "error": e })
                        

            # average the embeddings of the chunks
            emb_lis = []
            for api_res in api_res_doc:
                if api_res["api_res"] is not None:
                    emb_lis.append(np.array(api_res["api_res"]["data"][0]["embedding"]))
            text = " ".join(chunk_lis)
            embedding = np.mean(emb_lis, axis = 0)
            api_res_list.append(
                {"embedding": embedding, 
                "text": text, 
                "errors": [api_res["error"] for api_res in api_res_doc]}
                )
        return api_res_list
    
    def convert_api_res_list(self, api_res_list: list[dict]) -> dict:
         """
         Convert the api_res list in to a dictionary containing the embeddings as a matrix and the corpus as a list of string
            :param api_res_list: List of dictionaries. Where each dictionary contains the embedding of the document, the text of the document and a list of errors that occured during the embedding process.
            :return: Dictionary containing the embeddings as a matrix and the corpus as a list of string
         """

         embeddings = np.array([api_res["embedding"] for api_res in api_res_list])
         corpus = [api_res["text"] for api_res in api_res_list]
         errors = [api_res["errors"] for api_res in api_res_list]
         return {"embeddings": embeddings, "corpus": corpus, "errors": errors}

    
    def get_embeddings(self, corpus: list[str]) -> dict:
        """
        This function computes the embeddings of a corpus.
        :param corpus: List of strings to embed. Where each element in the list is a document.
        :return: Dictionary containing the embeddings as a matrix and the corpus as a list of strings
        """
        corpus_split = self.split_long_docs(corpus)
        corpus_emb = self.get_embeddings_doc_split(corpus_split)
        self.corpus_emb = corpus_emb
        res = self.convert_api_res_list(corpus_emb)
        return res