import os 
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import openai
import pickle

import unittest
from src.TopicGPT.TopicGPT import TopicGPT
#from src.TopicRepresentation.TopicRepresentation import Topic
from TopicRepresentation.TopicRepresentation import Topic

class TestTopicGPT_init_and_fit(unittest.TestCase):
    """
    Test the init and fit functions of the TopicGPT class
    """

    @classmethod
    def setUpClass(cls, sample_size = 0.1):
        """
        load the necessary data and only keep a sample of it 
        """
        print("Setting up class...")
        cls.api_key_openai = os.environ.get('OPENAI_API_KEY')
        openai.organization = os.environ.get('OPENAI_ORG')

        with open("Data/Emebeddings/embeddings_20ng_raw.pkl", "rb")  as f:
            data_raw = pickle.load(f)

        corpus = data_raw["corpus"]
        doc_embeddings = data_raw["embeddings"]

        n_docs = int(len(corpus) * sample_size)
        cls.corpus = corpus[:n_docs]
        cls.doc_embeddings = doc_embeddings[:n_docs]

        print("Using {} out of {} documents".format(n_docs, len(data_raw["corpus"])))

        with open("Data/Emebeddings/embeddings_20ng_vocab.pkl", "rb") as f:
            cls.embeddings_vocab = pickle.load(f)


    def test_init(self):
        """
        test the init function of the TopicGPT class
        """
        print("Testing init...")
        topicgpt = TopicGPT(openai_api_key = self.api_key_openai)
        self.assertTrue(isinstance(topicgpt, TopicGPT))

        topicgpt = TopicGPT(openai_api_key = self.api_key_openai, 
                            n_topics= 20)
        self.assertTrue(isinstance(topicgpt, TopicGPT))
        
        topicgpt = TopicGPT(openai_api_key = self.api_key_openai, 
                            n_topics= 20,
                            openai_prompting_model="gpt-4",
                            max_number_of_tokens=8000,
                            corpus_instruction="This is a corpus instruction", 
                            document_embeddings = self.doc_embeddings,
                            vocab_embeddings= self.embeddings_vocab)
        self.assertTrue(isinstance(topicgpt, TopicGPT))

        # check if assertions are triggered

        with self.assertRaises(AssertionError):
            topicgpt = TopicGPT(openai_api_key = self.api_key_openai, 
                                n_topics= 0,
                                openai_prompting_model="gpt-4",
                                max_number_of_tokens=8000,
                                corpus_instruction="This is a corpus instruction")
            
        with self.assertRaises(AssertionError):
            topicgpt = TopicGPT(openai_api_key = self.api_key_openai, 
                                n_topics= 20,
                                openai_prompting_model="gpt-4",
                                max_number_of_tokens=0,
                                corpus_instruction="This is a corpus instruction")
            
    def test_fit(self):
        """
        test the fit function of the TopicGPT class
        """
        print("Testing fit...")

        def instance_test(topicgpt):
            topicgpt.fit(self.corpus)

            self.assertTrue(hasattr(topicgpt, "vocab"))
            self.assertTrue(hasattr(topicgpt, "topic_lis"))

            self.assertTrue(isinstance(topicgpt.vocab, list))
            self.assertTrue(isinstance(topicgpt.vocab[0], str))

            self.assertTrue(isinstance(topicgpt.topic_lis, list))

            print(topicgpt.topic_lis[0])
            print(type(topicgpt.topic_lis[0]))
            self.assertTrue(type(topicgpt.topic_lis[0]) == Topic)

            self.assertTrue(len(topicgpt.topic_lis) <= 20)

            self.assertTrue(topicgpt.topic_lis == topicgpt.topic_prompting.topic_lis)
            self.assertTrue(topicgpt.vocab == topicgpt.topic_prompting.vocab)
            self.assertTrue(topicgpt.vocab_embeddings == topicgpt.topic_prompting.vocab_embeddings)

        
        topicgpt1 = TopicGPT(openai_api_key = self.api_key_openai, 
                            n_topics= 20,
                            document_embeddings = self.doc_embeddings,
                            vocab_embeddings = self.embeddings_vocab)
    
        topicgpt2 = TopicGPT(openai_api_key = self.api_key_openai,
                             n_topics= None,
                                document_embeddings = self.doc_embeddings, 
                                vocab_embeddings = self.embeddings_vocab,
                                openai_prompting_model="gpt-4",
                                max_number_of_tokens=8000)

        topic_gpt_list = [topicgpt1, topicgpt2]

        for topic_gpt in topic_gpt_list:
            instance_test(topic_gpt)
        



if __name__ == "__main__":
    unittest.main()