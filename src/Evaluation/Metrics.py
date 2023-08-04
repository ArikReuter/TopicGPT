from sentence_transformers import SentenceTransformer

from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import nltk

from sklearn.metrics.pairwise import cosine_similarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary




def Embed_corpus(dataset, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2"), emb_filename= None, emb_path = "Embeddings/"):
    """
    Create a dictionary with the word embedding of every word in the dataset. 
    Use the embedder. 
    If the file 'Embeddings/{emb_filename}.pickle' is available, read the embeddings from this file. 
    
    Otherwise create new embeddings.
    Returns the embedding dict
    """
    
    if emb_filename is None:
        emb_filename = str(dataset)
    try: 
        emb_dic = pickle.load(open(f'{emb_path}{emb_filename}.pickle', 'rb'))
    except FileNotFoundError:
        emb_dic = {}
        word_list = []
        for doc in dataset.get_corpus():
            for word in doc:
                word_list.append(word)

        word_list = set(word_list)
        for word in tqdm(word_list):
            emb_dic[word] = embedder.encode(word)

        with open(f'{emb_path}{emb_filename}.pickle', "wb") as handle:
            pickle.dump(emb_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return emb_dic

def Embed_vocab(vocab, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2"), emb_filename= None, emb_path = "Embeddings/"):
    """
    Create a dictionary with the word embedding of every word in the dataset. 
    Use the embedder. 
    If the file 'Embeddings/{emb_filename}.pickle' is available, read the embeddings from this file. 
    
    Otherwise create new embeddings.
    Returns the embedding dict
    """
    
    if emb_filename is None:
        emb_filename = str(vocab)
    try: 
        emb_dic = pickle.load(open(f'{emb_path}{emb_filename}.pickle', 'rb'))
    except FileNotFoundError:
        emb_dic = {}
        word_list = []
        for word in vocab:
            word_list.append(word)

        word_list = set(word_list)
        for word in tqdm(word_list):
            emb_dic[word] = embedder.encode(word)

        with open(f'{emb_path}{emb_filename}.pickle', "wb") as handle:
            pickle.dump(emb_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return emb_dic

def Update_corpus_dic_list(word_lis, emb_dic, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2"), emb_filename= None, emb_path = "Embeddings/"):
    """
    Updates embedding dict with embeddings in word_lis
    """

    if emb_filename is None:
        emb_filename = str(word)
    try: 
        emb_dic = pickle.load(open(f'{emb_path}{emb_filename}.pickle', 'rb'))
    except:
        FileNotFoundError
        print('No existing embedding found. Starting to embed corpus update dictionary')

        keys = set(emb_dic.keys())
        for word in tqdm(set(word_lis)):

            if word not in keys:
                emb_dic[word] = embedder.encode(word)
        
        with open(f'{emb_path}{emb_filename}.pickle', "wb") as handle:
            pickle.dump(emb_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return emb_dic

def Embed_topic(topics_tw, corpus_dict,  n_words = 10, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")):
    """
    takes the list of topics and embed the top n_words words with the corpus dict
    if possible, else use the embedder. 
    """
    topic_embeddings = []
    for topic in tqdm(topics_tw):
        if n_words != None: 
            topic = topic[:n_words]
        
        add_lis = []
        for word in topic:
            try: 
                add_lis.append(corpus_dict[word])
            except KeyError:
                #print(f'did not find key {word} to embedd topic, create new embedding...')
                add_lis.append(embedder.encode(word))

        topic_embeddings.append(add_lis)

    return topic_embeddings

def Embed_stopwords(stopwords, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")):
    """
    take the list of stopwords and embeds them with embedder 
    """
    
    sw_dic = {}  #first create dictionary with embedding of every unique word 
    stopwords_set = set(stopwords)
    for word in tqdm(stopwords_set):
        sw_dic[word] = embedder.encode(word)

    sw_list =[]     
    for word in stopwords:      #use this dictionary to embed all the possible stopwords 
        sw_list.append(sw_dic[word])

    return sw_list

def mean_over_diag(mat):
    """
    Calculate the average of all elements of a quadratic matrix
    that are above the diagonal
    """
    h, w = mat.shape
    assert h==w, 'matrix must be quadratic'
    mask = np.triu_indices(h, k = 1)
    return np.mean(mat[mask])

def cos_sim_pw(mat):
    """
    calculate the average cosine similarity of all rows in the matrix (but exclude the similarity of a row to itself)
    """
    sim = cosine_similarity(mat)
    return mean_over_diag(sim)

class Embedding_Coherence( ):
    """
    Average cosine similarity between all top words in a topic
    """

    def __init__(self, corpus_dict, n_words = 10):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider 
        """

        self.n_words = n_words
        self.corpus_dict = corpus_dict
  


    def score_per_topic(self, model_output):
        
        topics_tw = model_output['topics']


        emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw



        topic_sims = []
        for topic_emb in emb_tw:                          #for each topic append the average pairwise cosine similarity within its words 
            topic_sims.append(cos_sim_pw(topic_emb))

        return np.array(topic_sims)


    def score(self, model_output):
        res = self.score_per_topic(model_output)
        return sum(res)/len(res)

class Embedding_Topic_Diversity( ):
    """
    Measure the diversity of the topics by calculating the mean cosine similarity 
    of the mean vectors of the top words of all topics
    """

    def __init__(self, corpus_dict, n_words = 10):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider 
        """

        self.n_words = n_words
        self.corpus_dict = corpus_dict



    def score(self, model_output):
         
        topics_tw = model_output['topics']  #size: (n_topics, voc_size)
        topic_weights = model_output['topic-word-matrix'][:, :self.n_words]  #select the weights of the top words 

        topic_weights = topic_weights/np.sum(topic_weights, axis = 1).reshape(-1, 1) #normalize the weights such that they sum up to one

    
        emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
    


        weighted_vecs = topic_weights[:, :, None] * emb_tw  #multiply each embedding vector with its corresponding weight
        topic_means = np.sum(weighted_vecs, axis = 1) #calculate the sum, which yields the weighted average
  
        return float(cos_sim_pw(topic_means))

    
    def score_per_topic(self, model_output):
         
        topics_tw = model_output['topics']  #size: (n_topics, voc_size)
        topic_weights = model_output['topic-word-matrix'][:, :self.n_words]  #select the weights of the top words size: (n_topics, n_topwords)

        topic_weights = topic_weights/np.sum(topic_weights, axis = 1).reshape(-1, 1) #normalize the weights such that they sum up to one


        emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw
 

        weighted_vecs = topic_weights[:, :, None] * emb_tw  #multiply each embedding vector with its corresponding weight
        topic_means = np.sum(weighted_vecs, axis = 1) #calculate the sum, which yields the weighted average
    
        sim = cosine_similarity(topic_means)   #calculate the pairwise cosine similarity of the topic means 
        sim_mean = (np.sum(sim, axis = 1) - 1)/(len(sim)-1)  #average the similarity of each topic's mean to the mean of every other topic 

        return sim_mean

class Null_Space_Distance( ):
    """
    Measure the distance of the mean of the topic topwords to the mean of the embedding of the stop words
    """

    def __init__(self, stopword_list, corpus_dict, n_words = 10, embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")):
        """
        stopword_corpus: The list of all stopwords to compare with; i.e. the 
        specific stopwords of this corpus
        """
        self.stopword_list = stopword_list
        self.stopword_emb = Embed_stopwords(stopword_list.get_corpus(), embedder)  #embed all the stopwords size: (n_stopwords, emb_dim)
        self.stopword_mean = np.mean(np.array(self.stopword_emb), axis = 0)        #mean of stopword embeddings
        self.corpus_dict = corpus_dict
        self.n_words = n_words
        self.embeddings = None

    def score(self, model_output, new_Embeddings = True):
        if new_Embeddings:
            self.embeddings = None
        return float(np.mean(self.score_per_topic(model_output, new_Embeddings)))

    def score_per_topic(self, model_output, new_Embeddings = True):
        if new_Embeddings:
            self.embeddings = None

        topics_tw = model_output['topics']  #size: (n_topics, voc_size)
        topic_weights = model_output['topic-word-matrix'][:, :self.n_words]  #select the weights of the top words 

        topic_weights = topic_weights/np.sum(topic_weights, axis = 1).reshape(-1, 1) #normalize the weights such that they sum up to one

        if self.embeddings is None:
            emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
            emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
            self.embeddings = emb_tw
        else:
            emb_tw = self.embeddings

        weighted_vecs = topic_weights[:, :, None] * emb_tw  #multiply each embedding vector with its corresponding weight
        topic_means = np.sum(weighted_vecs, axis = 1) #calculate the sum, which yields the weighted average
        
        topword_sims =[]   #iterate over every topic and append the cosine similarity of the topic's centroid and the stopword mean
        for mean in topic_means:
            topword_sims.append(cosine_similarity(mean.reshape(1, -1), self.stopword_mean.reshape(1, -1))[0,0])

        return np.array(topword_sims)

class Embedding_Intruder_avg_cos_sim( ):
    """
    For each topic, draw several intruder words that are not from the same topic by first selecting some topics that are not the specific topic and 
    then selecting one word from each of those topics. 
    The intruder score for the topic is then calculated as the average cosine similarity of the intruder words and the top words. 
    """

    def __init__(self, corpus_dict, n_intruders = 1, n_words = 10):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider 
        """

        self.n_intruders = n_intruders
        self.corpus_dict = corpus_dict
        self.n_words = n_words
        self.embeddings = None


    def score_one_intr_per_topic(self, model_output, new_Embeddings = True):
        """
        Calculate the score for each topic but only with one intruder word
        """
        if new_Embeddings:  #for this function, reuse embeddings per default
            self.embeddings = None 
        
        topics_tw = model_output['topics']

        if self.embeddings is None:
            emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
            emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
            self.embeddings = emb_tw
        else:
            emb_tw = self.embeddings

        avg_sim_topic_list = []   #iterate over each topic and append the average similarity to the intruder word
        for idx, topic in enumerate(emb_tw):
            mask = np.full(emb_tw.shape[0], True) #mask out the current topic
            mask[idx] = False
            
            other_topics = emb_tw[mask] #embeddings of every other topic except the current one

            intr_topic_idx = np.random.randint(other_topics.shape[0])  #select random topic index
            intr_word_idx = np.random.randint(other_topics.shape[1])   #select random word index

            intr_embedding = other_topics[intr_topic_idx, intr_word_idx]  #select random word 

            sim = cosine_similarity(intr_embedding.reshape(1,-1), topic)  #calculate all pairwise similarities of intruder words and top words 
  
            avg_sim_topic_list.append(np.mean(sim))
       
        return np.array(avg_sim_topic_list)

    def score_one_intr(self, model_output, new_Embeddings = True):
        """
        Calculate the score for all topics combined but only with one intruder word
        """
        if new_Embeddings:
            self.embeddings = None
        return np.mean(self.score_one_intr_per_topic(model_output, new_Embeddings))

    def score_per_topic(self, model_output, new_Embeddings = True):
        """
        Calculate the score for each topic with several intruder words
        """
        if new_Embeddings:
            self.embeddings = None
        score_lis = []
        for _ in range(self.n_intruders): #iterate over the number of intruder words
            score_per_topic = self.score_one_intr_per_topic(model_output, new_Embeddings=False)  #calculate the intruder score, but re-use embeddings
            score_lis.append(score_per_topic)  #and append to list

        res = np.vstack(score_lis).T  #stack all scores and transpose to get a (n_topics, n_intruder words) matrix

        self.embeddings = None
        return(np.mean(res, axis = 1))  #return the mean score for each topic

    def score(self, model_output, new_Embeddings = True):
        if new_Embeddings:
            self.embeddings = None
        """
        Calculate the score for all topics combined but only with several intruder words
        """
        return float(np.mean(self.score_per_topic(model_output)))

class Embedding_Intruder_cos_sim_accuracy( ):
    """
    For each topic, draw several intruder words that are not from the same topic by first selecting some topics that are not the specific topic and 
    then selecting one word from each of those topics. 
    The embedding intruder cosine similarity accuracy for one intruder word is then calculated by the fraction of top words 
    that are least similar to the intruder 
    """

    def __init__(self, corpus_dict, n_intruders = 1, n_words = 10):
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider 
        """

        self.n_intruders = n_intruders
        self.corpus_dict = corpus_dict
        self.n_words = n_words
        self.embeddings = None

    def score_one_intr_per_topic(self, model_output, new_Embeddings = True):
        """
        Calculate the score for each topic but only with one intruder word
        """
        if new_Embeddings:
            self.embeddings = None
        topics_tw = model_output['topics']

        if self.embeddings is None:
            emb_tw = Embed_topic(topics_tw, self.corpus_dict,  self.n_words)  #embed the top words
            emb_tw = np.dstack(emb_tw).transpose(2,0,1)[:, :self.n_words, :]  #create tensor of size (n_topics, n_topwords, n_embedding_dims)
            self.embeddings = emb_tw
        else:
            emb_tw = self.embeddings #create tensor of size (n_topics, n_topwords, n_embedding_dims)

        avg_sim_topic_list = []
        for idx, topic in enumerate(emb_tw):
            mask = np.full(emb_tw.shape[0], True) #mask out the current topic
            mask[idx] = False
            
            other_topics = emb_tw[mask] #embeddings of every other topic except the current one

            intr_topic_idx = np.random.randint(other_topics.shape[0]) #select random topic index
            intr_word_idx = np.random.randint(other_topics.shape[1])  #select random word index

            intr_embedding = other_topics[intr_topic_idx, intr_word_idx] #select random word
            
            new_words = np.vstack([intr_embedding, topic])  #stack the intruder embedding above the other embeddings to get a matrix with shape ((1+n_topwords), n_embedding_dims)

            sim = cosine_similarity(new_words)  #calculate all pairwise similarities for matrix of shape ((1+n_topwords, 1+n_topwords))

            least_similar = np.argmin(sim[1:], axis = 1)  # for each word, except the intruder, calculate the index of the least similar word 
            intr_acc = np.mean(least_similar == 0)    #calculate the fraction of words for which the least similar word is the intruder word (at index 0)
  
            avg_sim_topic_list.append(intr_acc)  #append intruder accuracy for this sample

        return np.array(avg_sim_topic_list) 

    def score_one_intr(self, model_output, new_Embeddings = True):
        if new_Embeddings:
            self.embeddings = None
        self.embeddings = None
        """
        Calculate the score for all topics combined but only with one intruder word
        """
        return np.mean(self.score_one_intr_per_topic(model_output))
        
    def score_per_topic(self, model_output, new_Embeddings = True):
        """
        Calculate the score for each topic with several intruder words
        """
        if new_Embeddings:
            self.embeddings = None


        score_lis = []
        for _ in range(self.n_intruders):
            score_per_topic = self.score_one_intr_per_topic(model_output, new_Embeddings=False)
            score_lis.append(score_per_topic)
        self.embeddings = None
        res = np.vstack(score_lis).T

        return(np.mean(res, axis = 1))
      
    def score(self, model_output, new_Embeddings = True):
        if new_Embeddings:
            self.embeddings = None
        """
        Calculate the score for all topics combined but only with several intruder words
        """
        self.embeddings = None
        return float(np.mean(self.score_per_topic(model_output)))


    

class NPMI_coherence_gensim():
    """
    Compute NPMI coherence according to gensim: https://radimrehurek.com/gensim/models/coherencemodel.html
    """

    def __init__(self, corpus, vocab, coherence = 'u_mass'):
        """
        corpus: list of strings that represent document 
        coherence: type of coherence to compute
        vocab: list of all unique words in the corpus
        """
        self.corpus = corpus
        self.coherence = coherence 

        
        vocab_set = set(vocab)
        new_corpus = []
        for doc in corpus:
            new_doc = []
            for word in nltk.tokenize.word_tokenize(doc):
                if word in vocab_set:
                    new_doc.append(word)
            new_corpus.append(new_doc)
        dictionary = Dictionary(new_corpus)

        tokenized_corpus = []
        for doc in new_corpus:
            tokenized_doc = dictionary.doc2bow(doc)
            tokenized_corpus.append(tokenized_doc)

        
        self.corpus = tokenized_corpus
        self.dictionary = dictionary

    def score(self, model_output):
        """
        Compute coherence score
        """
        topics_tw = model_output['topics']
        print(self.corpus), print(self.dictionary)
        cm = CoherenceModel(topics = topics_tw, corpus = self.corpus, dictionary= self.dictionary, coherence = self.coherence)
        coherence_per_topic = cm.get_coherence_per_topic()
        coherence = np.nanmean(coherence_per_topic)
        return coherence
                

        
