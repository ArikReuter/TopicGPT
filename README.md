# TopicGPT

Traditional Topic Models identify common topics within a given corpus and represent them as a list of words ("top-words"). 
The aim of TopicGPT is to offer not only a rich and concise topic description, but to also allow users to dynamically investigate, change and modify the identified topics. 

More specifically, TopicGPT can extract and visualize topics based on document embeddings. Furthermore, retrieval-augmented generation (RAG) based on specific topics allows to interactively find out more about specific aspects of topics of interest. 
Additionally, Topics can be split, combined and deleted based on interactive queries. 

It is further possible, to directly interact with TopicGPT via direct prompting and without explizit calling of functions (GPT decides which functionality to use). 

### Example use

In the following example, the usage of a few functions on the 20 Newsgroups corpus is demonstrated (https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html). It is assumed that embeddings of the respective corpus and the repsective vocabulary have already been computed by using the class "GetEmbeddings.py" from this repository. 

#### Extracting Topics 

```python
clusterer = Clustering.Clustering_and_DimRed() # define object to reduce dimensionality and cluster documents 
enhancer = TopwordEnhancement(openai_key = <your_openai_key>, openai_model = "gpt-4", max_context_length = 8000) # define object used for describing and naming objects

topics = TopicRepresentation.extract_and_describe_topics(
                                            corpus = corpus,  # corpus of documents to be analyzed. Is of type list[str] where each string is a document
                                            document_embeddings = embeddings,  # embeddings for each document. Is an np.ndarray of shape (n_documents, n_embedding_dimensions)
                                            clusterer = clusterer, # object to cluster documents
                                            vocab_embeddings = vocab_embeddings, # A dictionary of type dict[str, np.ndarray] where each key is a word in the vocabulary of the corpus and each value is the corresponding embedding obtained with the same embedding model as for the document embeddings. 
                                            enhancer=enhancer, # object to describe topics
                                            )
```

#### Topic-based Prompting 

```python
from TopicPrompting.TopicPrompting import TopicPrompting

pmp = TopicPrompting(
    openai_prompting_model = "gpt-4",
    max_context_length_promting = 4000,
    topic_lis = topics,
    openai_key = <your_openai_key>, 
    enhancer=enhancer,
    vocab_embeddings=vocab_embeddings
)
pmp.show_topic_list() #display list of available topics 
```

See the detailed topic description for topic 13

```python
pmp.topic_lis[13].topic_description 
```

This will execute retrieval-augmented generation based on the keyword "Jupiter" for topic 13 and tell you which information on Jupiter topic 13 contains
```python
print(pmp.prompt_knn_search(llm_query = "What information on Jupiter does topic 13 contain)) 
```
You can identify the subtopics of a given topic.
```python
pmp.general_prompt("What subtopics does topic 6 have?")
```

Based on the previous analysis, you can ask TopicGPT to actually split a topic based on the previous analysis. 
```python
pmp.general_prompt("Please actually split topic 6 into its subtopics. Do this inplace.")
```

One can also combine topics. 
```python
pmp.general_prompt("Combine the topics 19 and 20 into one single topic")
```

It is also possible to create completely new, additional topics
```python
pmp.general_prompt("Please create a new topic based on Climate Change")
```
👷‍♀️🚧👷
Note that this repository is still under developement and will be finished by 08.09.2023. 
👷‍♀️🚧👷
