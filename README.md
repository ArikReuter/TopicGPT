# TopicGPT
TopicGPT integrates the remarkable capabilities of current LLMs such as GPT-3.5 and GPT-4 into topic modelling. 

While traditional topic models extract topics as simple lists of top-words, such as ["Lion", "Leopard", "Rhino", "Elephant", "Buffalo"], TopicGPT offers rich and dynamic topic representations that can be intuitively understood, extensively investigated and modified in various ways via a simple text commands. 

More specifically, it provides the following core functionalities: 
- Identification of clusters of documents and top-word extraction
- Generation of detailed and informative topic descriptions 
- Extraction of detailed information about topics via Retrieval-Augmented-Generation (RAG)
- Comparison of topics
- Splitting and combining of identified topics
- Addition of new topics based on keywords
- Deletion of topics
  
It is further possible, to directly interact with TopicGPT via prompting and without explicitly calling  functions - an LLM autonomously decides which functionality to use.

## Installation

## Example 

The following example demonstrates how TopicGPT can be used on a real-world dataset. The Twenty Newsgroups corpus (https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) will be used for this purpose. 

### Load the data

```python
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')) #download the 20 Newsgroups dataset
corpus = data['data'] 

corpus = [doc for doc in corpus if doc != ""] #remove empty documents
```
### Initialize the model 

Note that an OpenAi API-Key is needed to compute the embeddings and execute the prompts. See https://platform.openai.com/account/api-keys for more details. We select 20 topics in this case since the Twenty Newsgroups corpus comprises documents from 20 different newsgroups. It is also possible to let Hdbscan determine the number of topics automatically. 

```python 
from topicgpt.TopicGPT import TopicGPT

tm = TopicGPT(
    openai_api_key = <your-openai-api-key>,
    n_topics = 20 # select 20 topics since the true number of topics is 20 
)
```

### Fit the model 

The fit-method fits the model. This can take, depending on the size of the dataset and wether embeddings have been provided, from a few minutes to several hours. Especially the computation of the embeddings can take some time. 

```python 
tm.fit(corpus) # the corpus argument should be of type list[str] where each string represents one document
```

### Inspect the found topics

Obtain an overview over the indentified topics
```python
print(tm.topic_lis)
```
_Output_
```
[Topic 0: Electronics Equipment Sales,
 Topic 1: Image Processing,
 Topic 2: Gun control,
 Topic 3: Online Privacy and Anonymity,
 Topic 4: Conflict and Violence.,
 Topic 5: Computer Hardware,
 Topic 6: Belief and Atheism,
 Topic 7: Online Discussions,
 Topic 8: Computer Software,
 Topic 9: Car Features and Performance,
 Topic 10: Encryption and Government,
 Topic 11: Technology and Computing.,
 Topic 12: Technology and Computing,
 Topic 13: Space Exploration,
 Topic 14: Motorcycle Riding Techniques,
 Topic 15: Technology,
 Topic 16: Hockey Games,
 Topic 17: Health and Medicine.,
 Topic 18: Baseball games and teams.,
 Topic 19: Beliefs about Homosexuality.]
```
To obtain more detailed information on each topic, we can call the "print_topics" method: 

```python
tm.print_topics()
```
_Output_
```
Topic 0: Electronics Equipment Sales

Topic_description: The common topic of the given words appears to be "electronics and technology". 

Various aspects and sub-topics of this topic include:
1. Buying and selling: "offer", "sale", "sell", "price", "buy"
2. Device usage and features: "use", "get", "new", "used", "condition"
3. Technical specifications: "wire", "ground", "power", "circuit", "voltage"
4. Communication and connectivity: "phone", "email", "modem", "wireless", "connection"
5. Accessories and peripherals: "battery", "cable", "manuals", "disk", "monitor"
Top words: ["n't", 'one', 'would', 'use', 'like', 'get', 'new', 'used', 'offer', 'sale']

[...]
```
We can also visualize the resulting clusters to get an overview of the shape and size of the clusters
```
tm.visualize_clusters()
```

### Find out more detailed information about the identified topics

First, we might be interested in knowing what information the space topic (topic 13) contains on the moon landing. 

```python 
tm.pprompt("Which information related to the keyword 'moon landing' does topic 13 have?")
```

_Output_
```
GPT wants to the call the function:  {
  "name": "knn_search",
  "arguments": "{\n  \"topic_index\": 13,\n  \"query\": \"moon landing\",\n  \"k\": 5\n}"
}
Topic 13, which is related to the keyword "moon landing," has the following information:

1. Document index 258: This document provides an introduction to the solar system and mentions that advancements in rocketry after World War II enabled machines to travel to the Moon and other planets. It highlights that the United States has sent both automated spacecraft and human-crewed expeditions to explore the Moon.

2. Document index 535: This document discusses a $65 million program called the Back to the Moon bill, which aims to encourage private companies to develop lunar orbiters. It mentions that there is a chance of making a lunar mission happen in this decade through this program.

3. Document index 357: This document is a request for more information on a recent newspaper article about the Japanese crashing or crash-landing a package on the Moon. It indicates that the article was vague and unclear.

4. Document index 321: This document speculates about what would have happened if the Soviets had beaten the United States in the Moon race. It suggests that the US would have still performed Moon landings and potentially set up a lunar base. The focus on Mars exploration would have depended on the Soviets' actions.

5. Document index 102: This document mentions the Hiten engineering-test mission, which spent time in a highly eccentric Earth orbit and performed lunar flybys before being inserted into lunar orbit using gravity-assist-like maneuvers. It states that the mission was expected to crash on the Moon eventually.

Please note that the above summaries are based on the content of the documents and may not capture all the information contained within them.
```

From this output we see that an instance of a GPT decided to call the function "knn_search" from the class "TopicPrompting". Indeed some documents on the topic "moon landing" have been found and the model summarizes the relevant information accordingly. 

If we want to check, for instance the document with index 102 in topic 13 to learn more about the Hiten engineering-test mission, we can simply do the following:

```python
print(tm.topic_lis[13].documents[535])
```
_Output_
```
Their Hiten engineering-test mission spent a while in a highly eccentric Earth orbit doing lunar flybys, and then was inserted into lunar orbit using some very tricky gravity-assist-like maneuvering.  This meant that it would crash on the Moon eventually, since there is no such thing as a stable lunar orbit (as far as anyone knows), and I believe I recall hearing recently that it was about to happen.
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
print(pmp.prompt_knn_search(llm_query = "What information on Jupiter does topic 13 contain?")) 
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
## How TopicGPT works

## Disclaimer

## References

Please note that the topword extraction methods used for this package are based on similar ideas as found in the Bertopic Model (Grootendorst, Maarten. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." arXiv preprint arXiv:2203.05794 (2022)) in the case of the tf-idf method and in Top2Vec for the centroid-similarity method (Angelov, Dimo. "Top2vec: Distributed representations of topics." arXiv preprint arXiv:2008.09470 (2020)).


👷‍♀️🚧👷
Note that this repository is still under developement and will be finished by 08.09.2023. 
👷‍♀️🚧👷
