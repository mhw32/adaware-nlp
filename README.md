# smart-summary

Goal of this app is given a piece of text, image, or video, extract information, and output a summarized version. By doing so, the algorithm should produce a knowledge graph where the user can ask more questions about the text. 

Knowledge graphs should be shared between users into (multiple or single) graphs. The algorithm can then draw on more information to provide the user additional informations. 

## vision
With the ability to convert data into a knowledge graph, the summarization tool is only a start. The larger goal is to convert the algorithm into a AI/ML based frontend for the web. Instead of googling things, you just ask questions and receive answers (more "ergonomic" approach.

## technical pieces
- Text analysis
	- Tokenizer (Learn keywords)
	- Learn the high dimensional space that text lies in
	- Convert the features into a knowledge graph (unsupervised clustering possibly)
	- Summarization
- App 
	- Chrome Extension
	- Server to store the knowledge graph (try neo4j)
	- Get

## summarization
1. Extraction-based
	 - Gets objects w/o modifying them
	 - Keyphrase extraction: select individual raw words/phrases 
2. Abstraction-based
	- Same as extraction but paraphrases sections
	- Condenses text better but harder
3. Entropy-based
4. Aid-based
	- You help it summarize

It sounds like we need abstraction-based and take advantage of ML/NLP algorithms to do this.

### single-document extraction
The very naive approach is to use keyphrase extraction with "metrics" like tf*idf, distance, spread, structure, wiki-measure, metadata as features in some supervised problem. 

Topic-driven summarization: summary made based on a topic. Can we combine this in an unsupervised manner?

#### early work
- sorted frequency of words
- significance factor for # of occurrences within sentence to get linear distance between sentences
- top significant sentences are used to form abstract
- sentence position: aka topic sentence is first or last sentence
- cue words aka "significant", "hardly"

#### datasets
- TREC dataset
- CNN dataset
- ROUGE-1 dataset

#### baseline models
- take first n sentences of article

#### naive ML models
- naive bayes to pick the probability of each sentence being worthy of extraction
	- assumes independence between features F1...Fk
	- p(sentence|features) = prod(p(feature\_i|sentence)*p(sentence)) / prod(p(feature\_i))
	- top n sentences are extracted
	- position, cue features, sentence length
	- term freq, inverse document freq to derive signature words (tf-idf)
- if features are not iid, then try decision-trees (capable of learning non-linear relationships)
	- TIPSTER-SUMMAC
	- query signature = normalized score based on number of query words that a sentence contains
	- IR signiature = m most salient words (basically same)
	- numerical data = boolean value 1 if sentence contains a number
	- proper name = boolean value 1 if sentence contains a proper name
	- pronoun or adjective
	- weekday or month
	- quotation

#### generative ML models
- learn local dependencies between sentences using HMM (i like this one)
	- 3 features: position of sentence in doc, number of terms in sentence, likeliness of sentence terms given doc terms
	- 2s+1 states: s summary states and s+1 nonsummary states
	- "hesistation" in nonsummary states and "skipping" in summary states
	- learn a transition matrix
	- assumed features are multivariate normal

#### neural network models
- RankNet: pair based nn to rank a set of inputs using sgd

#### deep nlp models
- separation between learning semantic structure of text vs word statistics of document
- not really ml, more using set of heuristics to create document extracts (i don't reall like this)
	- lexical chain: sequence of related words (could be adjacent or long-distance)
	- segmentation of text
	- get lexical chains
	- strong lexical chains for extraction
	- WordNet for lexical chains
		- select a set of candidate words
		- for each word, find chain based on relatedness (Wordnet distance)
		- if found, insert into chain
	- define a strong lexical chain based on length and homogeneity
- rhetorical structure (semantics argument)
	- binary tree with relations between chunks of sentences
		- sentence analysis
		- rhetorical relation extraction
		- segmentation
		- candidate generation
		- preference judgement

### multi-document extraction
Get a summary from multiple documents. More than 1 source of info that overlap and supplement each other. How to measure redundancy? How to recognize novelty? This would be pretty cool to do.

- cluster similar sentences together. aka word2vec and then k-means. Select centroid to represent cluster. Or generate composite sentence from each cluster.

- then just do it as a single doc extraction

### abstraction

Two-steps: (1) processing full text as input to make template slots; (2) creating a summary from extracted info.

- content planner: selects info to include in summary using input templates (kind of like extraction but you need to build a graph)
- linguistic generator: selects the right words to express info in a grammatical way

#### existing generators
- FUF / SURGE system

- summary operators : set of heuristic rules thst perform operations like
	- change of perspective
	- contradiction
	- refinement

- hard to generalize this to large domains but we can solve this by only using select documents that are similar (knowledge graph)

- identify themes (similar paragraphs)	
	- text is mapped to vectors, single words are weighted by TF-IDF scores, noun, proper noun, synsets from Wordnet
	- vector for each pair of paragraphs
	- decide if pairs are similar or dissimilar
	- end up with similar paragraphs in themes
- information fusion
	- which sentences of a theme should be included
		- Collins' statistical parser
		- put into dependency trees (predicate-arg)
		- drop determiners and auxiliaries
- FUF/SURGE
	- generate grammatical text 

### topic-driven summarization
- maximal marginal relvance (MMR)
	- combines query relevance and info novelty
	- rewards relevant sentences and penalizes redundant ones 
	- linear combo of 2 similarity measures
	- lambda param to balance relevance and redundancy (just a regularizer)
	-document w/ highest MMR is selected for summary; do this until minimum threshold is attained
	- can shift lambdas in order to get what the user wants
	- requires a query Q s.t. different user with different profile generates a different summary

### graph-spreading activation
- no textual summary is generated but summary content is represented as nodes and edges.
- I think this is what we want
- detect salient regions of a graph 
	- topic is a set of entry nodes
	- convert document into graph: each node represents occurrence of a single word
	- each node has several links:
		- adjacency links to adjacent words in text 
		- same links to other occurrences of the same word
		- alpha links encoding semantic relationships using WordNet
		- phrase links tie together sequences of adjacent nodes that are in the same phrase
		- name / coref links for co-referential name occurrences
	- find topic nodes using stem comparison (these become the entry nodes)
	- search for semantically related text using BFS (spreading activation)
	- salient word and phrases initialized using TF-IDF score
	- weight of neighboring nodes dependent on the node link travelled and is exponentially decaying based on distance of path
	- travelling within a sentence is cheaper than across sentences, which is cheaper than across paragraphs
	- Given two documents, common nodes are identified
		- for each sentence, get a score for avg weight of common nodes and a score for average weights of difference nodes
		- sentences with higher common and different scores are highlighted. 
		- compose abstractive summaries using these nodes (something we can do)

### centroid-based summary
- MEAD system
- no language generation module; only uses bags-of-words; domain-independent
	- topic detection: group together news articles that describe the same event
		- cluster of TF-IDF vector representations of documents (and keep track of centroids): this is basically KNN
	- use centroids to find sentences in each cluster that are important to the topic of that cluster
		- CBRU (cluster-based relative utility): how relevant a sentence is to topic of cluster
		- CSIS (cross-sentence informational subsumption): measure of redundancy
		- these two metrics are not query-dependent (unlike MMR)!
	- cluster C of docs segemented into n sentences with R compression rate gives us a sequence of nR sentences. 
		- for each sentence, get centroid value (sum of centroid for all words in sentences)
		- positional value (make leading setences more important)
		- first-sentence overlap (inner pdt between word occurrence vector and firest setence)
	- final score = combo of 3 scores + redundancy penalty for overlapping

## knowledge-graph
Instead of just looking at the link provided by the user, the AI should amass knowledge over time. After many uses, the algorithm should be able to prioritize the currently article but optionally supplant it with information from previously read articles to give a holistic point of view. 

To prevent this from exploding unintentionally, similar to the graph used in activation summarization, it is necessary to create a graph of article similarities. This may be an extension to the already existing graph but it could also be a separate instance.  

A centroid-based organization may be good. Or we also do a graph-spreading but not on words but documents, it may be good. 

From there, if you do BFS on a graph, then the subset of nodes (docs) can be used in a multi-document abstraction problem. 

## question & answer system
An interesting feature of this system is the ability to answer questions about the summarized article. (In the future, this should be answer questions about anything). 

Required additional algorithms:

- Given an question, parse it for a semantic structure. A question should be analyzable as a small piece of text using the same graph algorithm.
- Find the k-nearest neighbors to the question in our knowledge graph.
- Form an answer ... this is different than form a summary but should share a similar procedure. Or we can view it as summarizing the relevant areas of the articles.

## state-of-development

- document graph (not started)
- knowledge graph (not started)
- q&a system (not started)
- chrome extension (not started)
- server (not started)
