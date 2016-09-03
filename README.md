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

### approaches
The very naive approach is to use keyphrase extraction with "metrics" like tf*idf, distance, spread, structure, wiki-measure, metadata as features in some supervised problem. 

Topic-driven summarization: summary made based on a topic. Can we combine this in an unsupervised manner?

### early work
- sorted frequency of words
- significance factor for # of occurrences within sentence to get linear distance between sentences
- top significant sentences are used to form abstract
- sentence position: aka topic sentence is first or last sentence
- cue words aka "significant", "hardly"

### datasets
- TREC dataset
- CNN dataset
- ROUGE-1 dataset

### baseline models
- take first n sentences of article

### naive ML models
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

### generative ML models
- learn local dependencies between sentences using HMM (i like this one)
	- 3 features: position of sentence in doc, number of terms in sentence, likeliness of sentence terms given doc terms
	- 2s+1 states: s summary states and s+1 nonsummary states
	- "hesistation" in nonsummary states and "skipping" in summary states
	- learn a transition matrix
	- assumed features are multivariate normal

### neural network models
- RankNet: pair based nn to rank a set of inputs using sgd

### deep nlp models
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
