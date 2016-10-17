# Ada

Goal of this app is to extract information and output a summarized version of a given piece of text, image, or video. By doing so, the algorithm should produce a knowledge graph in which the user can ask more questions about the text.

Knowledge graphs should be shared between users into (multiple or single) graphs. The algorithm can then draw on more information to provide the user additional informations.

## vision
With the ability to convert data into a knowledge graph, the summarization tool is only a start. The larger goal is to convert the algorithm into a AI/ML based frontend for the web. Instead of googling things, you just ask questions and receive answers (more "ergonomic") approach.

**Step 1 :** a chrome extension that you can query.

**Step 2 :** a browser that isn't really a browser. It's just a AI that you can chat with to surf the semantic web.

**Step 3:** an operating system basically like the one from the movie "Her".

## technical pieces
- Text analysis
	- Tokenizer (Learn keywords)
	- Learn the high dimensional space that text lies in
	- Convert the features into a knowledge graph (unsupervised clustering possibly)
	- Summarization
- App
	- Chrome Extension
	- Server to store the knowledge graph (try neo4j)
