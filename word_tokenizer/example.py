from treebank_tokenizer import TreeBankTokenizer

sentences = [
    "Good muffins cost $3.88 in New York",
    "Please buy me two of them, thanks.",
    "I've had a number of interesting conversations",
    "He's an (ex--)Intel engineer"
]

tok = TreeBankTokenizer()
for sentence in sentences:
    print(tok.tokenize(sentence))
