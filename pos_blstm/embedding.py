from gensim import models
import pdb
import utils


class sentence_iter:
	def __init__(self, file_list):
		self.file_list = file_list

	def __iter__(self):
		for text_file in self.file_list:
			with open(text_file) as f:
				sentence = []
				for line in f:
					if line == '\n':
						yield sentence
						sentence = []
					else:
						word = line.split()[0]
						sentence.append(utils.format_word(word))

sentences = sentence_iter(['train.txt', 'test.txt'])
model = models.Word2Vec(sentences, min_count=1)
model.save('tmp/embedding_map')