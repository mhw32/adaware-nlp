from gensim import models
import pickle
import utils
import pdb
import numpy as np
import pdb

model = models.Word2Vec.load('tmp/embedding_map')

X_train = []
Y_train = []
X_test = []
Y_test = []
capitalization_train = []
capitalization_test = []


def cap_vector(word): # returns vector (x,y,z), where x = 1 if word is all lowercase, y = 1 if all uppercase, z = 1 if leads with capital
	x = int(word.lower() == word)
	y = int(word.upper() == word)
	z = int(word[0].upper() == word[0])
	return np.array((x,y,z))

try:
	with open('tmp/one_hot_list') as f:
		one_hot = pickle.load(f)
		print('loaded one_hot from pickle file')
except:
	one_hot = []
	print('new one_hot')

max_sentence = 0

with open('train.txt') as f:
	X = []
	Y = []
	cap = []
	for line in f:
		if line == '\n':
			if len(X) > max_sentence:
				max_sentence = len(X)
			X_train.append(X)
			Y_train.append(Y)
			capitalization_train.append(cap)
			X = []
			Y = []
			cap = []
		else:
			word, pos = line.split()[:2]
			X.append(model[utils.format_word(word)])
			if pos not in one_hot:
				one_hot.append(pos)
			Y.append(one_hot.index(pos))
			cap.append(cap_vector(word))


with open('test.txt') as f:
	X = []
	Y = []
	cap = []
	for line in f:
		if line == '\n':
			if len(X) > max_sentence:
				max_sentence = len(X)
			X_test.append(X)
			Y_test.append(Y)
			capitalization_test.append(cap)
			X = []
			Y = []
			cap = []
		else:
			word, pos = line.split()[:2]
			X.append(model[utils.format_word(word)])
			if pos not in one_hot:
				one_hot.append(pos)
			Y.append(one_hot.index(pos))
			cap.append(cap_vector(word))

train_mask = np.array([len(x) for x in X_train]) # value is first index of mask
X_train = np.array([np.vstack((np.array(x), np.zeros((max_sentence-len(x), 100)))) for x in X_train])
Y_train = np.array([
	np.vstack(
		([np.eye(1,M=len(one_hot),k=x).ravel() for x in y], 
		np.zeros((max_sentence - len(y),len(one_hot)))))
	 for y in Y_train])
capitalization_train = np.array([
	np.vstack((np.array(x),np.zeros((max_sentence-len(x),3)))) for x in capitalization_train
	])
# pdb.set_trace()
test_mask = np.array([len(x) for x in X_test]) # value is first index of mask
X_test = np.array([np.vstack((np.array(x), np.zeros((max_sentence-len(x), 100)))) for x in X_test])
Y_test = np.array([
	np.vstack(
		([np.eye(1,M=len(one_hot),k=x).ravel() for x in y], 
		np.zeros((max_sentence - len(y),len(one_hot)))))
	 for y in Y_test])
capitalization_test = np.array([
	np.vstack((np.array(x),np.zeros((max_sentence-len(x),3)))) for x in capitalization_test
	])

with open('tmp/one_hot_list','w') as f:
	pickle.dump(one_hot, f)



path = 'tmp/'
np.save(path + 'X_train.npy', X_train)
np.save(path + 'Y_train.npy', Y_train)
np.save(path + 'X_test.npy', X_test)
np.save(path + 'Y_test.npy', Y_test)
np.save(path + 'cap_train.npy', capitalization_train)
np.save(path + 'cap_test.npy', capitalization_test)
np.save(path + 'train_mask.npy', train_mask)
np.save(path + 'test_mask.npy', test_mask)

print len(one_hot)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(capitalization_train.shape)
print(capitalization_test.shape)
print(train_mask.shape)
print(test_mask.shape)