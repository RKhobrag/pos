import numpy as np
from gensim.models import Word2Vec
class embeddings:
	embeddings_index = {}
	words_dict = {}
	label_dict = {}	

	def create_embedding_index(self, path):
		word_embeddings = open(path)
		i=0
		for line in word_embeddings:
		    values = line.split()
		    word = values[0]
		    try:
		    	coefs = np.asarray(values[1:], dtype='float32')
		    	pass
		    except :
		    	pass
		    	
		    self.embeddings_index[word] = coefs
		    i+=1
		word_embeddings.close()
		return self.embeddings_index

	def embed(self, path, words, labels):
		self.create_embedding_index(path)
		emb_x=[]
		emb_y=[]
		for i in range(len(words)):
			if words[i] in self.embeddings_index:
				emb_x.append(self.embeddings_index[word[i]])
				emb_y.append(self.embeddings_index[labels[i]])

		return emb_x, emb_y

	def make_dictionary(self, words, labels, x_test, y_test):
		
		id=1
		for w in words + x_test:
			if w not in self.words_dict:
				self.words_dict[w]=id
				id+=1
		id=1
		for l in labels + y_test:
			if l not in self.label_dict:
				self.label_dict[l] = id
				id+=1

		for i in range(len(words)):
			words[i] = self.words_dict[words[i]]
			labels[i] = self.label_dict[labels[i]]


		for i in range(len(x_test)):
			x_test[i] = self.words_dict[x_test[i]]
			y_test[i] = self.label_dict[y_test[i]]

		return words, labels, x_test, y_test

	def train_embedding(self, sentences):
		model = Word2Vec(sentences, min_count=1)
		words = list(model.wv.vocab)

		x_train_emb = []

		for w in sentences:			
			if w in model:
				x_train_emb.append(np.array(model[w]))
			else:
				x_train_emb.append(np.zeros(100))
		# for w in x_train:
		# 	if w in model:
		# 		x_train_emb.append(np.array(model[w]))
		# 	else:
		# 		x_train_emb.append(np.zeros(100))

		# x_test_emb = []
		# for w in x_test:
		# 	if w in model:
		# 		x_test_emb.append(np.array(model[w]))
		# 	else:
		# 		x_test_emb.append(np.zeros(100))

		return np.array(x_train_emb)