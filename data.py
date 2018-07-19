import nltk
import numpy as np

def features(sen, index):
	f = []
	# append the actual word
	f.append(('word', sen[index][0]))
	f.append(('suff', sen[index][0][-3:]))	
	f.append(('pref', sen[index][0][:3]))	
	f.append(('is_num', sen[index][0].isdigit()))
	f.append(('is_upper', sen[index][0].upper() == sen[index][0]))
	f.append(('is_lower', sen[index][0].lower() == sen[index][0]))	
	f.append(('is_title', sen[index][0][0].upper() == sen[index][0][0] and sen[index][0].upper() != sen[index][0]))	
	
	
	#previous word in the sentence
	if(index > 0):
		f.append(('prev', sen[index-1][0]))
		# f.append(('prev_suff', sen[index-1][0][-3:]))	
		# f.append(('prev_pref', sen[index-1][0][:3]))				
	else:
		f.append(('prev', 0))
		# f.append(('prev_suff', 0))	
		# f.append(('prev_pref', 0))	

	#next word in the sentence
	if(index < len(sen)-1):
		f.append(('next', sen[index+1][0]))
		# f.append(('next_suff', sen[index+1][0][-3:]))	
		# f.append(('next_pref', sen[index+1][0][:3]))	
	else:
		f.append(('next', 0))	
		# f.append(('next_suff', 0))	
		# f.append(('next_pref', 0))		


	return f;


def load_data(corpus):
	if(corpus=='treebank'):
		nltk.download('treebank')
		nltk.download('universal_tagset')

		d = nltk.corpus.treebank.tagged_sents(tagset='universal')	
		x = []
		y = []
		for sen in d:
			for i in range(0, len(sen)):
					x.append(sen[i][0])
					y.append(sen[i][1])
	elif(corpus=='brown'):
		nltk.download('brown')
		d = nltk.corpus.brown.tagged_words()
		x = []
		y = []
		for i in range(0, len(d)):
			if(d[i][1] != ',' and d[i][1] != '.'):
				x.append(d[i][0])
				y.append(d[i][1])

	test_train_ratio = int(0.80*len(x))
	
	x_train = x[:test_train_ratio]
	y_train = y[:test_train_ratio]
	
	x_test = x[test_train_ratio:]	
	y_test = y[test_train_ratio:]

	return x_train, y_train, x_test, y_test

def load_sentences():
	nltk.download('treebank')
	nltk.download('universal_tagset')

	d = nltk.corpus.treebank.tagged_sents(tagset='universal')	
	x = []
	y = []
	label_dict = {}
	id=1;
	max=0
	for sen in d:
		t=[]
		u=[]
		for i in range(0, len(sen)):
				t.append(sen[i][0])
				u.append(sen[i][1])
				if sen[i][1] not in label_dict:
					label_dict[sen[i][1]] = id
					id+=1
		if len(sen) > max: 
			max = len(sen)
		x.append(np.array(t))
		y.append(np.array(u))

	label_id = []
	for sen in y:
		t = []
		for label in sen:
			t.append(label_dict[label])
		label_id.append(np.array(t))

	padded_x=[]
	padded_y=[]
	for x_ in x:
		padded_x.append(np.array(np.pad(x_, (0, max-x_.shape[0]), 'constant')))
	for y_ in label_id:	
		padded_y.append(np.array(np.pad(y_, (0, max-y_.shape[0]), 'constant')))
	return np.array(padded_x), np.array(padded_y), id