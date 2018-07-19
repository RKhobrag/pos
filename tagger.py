import nltk
import data
import embed
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

num_units = 100
learning_rate= 0.001
batch_size = 50 

def train_using_lstm(x_train, y_train, batch_size, n_classes, time_steps):
	# print(x_train.shape)	#80540*100
	
	out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
	out_bias = tf.Variable(tf.random_normal([n_classes]))

	x = tf.placeholder("float", [batch_size, time_steps, 100])
	y = tf.placeholder(tf.int64, [batch_size, time_steps])

	# input = tf.unstack(x, time_steps, 1)

	cell = rnn.LSTMCell(num_units, state_is_tuple=True)
	outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype="float32")

	context_rep = tf.reshape(outputs, [-1, num_units])
	pred = tf.matmul(context_rep, out_weights) + out_bias
	logits = tf.reshape(pred, [-1, time_steps, n_classes])

	# labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

	losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
	mask = tf.sequence_mask(time_steps)
	losses = tf.boolean_mask(losses, mask)
	loss = tf.reduce_mean(losses)

	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)

	prediction = tf.equal(tf.argmax(logits, 2), y)
	print(logits)
	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		i = 1;
		prev_i = 0;

		while(batch_size*i<x_train.shape[0]):
			x_train_batch = x_train[prev_i:batch_size*i,:,:]
			y_train_batch = y_train[prev_i:batch_size*i,:]

			sess.run(train_op, feed_dict={x: x_train_batch, y: y_train_batch})
			
			acc = sess.run(accuracy, feed_dict={x: x_train[:batch_size,:,:], y: y_train[:batch_size,:]})
			print("Accuracy: ", acc)
			print("\n")
			prev_i = batch_size*i
			i+=1

def main():
	#load data 
	# x_train, y_train, x_test, y_test = data.load_data('treebank')

	sentences, labels, n_classes = data.load_sentences()
	n_input = sentences.shape[0]
	time_steps = sentences.shape[1]

	embeddings = embed.embeddings()
	x_train = embeddings.train_embedding(sentences.reshape(n_input*time_steps))
	x_train = x_train.reshape(n_input, time_steps, 100)
	print(x_train.shape)


	# # print(x_train.shape)
	# #replace words with embeddings
	# # path = 'glove.840B.300d.txt'
	# # x_train, y_train = embeddings.embed(path, x_train, y_train)

	# # x_train_id, y_train_id, x_test_id, y_test_id = embeddings.make_dictionary(x_train, y_train, x_test, y_test)
	# x_train = x_train.reshape(batch_size, time_steps, 100)
	train_using_lstm(x_train, labels, 400, n_classes, time_steps)
	
if __name__ == '__main__':
	main()