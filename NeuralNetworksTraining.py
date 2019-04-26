import numpy as np
import tensorflow as tf








def create_featureset_and_labels(pos, neg, test_size=0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('Pos.txt', lexicon, [1, 0])
	features += sample_handling('Neg.txt', lexicon, [0, 1])
	random.shuffle(features)

	features = np.array(features)
	testing_size = int(test_size * len(features))


	train_x = list(features[:, 0][: - testing_size])
	train_y = list(features[:, 1][: - testing_size])

	test_x = list(features[:, 0][- testing_size:])
	test_y = list(features[:, 1][- testing_size:])

	return train_x, train_y, test_x, test_y
































import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100


# hight < witdh // , [None, 784] NOT NESSESARY!
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')



class Own_NN:
	def __init__(self, batch_size=100, n_classes=8, hm_epochs=10, max_nodes=300, max_hidden_layers=3):
		self.batch_size = batch_size
		self.n_classes = n_classes
		self.hm_epochs = hm_epochs
		self.max_nodes = max_nodes
		self.max_hidden_layers = max_hidden_layers
		n_nodes = [0, 784, 500, 500, 500]
     	#   	  |___ dummy value so that n_nodes[i] and n_nodes[i+1] stores
     	#  		        the input and hidden number of the i-th hidden layer
     	#  		        (1-based) because layers[0] is the input.




	def neural_network_model(self, data, n_nodes):
	   layers = []*len(n_nodes)
	   layers[0] = data
	   for i in in range(1, n_nodes-1):
	     hidden_i = make_hidden(n_nodes[i], n_nodes[i+1]
	     layers[i] = tf.add(tf.matmul(layers[i-1], hidden_i['weight']), hidden_i['biases'])
	     layers[i] = tf.nn.relu(layers[i])

	   output_layer = make_output(n_nodes[-1], n_classes)
	   output = tf.matmul(layers[-1], output_layer['weight']) + output_layer['biases']

	   return output


def train_neural_network(self, x):
	self.x = x
	data_length = len(x)
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))

	# AdamOptimizer learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles feed forward + backprop


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch:', epoch, 'Completed out of:', hm_epochs, 'Loss:', epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)
