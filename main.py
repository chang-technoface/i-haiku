# -*- coding: utf-8 -*-
#import gensim 
import MeCab
import sys
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

#tagger = mecab.Tagger('-F\s%f[6] -U\s%m -E\\n')
#tagger = MeCab.Tagger('-Owakati')
tagger = MeCab.Tagger()

if __name__ == "__main__":
	vocabulary = set()
	keywords = []
	sentences = []
	
	lines = open("haiku.txt", "r", encoding="utf-8")
	for line in lines:
		line = line.replace("\n","")
		vocabulary = vocabulary.union(line)
		sentences.append(list(line))	
		node = tagger.parseToNode(line)
		while node:
			if len(node.surface) == 0:
				node = node.next
				continue
			if node.feature.split(",")[0] == u"名詞":
				keywords.append(node.surface)
			node = node.next

	lines.close()

	#print( keywords )
	#print( "keywords	size :" + str(len(keywords)) )

	vacabulary = set(vocabulary)
	print( vocabulary )
	print( "vocabulary size :" + str(len(vocabulary)) )

	#print( sentences )
	
	word2int = {}
	int2word = {}
	for i, word in enumerate(vocabulary):
		word2int[word] = i
		int2word[i] = word
	#print(word2int['ん'])
	#print(int2word[10])

	W_SIZE = 2
	V_SIZE = len(vocabulary)

	data = []
	for s in sentences:
		for i, word in enumerate(s):
			for n in s[max(i-W_SIZE,0):min(i+W_SIZE,len(s))+1]:
				if n != word:
					data.append([word, n])
	#print(data)
				
	def one_hot(idx, size):
		tm = np.zeros(size)
		tm[idx] = 1
		return tm

	x_train = []
	y_train = []
	
	for wd in data:
		x_train.append(one_hot(word2int[wd[0]], V_SIZE))
		y_train.append(one_hot(word2int[wd[1]], V_SIZE))
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	
	#print(x_train)
	print(x_train.shape, y_train.shape)

	x = tf.placeholder(tf.float32, shape=(None, V_SIZE))
	y = tf.placeholder(tf.float32, shape=(None, V_SIZE))

	E_DIM = 5
	W1 = tf.Variable(tf.random_normal([V_SIZE, E_DIM]))
	B1 = tf.Variable(tf.random_normal([E_DIM]))
	H1 = tf.add(tf.matmul(x,W1),B1)

	W2 = tf.Variable(tf.random_normal([E_DIM, V_SIZE]))
	B2 = tf.Variable(tf.random_normal([V_SIZE]))
	P1 = tf.nn.softmax(tf.add(tf.matmul(H1, W2),B2))

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	cel = tf.reduce_mean(-tf.reduce_sum(
					y * tf.log(P1),
					reduction_indices=[1]))
	step = tf.train.GradientDescentOptimizer(0.1).minimize(cel)
	for _ in range(10000):
		sess.run(step, feed_dict={x: x_train, y: y_train})
		#print('loss :', sess.run(cel, feed_dict={x:x_train,y:y_train}))

	#print(sess.run(W1))
	#print(sess.run(B1))

	vectors = sess.run(W1+B1)

	def dist(v1, v2):
		return np.sqrt(np.sum((v1-v2)**2))

	def closest(idx, vs):
		min_d = 10000
		min_i = -1
		query_v = vs[idx]
		for idx, vec in enumerate(vs):
			d = dist(vec, query_v)			
			if d < min_d and not np.array_equal(vec, query_v):
				 min_d = d
				 min_i = idx
		return min_i

	model = TSNE(n_components=2, random_state=0)
	np.set_printoptions(suppress=True)
	vectors = model.fit_transform(vectors)

	from sklearn import preprocessing
	normalizer = preprocessing.Normalizer()
	vectors = normalizer.fit_transform(vectors, '12')
	print(vectors)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	for word in vocabulary:
		print (word, vectors[word2int[word]][1])
		ax.annotate(word, (vectors[word2int[word]][0],
											 vectors[word2int[word]][1]))
	plt.show()	
					
