# -*- coding: utf-8 -*-
#import gensim 
import MeCab
import sys
import numpy as np
import tensorflow as tf

#tagger = mecab.Tagger('-F\s%f[6] -U\s%m -E\\n')
#tagger = MeCab.Tagger('-Owakati')
tagger = MeCab.Tagger()

if __name__ == "__main__":
	vocabulary = set()
	keywords = []
	sentences = []
	
	lines = open("haiku.txt", "r", encoding="utf-8")
	for line in lines:
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
	#print( "keywords   size :" + str(len(keywords)) )

	vacabulary = set(vocabulary)
	print( vocabulary )
	print( "vocabulary size :" + str(len(vocabulary)) )

	print( sentences )
	
	word2int = {}
	int2word = {}
	for i, word in enumerate(vocabulary):
		word2int[word] = i
		int2word[i] = word
	print(word2int['ん'])
	print(int2word[10])

	data = []
	W_SIZE = 2
	for s in sentences:
		for i, word in enumerate(s):
			for n in s[max(i-W_SIZE,0):min(i+W_SIZE,len(s))+1]:
				if n != word:
					data.append([word, n])
	print(data)
				

		
