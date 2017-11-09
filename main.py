# -*- coding: utf-8 -*-
#import gensim 
import MeCab
import sys

#tagger = mecab.Tagger('-F\s%f[6] -U\s%m -E\\n')
#tagger = MeCab.Tagger('-Owakati')
tagger = MeCab.Tagger()

if __name__ == "__main__":
	vocabulary = []
	keywords = []
	lines = open("haiku.txt", "r", encoding="utf-8")
	for line in lines:
		#print (line)
		node = tagger.parseToNode(line)
		while node:
			if len(node.surface) == 0:
				node = node.next
				continue
			if node.feature.split(",")[0] == u"名詞":
				keywords.append(node.surface)

			vocabulary.append(node.surface)
			node = node.next

	print( keywords )
	print( vocabulary)
	lines.close()
