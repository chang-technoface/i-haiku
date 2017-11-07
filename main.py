# -*- coding: utf-8 -*-
import gensim 
import mecab
import sys

tagger = mecab.Tagger('-F\s%f[6] -U\s%m -E\\n')

if __name__ == "__main__":
	lines = open("haiku.txt", "r")
	dictionary = gensim.corpora.Dictionary(raws)
	for line in lines:
		result = tagger.parse(line)
		print result[1:]

	lines.close()