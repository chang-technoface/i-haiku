# -*- coding: utf-8 -*-
import gensim 
import MeCab
import sys

#tagger = mecab.Tagger('-F\s%f[6] -U\s%m -E\\n')
tagger = mecab.Tagger('-Owakati')

if __name__ == "__main__":
	lines = open("haiku.txt", "r")
	for line in lines:
		result = tagger.parse(line)
		print result[1:]

	lines.close()
