# -*- coding: utf-8 -*-
import gensim 
import MeCab
import sys

tagger = MeCab.Tagger('-F\s%f[6] -U\s%m -E\\n')

if __name__ == "__main__":
	lines = open("haiku.txt", "r")
	for line in lines:
		result = tagger.parse(line)
		print result[1:]

	lines.close()
