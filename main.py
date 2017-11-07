# -*- coding: utf-8 -*-
import gensim 

if __name__ == "__main__":
	raws = open("haiku.txt", "r")
	for line in raws:
		print line