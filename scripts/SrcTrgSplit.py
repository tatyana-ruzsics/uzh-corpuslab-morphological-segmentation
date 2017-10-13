#! /usr/bin/python
# -*- coding: utf-8 -*-

import codecs, unicodedata, sys, os

def writeData(wordlist, filehandle, line_number, segmentation=False):
	newlist = []
    #for word in wordlist:
	for letter in wordlist:
			if unicodedata.combining(letter):
				if (newlist == []) or (newlist[-1] == "_") or (newlist[-1] == "<s>"):
					print "** nothing to attach diacritic:", ("".join(wordlist)).encode('utf8'), "|", (" ".join(newlist)).encode('utf8'), "|", letter.encode('utf8'), line_number
				else:
					newlist[-1] = newlist[-1] + letter
			elif letter == " ":
				newlist.append("|")
			else:
				newlist.append(letter)
 	filehandle.write(" ".join(newlist) + "\n")


def extractData(inputfile, outputfile1, outputfile2):
	f = codecs.open(inputfile, 'r', 'utf8')
	of1 = codecs.open(outputfile1, 'w', 'utf8')
	of2 = codecs.open(outputfile2, 'w', 'utf8')
	for i,line in enumerate(f):
		elements = line.strip().lower().split("\t")
		if (elements[0] == "") or (elements[2] == ""):
			continue
		writeData(elements[0], of1, i)
		writeData(elements[2], of2, i)


if __name__ == "__main__":
	inputfile = sys.argv[1]
	outputfile1 = sys.argv[2]
	outputfile2 = sys.argv[3]
	print inputfile, "==>", outputfile1, outputfile2
	extractData(inputfile, outputfile1, outputfile2)
			
