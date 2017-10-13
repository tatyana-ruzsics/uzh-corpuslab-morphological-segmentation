#! /usr/bin/python
# -*- coding: utf-8 -*-
"""This script builds dictionary from inputfile: an input file to NMT. 
    I.e. in the chaarcter-based setting dictionary has a form {letter:index}.
"""

import codecs, unicodedata, sys, os, logging, argparse

def process_char(inputfile, dict_file, capitalized, numbers):
    vocab_dict = {}
    f = codecs.open(inputfile, 'r', 'utf8')
    d = codecs.open(dict_file, 'w', 'utf8')
# create dictionary of all letters in the corpus with counts
    for i,line in enumerate(f):
        #words = line.strip().lower().split(" ")
        words = line.strip().split(" ") #Capital letters become separate chars
        for word in words:
            if word not in vocab_dict.keys():
                vocab_dict[word] = 1
            else:
                vocab_dict[word] += 1
    if numbers:
        for k in range(10):
            if str(k) not in vocab_dict.keys():
                vocab_dict[k] = 1
    if capitalized:
        # create dictionary which includes all capitalized letters
        vocab_cap = {}
        for word in vocab_dict.keys():
            vocab_cap[word] = 1
            word_cap = word.upper()
            if word_cap not in vocab_cap.keys():
                vocab_cap[word_cap] = 1
        # give number to each vocab item (new dict) and print vocabulary to the output file
        vocab = {'UNK': 0, '<s>': 1, '</s>': 2}
        for j, word in enumerate(vocab_cap.keys()):
            vocab[word] = j + 3
#            print word
    else:
        # give number to each vocab item (new dict) and print vocabulary to the output file
        vocab = {'UNK': 0, '<s>': 1, '</s>': 2}
        for j, word in enumerate(vocab_dict.keys()):
            vocab[word] = j + 3
#            print word

    print "Vocabulary size of " + str(dict_file) + " :"
    print len(vocab)

    for j, word in enumerate(vocab.keys()):
        d.write('%s\t%s\n' %(word,vocab[word]))

def process_morf(inputfile, dict_file):
    vocab_dict = {}
    f = codecs.open(inputfile, 'r', 'utf8')
    d = codecs.open(dict_file, 'w', 'utf8')
    # create dictionary of all letters in the corpus with counts
    for i,line in enumerate(f):
        morpfs = line.strip().split(" | ") # Split word into morphemes
        for m in morpfs:
            chars = m.split(" ")
            morf = "".join(chars)

            if morf not in vocab_dict:
                vocab_dict[morf] = 1
            else:
                vocab_dict[morf] += 1
# give number to each vocab item (new dict) and print vocabulary to the output file
    #vocab = {'UNK': 0, '<s>': 1, '</s>': 2, '|': 3}
    vocab = {'UNK': 0, '<s>': 1, '</s>': 2}
    for j, word in enumerate(vocab_dict.keys()):
        vocab[word] = j + 3
        #vocab[word] = j + 4
        #print word

    print "Vocabulary size:"
    print len(vocab)
    
    for j, word in enumerate(sorted(vocab.keys())):
        d.write('%s\t%s\n' %(word,vocab[word]))


def load_dict(path):
    with codecs.open(path, 'r', 'utf8') as f:
        d = dict(line.strip().split(None, 1) for line in f)
        for (s, i) in [('<s>', '1'), ('</s>', '2')]:
            if not s in d or d[s] != i:
                logging.warning("%s has not ID %s in word map %s!" % (s, i, path))
    return d

def process_m2c_map(inputfile, char_dict_file, morf_dict_file, morf2char_map, outputfile):
    '''This function produces a file with a map morf ID to chars IDs (morf2char_map)
        and prepares input for LM model running on morphemes IDs (outputfile)
        '''
    map_dict = {}
    f_in = codecs.open(inputfile, 'r', 'utf8')
    f_m2c = codecs.open(morf2char_map, 'w', 'utf8')
    f_out = codecs.open(outputfile, 'w', 'utf8')
    d_char = load_dict(char_dict_file)
    d_morf = load_dict(morf_dict_file)
    
    
    mapping = {}
    # process input file for LM: map all morphemes on target side to integers
    for i,line in enumerate(f_in):
        #border_morf = d_morf["|"]
        #mapping[d_morf["|"]] = d_char["|"]
        morfs = line.strip().split(" | ") # Split word into morphemes
        morfs_to_int = [] # list of integers to map morphemes in the current line
        for i,m in enumerate(morfs): #concatenate the chars inside morphemes
            chars = m.split(" ")
            morf = "".join(chars)
            morfs_to_int.append(d_morf[morf])
            if d_morf[morf] not in mapping:
                mapping[d_morf[morf]] = " ".join([d_char[c] for c in chars])
                #if i!=len(morfs)-1:
                #morfs_to_int.append(border_morf)
        f_out.write('%s\n' %" ".join(morfs_to_int))

    # create mapping file
    for m in mapping.keys():
            f_m2c.write('%s\t%s\n' %(m,mapping[m]))

#
#def process_hybrid(inputfile, dict_file):
#    vocab_dict = {}
#    f = codecs.open(inputfile, 'r', 'utf8')
#    d = codecs.open(dict_file, 'w', 'utf8')
#    # create dictionary of all letters in the corpus with counts
#    for i,line in enumerate(f):
#        #chars = line.strip().split(" ") # Capital letters become separate chars
#        #word_joint = "".join(chars) # Initial segmentation of the word
#        #morpfs = word_joint.split("|") # Split word into morphemes
#        morpfs = line.strip().split(" | ") # Split word into morphemes
#        for m in morpfs:
#            chars = m.split(" ")
#            morf = "".join(chars)
#            #for char in m:
#            for char in chars:
#                if char not in vocab_dict:
#                    vocab_dict[char]={}
#                    vocab_dict[char][morf] = 1
#                else:
#                    if m not in vocab_dict[char]:
#                        vocab_dict[char][morf] = 1
#                    else:
#                        vocab_dict[char][morf] += 1
#
#    # create dictionary which includes all capitalized letters
#    vocab_cap = {}
#    for char in vocab_dict.keys():
#        vocab_cap[char]={}
#        for m in vocab_dict[char].keys():
#            vocab_cap[char][m] = 1
#        char_cap = char.upper()
#        if char_cap not in vocab_cap.keys():
#            for m in vocab_dict[char].keys():
#                vocab_cap[char_cap] = {}
#                vocab_cap[char_cap][m] = 1
#
#
#    # give number to each vocab item (new dict) and print vocabulary to the output file
#    vocab = {'UNK':{'UNK': 0}, '<s>':{'UNK': 1}, '</s>':{'UNK': 2}, '|':{'UNK': 2}}
#    index = 4
#    for char in vocab_cap.keys():
#        vocab[char]={}
#        vocab[char]['UNK']=index
#        index +=1
#        for m in vocab_cap[char].keys():
#            vocab[char][m] = index
#            index +=1
#        print char
#
#    print "Vocabulary size:"
#    print len(vocab)
#    
#    for char in vocab.keys():
#        for m in vocab[char].keys():
#            d.write('%s\t%s\t%s\n' %(char,m,vocab[char][m]))


parser = argparse.ArgumentParser(description='Convert between written and ID representation of words. '
                                 'The index 0 is always used as UNK token, wmap entry for 0 is ignored. '
                                 'Usage: python build_dict.py train.txt vocab.txt')
parser.add_argument('-c', '--char', help='build a dictionary of letters',
                    action='store_true')

parser.add_argument('--cap', help='add capitalized letters to the dictionary',
                    action='store_true')

parser.add_argument('--num', help='add numbers to the dictionary',
                    action='store_true')


parser.add_argument('-m', '--morf', help='build a dictionary of morphemes',
                    action='store_true')
parser.add_argument('--morf2char', help='build a dictionary of morphemes',
                    action='store_true')

#parser.add_argument('--hybrid', help='hybrid: build a dictionary of letters x morphemes',
#                    action='store_true')
parser.add_argument('-f', '--inputfile', help='inputfile: input file to NMT',
                    required=True)
parser.add_argument('--chardict', help='char dictionary file for m2c mapping',
                    required=False)
parser.add_argument('--morfdict', help='morf dictionary file for m2c mapping',
                    required=False)



parser.add_argument('-d', '--dictfile', help='dictfile: where to save dictionary/map',
                    required=True)
parser.add_argument('-o', '--outputfile', help='outputfile: mask morphemes with their IDs for LM training',
                    required=False)
args = parser.parse_args()



if __name__ == "__main__":
    #inputfile = args.f
    #dict_file = args.d
#	if args.hybrid:
#		process_hybrid(args.inputfile, args.dictfile)
	if args.char:
		process_char(args.inputfile, args.dictfile, args.cap, args.num)
	if args.morf:
		process_morf(args.inputfile, args.dictfile)
	if args.morf2char:
		process_m2c_map(args.inputfile, args.chardict, args.morfdict, args.dictfile, args.outputfile)

