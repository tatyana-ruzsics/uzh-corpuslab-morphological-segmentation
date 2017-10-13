"""This script applies a word map to sentences in stdin. If --dir is
set to s2i, the word strings in stdin are converted to their ids. If
--dir is i2s, we convert word IDs to their readable representations.
"""

import logging
import argparse
import sys
import codecs

def load_wmap(path, inverse=False):
    with codecs.open(path, 'r', 'utf8') as f:
        d = dict(line.rstrip().split('\t') for line in f)
        #if inverse:
        #d = dict(zip(d.values(), d.keys()))
        for (s, i) in [('<s>', '1'), ('</s>', '2')]:
            if not s in d or d[s] != i:
                logging.warning("%s has not ID %s in word map %s!" % (s, i, path))
        #print(d)
        return d

def load_wmap_hybrid(path, inverse=False):
    d = {}
    with codecs.open(path, 'r', 'utf8') as f:
        if inverse:
            for line in f:
                items = line.strip().split(None,2)
                d[items[2]] = items[0]
        else:
            for line in f:
                items = line.strip().split(None,2)
                if items[0]  not in d.keys():
                    d[items[0]]={}
                d[items[0]][items[1]] = items[2]
        for (s, i) in [('<s>', '1'), ('</s>', '2')]:
            if not s in d or d[s]['UNK'] != i:
                logging.warning("%s has not ID %s in word map %s!" % (s, i, path))
        return d

parser = argparse.ArgumentParser(description='Convert between written and ID representation of words. '
                                 'The index 0 is always used as UNK token, wmap entry for 0 is ignored. '
                                 'Usage: python apply_wmap.py < in_sens > out_sens')
parser.add_argument('-d','--dir', help='s2i: convert to IDs (default), i2s: convert from IDs',
                    required=False)
parser.add_argument('-m','--wmap', help='Word map to apply (format: see -i parameter)',
                    required=True)
#parser.add_argument('-f','--fields', help='Comma separated list of fields (like for linux command cut)')
parser.add_argument('-i','--inverse_wmap', help='Use this argument to use word maps with format "id word".'
                    ' Otherwise the format "word id" is assumed', action='store_true')
parser.add_argument('--hybrid', help='hybrid: build a dictionary of letters x morphemes',
                    action='store_true')
args = parser.parse_args()


def process():

    wmap = load_wmap(args.wmap, args.inverse_wmap)
    #print("Hello")
    #print(wmap)
    unk = '0'
    if args.dir and args.dir == 'i2s': # inverse wmap
        wmap = dict(zip(wmap.values(), wmap.keys()))
        unk = "NOTINWMAP"

    #fields = None
    #if args.fields:
    #    fields = [int(f)-1 for f in args.fields.split(',')]

    # do not use for line in sys.stdin because incompatible with -u option
    # required for lattice mert
    while True:
        line = sys.stdin.readline().decode('utf-8')
        if not line: break # EOF
    #    if fields:
    #        words = line.strip().split()
    #        for f in fields:
    #            if f < len(words):
    #                words[f] = wmap.get(words[f], unk)
    #        print("\t".join(words))
    #    else:
        print(' '.join([wmap[w] if (w in wmap) else unk for w in line.strip().split()]))

def process_hybrid():
    
    if args.dir and arg.dir == 'i2s':
        wmap = load_wmap_hybrid(args.wmap, True)
        unk = "NOTINWMAP"
        while True:
            line = sys.stdin.readline()
            if not line: break # EOF
            print(' '.join([wmap[w] if (w in wmap) else unk for w in line.strip().split()]))

    else:
        wmap = load_wmap_hybrid(args.wmap)
        unk = '0'
        while True:
            line = sys.stdin.readline().decode('utf-8')
            if not line: break # EOF
            mapping = []
            morpfs = line.strip().split(" | ") # Split word into morphemes
            for m in morpfs:
                chars = m.split(" ")
                morf = "".join(chars)
                for char in chars:
                    try:
                        if char in wmap.keys() and morf in wmap[char].keys():
                            mapping.append(wmap[char][morf])
                        else:
                            mapping.append(unk)
                    except:
                        print line

            print(' '.join(mapping))

if __name__ == "__main__":
    if args.hybrid:
        process_hybrid()
    else:
        process()

