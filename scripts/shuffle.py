#!/usr/bin/env python

import argparse
import logging
import os
import random
import codecs


parser = argparse.ArgumentParser(
    description="""
This takes a.txt files and does shuffling of lines.
""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("input",
                    help="The input source file", nargs='+')
parser.add_argument("-s", "--seed", type=int,
                    help="random number seed")


def shuffle_lines(filename_src, filename_trg):
    with codecs.open(filename_src, mode="r", encoding="utf-8") as myFile:
        lines_src = list(myFile)
    with codecs.open(filename_trg, mode="r", encoding="utf-8") as myFile:
        lines_trg = list(myFile)
    lines = list(zip(lines_src,lines_trg))
    random.seed(a=args.seed)
    random.shuffle(lines)
    filename_shufffled_src = filename_src+"-shuf"
    filename_shufffled_trg = filename_trg+"-shuf"
    with codecs.open(filename_shufffled_src, mode="w", encoding="utf-8") as myFile:
        for line in lines:
            myFile.write(line[0])
    with codecs.open(filename_shufffled_trg, mode="w", encoding="utf-8") as myFile:
        for line in lines:
            myFile.write(line[1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('shuffle')
    args = parser.parse_args()
    filename_src = args.input[0]
    filename_trg = args.input[1]
    filename_shufffled_src = args.input[2]
    filename_shufffled_trg = args.input[3]
    shuffle_lines(filename_src, filename_trg)
