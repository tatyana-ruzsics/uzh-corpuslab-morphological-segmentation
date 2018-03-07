# -*- coding: utf-8 -*-
"""This module contains predictors used for morphological segmentation (cED+LM) model:
    
    Word2charPredictorSegm supports decoding using predictors which operate on different tokenization level: chars and morphemes. It relies on the SGNMT ``Word2charPredictor``which is used if the decoder operates on fine-grained tokens such as characters, but the tokenization of a predictor is coarse-grained (e.g. words or subwords)
    
    SRILMPredictorSegm is a predictor for n-gram (Kneser-Ney) language model over morpheme. It relies on SGNMT
    ``SRILMPredictor`` class which is based on the swig-srilm package.
    
    WordCountPredictorSegm is a predictor for `length contro`, i.e. it adds a relative difference in morpheme length to the source word length (in chars) as a cost,
    which can be used to prevent hypotheses from getting to short when using a language model. It relies on SGNMT ``WordCountPredictor`` class.
    """

import logging
from cam.sgnmt.predictors.tokenization import Word2charPredictor
from cam.sgnmt.predictors.length import WordCountPredictor
from cam.sgnmt.predictors.ngram import SRILMPredictor

from cam.sgnmt import utils

import math

try:
    # Requires swig-srilm
    from srilm import getNgramProb
except ImportError:
    pass # Deal with it in decode.py


class Word2charPredictorSegm(Word2charPredictor):
    """This predictor wraps morpheme level predictors when SGNMT is running
        on the character level. The mapping between morpheme ID and character
        ID sequence is loaded from the file system. The MORPHEME boundary
        symbol is passed as a parameter. The wrapper blocks consume and predict_next
        calls until a MORPHEME boundary marker is consumed, and updates the slave predictor
        according the MORPHEME between the last two MORPHEME boundaries.
        """
    def __init__(self,  map_path, slave_predictor, sync_symb = -1):
        """Creates a new word2char wrapper predictor. The map_path
            file has to be plain text files, each line containing the
            mapping from a word index to the character index sequence
            (format: word char1 char2... charn).
            
            Args:
            map_path (string): Path to the mapping file
            slave_predictor (Predictor): Instance of the predictor with
            a different wmap than SGNMT
            sync_symb (int): MORPHEME boundary symbol
            """
        
        super(Word2charPredictorSegm, self).__init__(map_path, slave_predictor)
        self.sync_symb = sync_symb
    
    def _get_stub_prob_unbounded(self, ch):
        """get_stub_prob implementation for unbounded vocabulary slave
            predictors. (LM is an unbouded vocabulary predictor)
            """
        word = self.words.get(self.word_stub)
        
        if word:
            if ch in [utils.EOS_ID]: # end of word char
                posterior = self.slave_predictor.predict_next([word],1)
            else: # segmentation boundary  ch in [self.sync_symb]
                posterior = self.slave_predictor.predict_next([word])
            return utils.common_get(posterior, word, self.slave_unk)
        return self.slave_unk

    def predict_next(self, trgt_chars):
        posterior = {}
        stub_prob = False
        
        for ch in trgt_chars:
            if ch in [self.sync_symb, utils.EOS_ID]: # Segmentation boundary marker
                stub_prob = self._get_stub_prob(ch)
                posterior[ch] = stub_prob
            else:
                posterior[ch] = 0.0
        return posterior

    def consume(self, char):
        """If ``char`` is a segmentation boundary marker, truncate ``word_stub``
        and let the slave predictor consume word_stub. Otherwise,
        extend ``word_stub`` by the character.
        """
        if not char in [utils.EOS_ID, self.sync_symb]:
            self.word_stub.append(char)
        elif self.word_stub:
            morpheme = self.words.get(self.word_stub)
            self.slave_predictor.consume(morpheme if morpheme else utils.UNK_ID)
            self._start_new_word()

class SRILMPredictorSegm(SRILMPredictor):
    """SRILM predictor based on swig
        https://github.com/desilinguist/swig-srilm.
        
        The predictor state is described by the n-gram history."""

    def __init__(self, path, ngram_order, convert_to_ln=False):
        """Creates a new n-gram language model predictor.
            
            Args:
            path (string): Path to the ARPA language model file
            ngram_order (int): Order of the language model
            
            Raises:
            NameError. If srilm-swig is not installed
            """

        super(SRILMPredictorSegm, self).__init__(path, ngram_order, convert_to_ln=False)

    def predict_next(self, morphemes, eow=0):
        """Score the set of target MORPHEMES with the n-gram language model given the current history of MORPHEMES.
        
        Args:
        words (list): Set of morphemes to score
        Returns:
        dict. Language model scores for the words in ``words``
        """
        prefix = "%s " % ' '.join(self.history)
        order = len(self.history) + 1

        scaling_factor = math.log(10) if self.convert_to_ln else 1.0
            
        if eow==1:
            # Score for the end of word symbol:
            # logP(second-last-morf last-morf morf </s>) = logP(second-last-morf last-morf morf) + logP(last-morf morf </s>)
            prefix_eos = "%s " % ' '.join(self.history[1:])
            logging.debug(u"prefix {} w {}".format(prefix,str(morphemes)))
            logging.debug(u"prefix_eos {} w[0]: {}".format(prefix_eos,str((morphemes))))
            prob = {w: (getNgramProb(self.lm, prefix + str(w), order) + getNgramProb(self.lm, prefix_eos + str(w) + " </s>", order)) * scaling_factor for w in morphemes}
                    
        else:
            # Score for the segmentation boundary symbol:
            prob = {w: getNgramProb(self.lm, prefix + str(w), order) * scaling_factor for w in morphemes}

        return prob

class WordCountPredictorSegm(WordCountPredictor):
    """This predictor adds the relative difference between src word and its predicted segmentation (in chars). """

    def __init__(self, word = -1, sync_symb = -1):
        """Creates a new word count predictor instance.
            
            Args:
            word (int): If this is non-negative we count only the
            number of the specified chars. If its
            negative, count all chars
            sync_symb (int): MORPHEME boundary symbol
            """
        super(WordCountPredictorSegm, self).__init__(word = -1)
        self.unk_prob = 0.0
        self.sync_symb = sync_symb
        #self.history = []
    
    def initialize(self, src_sentence):
        self.src_len = len(src_sentence)
        self.history = []

    def predict_next(self):
        """Set score for EOS to the difference the relative difference between src word and its predicted segmentation (in chars) """
        logging.debug(u"History: {}".format(self.history))

        if len(self.history)!=0:
            penalty = -abs((len(self.history)-self.src_len)/self.src_len)
            logging.debug(u"penalty: {}".format(penalty))
            #penalty = -math.log(-penalty+1)
            
            self.posterior = {utils.EOS_ID : penalty}
        else:
            self.posterior = {utils.EOS_ID : -99, self.sync_symb : -99}

        return self.posterior

    def consume(self, char):
        if char not in [utils.EOS_ID, self.sync_symb]:
            self.history.append(char)

    def get_state(self):
        """Returns true """
        #return True
        return self.history
    
    def set_state(self, state):
        """Empty """
        self.history = state
        #pass
    
    def reset(self):
        """Empty method. """
        self.history = []
        #pass
