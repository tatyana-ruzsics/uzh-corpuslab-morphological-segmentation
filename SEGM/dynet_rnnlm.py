
from __future__ import division
from cam.sgnmt.predictors.core import UnboundedVocabularyPredictor
from cam.sgnmt import utils

import dynet as dy
import numpy as np
import codecs

class DynetRNNLMPredictor(UnboundedVocabularyPredictor):
    """Creates a new RNN language model predictor.
    
    Args:
    path (string): Path to the language model folder
    """
    def __init__(self, model_path):
        model_folder = nmt_model_path
        best_model_path  = model_folder + '/bestmodel.txt'
        hypoparams_file = model_folder + '/best.dev'
        
        hypoparams_file_reader = codecs.open(hypoparams_file, 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        self.hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                #'FEAT_INPUT_DIM': int(hyperparams_dict['FEAT_INPUT_DIM']),
                'LAYERS': int(hyperparams_dict['LAYERS']),
                    'VOCAB_PATH': hyperparams_dict['VOCAB_PATH'],
                    'OVER_SEGS':  'OVER_SEGS' in hyperparams_dict}
    
        self.pc = dy.ParameterCollection()
        
        print 'Loading vocabulary from {}:'.format(self.hyperparams['VOCAB_PATH'])
        self.vocab = Vocab.from_file(self.hyperparams['VOCAB_PATH'])
        #        BEGIN_CHAR   = u'<s>'
        #        STOP_CHAR   = u'</s>'
        #        UNK_CHAR = u'<unk>'
        #        self.BEGIN   = self.vocab.w2i[BEGIN_CHAR]
        #        self.STOP   = self.vocab.w2i[STOP_CHAR]
        #        self.UNK       = self.vocab.w2i[UNK_CHAR]
        self.BEGIN = utils.GO_ID
        self.STOP = utils.EOS_ID
        self.UNK = utils.UNK_ID
        self.hyperparams['VOCAB_SIZE'] = self.vocab.size()
        
        print 'Model Hypoparameters:'
        for k, v in self.hyperparams.items():
            print '{:20} = {}'.format(k, v)
        print
        
        print 'Loading model from: {}'.format(best_model_path)
        self.RNN, self.VOCAB_LOOKUP, self.R, self.bias  = dy.load(best_model_path, self.pc)

        print 'Model dimensions:'
        print ' * VOCABULARY EMBEDDING LAYER: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['VOCAB_SIZE'], self.hyperparams['INPUT_DIM'])
        print
        print ' * LSTM: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['INPUT_DIM'], self.hyperparams['HIDDEN_DIM'])
        print ' LSTM has {} layer(s)'.format(self.hyperparams['LAYERS'])
        print
        print ' * SOFTMAX: IN-DIM: {}, OUT-DIM: {}'.format(self.hyperparams['HIDDEN_DIM'], self.hyperparams['VOCAB_SIZE'])
        print

    def initialize(self, src_sentence):
        """Initializes the history with the start-of-sentence symbol.
        
        Args:
        src_sentence (list): Not used
        """
        R = dy.parameter(self.R)   # from parameters to expressions
        bias = dy.parameter(self.bias)
        self.cg_params = (R, bias)
        
        self.s = self.RNN.initial_state()
        self.s = self.s.add_input(self.VOCAB_LOOKUP[self.BEGIN])

    def predict_next(self, segm_id, eow=0):
        """Score the set of target words with the n-gram language
        model given the current history
        
        Args:
        words (list): Set of segments to score
        Returns:
        dict. Language model scores for the words in ``words``
        """
        (R, bias) = self.cg_params
        if eow==1:
            # Score for the end of word symbol:
            next_scores = dy.log_softmax(bias + (R * self.s.output()))
            s_temp = self.s
            s = s_temp.transduce([self.VOCAB_LOOKUP[segm_id]])
            scores_eof = dy.log_softmax(bias + (R * s[-1]))
            logprob = {s: next_scores[s].value() + scores_eof[self.STOP].value() for s in segm_id}
        else:
            next_scores = dy.log_softmax(bias + (R * self.s.output()))
            logprob = {s: next_scores[s].value() for s in segm_id}
        return logprob

    def get_unk_probability(self, posterior):
        """Use the probability for 'UNK' in the language model """
        (R, bias) = self.cg_params
        next_scores = dy.log_softmax(bias + (R * self.s.output()))
        return next_scores[self.UNK]
    
    def consume(self, next_id):
        """Extends the current history by ``next_id`` """
        self.s = self.s.add_input(self.VOCAB_LOOKUP[next_id])

