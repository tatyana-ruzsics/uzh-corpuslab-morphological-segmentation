from __future__ import division

#from memory_profiler import profile
from cam.sgnmt.decoding.core import Decoder
from cam.sgnmt.decoding.core import Hypothesis
from cam.sgnmt import utils
from cam.sgnmt.predictors.core import Predictor
from vocab_builder import build_vocabulary, Vocab

import codecs
import numpy as np
import dynet as dy
import logging

class DynetNMTPredictor(Predictor):
    """This is the neural machine translation predictor. The predicted
        posteriors are equal to the distribution generated by the decoder
        network in NMT.
        """
    
    def __init__(self, nmt_model_path):
        """Creates a new NMT predictor.
            
            Args:
            nmt_model_path (string):  Path to the NMT model file
            
            Raises:
            ValueError. If a target sparse feature map is defined
            """
        super(DynetNMTPredictor, self).__init__()
        self.set_up_predictor(nmt_model_path)
    
    def set_up_predictor(self, nmt_model_path):
        """Initializes the predictor with the given NMT model.
            """
        
        model_folder = nmt_model_path
        best_model_path  = model_folder + '/bestmodel.txt'
        hypoparams_file = model_folder + '/best.dev'
        
        hypoparams_file_reader = codecs.open(hypoparams_file, 'r', 'utf-8')
        hyperparams_dict = dict([line.strip().split(' = ') for line in hypoparams_file_reader.readlines()])
        self.hyperparams = {'INPUT_DIM': int(hyperparams_dict['INPUT_DIM']),
            'HIDDEN_DIM': int(hyperparams_dict['HIDDEN_DIM']),
                #'FEAT_INPUT_DIM': int(hyperparams_dict['FEAT_INPUT_DIM']),
                'LAYERS': int(hyperparams_dict['LAYERS']),
                    'VOCAB_PATH': hyperparams_dict['VOCAB_PATH']}
    
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
        self.fbuffRNN, self.bbuffRNN, self.VOCAB_LOOKUP, self.decoder, self.R, self.bias, self.W_c, self.W__a, self.U__a,  self.v__a = dy.load(best_model_path, self.pc)


    def initialize(self, input):
        """Runs the encoder network to create the source annotations
        for the source sentence. If the cache is enabled, empty the
        cache.
        
        Args:
        input (list): List of word ids without <S> and </S>
        which represent the source sentence.
        """
       
        R = dy.parameter(self.R)   # from parameters to expressions
        bias = dy.parameter(self.bias)
        W_c = dy.parameter(self.W_c)
        W__a = dy.parameter(self.W__a)
        U__a = dy.parameter(self.U__a)
        v__a = dy.parameter(self.v__a)
        
        self.cg_params = (R, bias, W_c, W__a, U__a, v__a) # params for current cg and input
        
        # biLSTM encoder of input string
        input = [self.BEGIN] + input + [self.STOP]
#        print input

        input_emb = []
        for char_id in reversed(input):
            char_embedding = self.VOCAB_LOOKUP[char_id]
            input_emb.append(char_embedding)
        self.biencoder = self.bilstm_transduce(self.fbuffRNN, self.bbuffRNN, input_emb)

        self.s = self.decoder.initial_state()
        self.s = self.s.add_input(self.VOCAB_LOOKUP[self.BEGIN])
        self.output_state = self.s.output()

#        initial_state = self.decoder.initial_state()#SGNMT
#        initial_state = initial_state.add_input(self.VOCAB_LOOKUP[self.BEGIN])#SGNMT
#        self.output_state = initial_state.output()#SGNMT


        logging.debug(u'NMT initialized with input: {}'.format(input))
            
        self.consumed = []#SGNMT
        self.logprobs = np.zeros(self.vocab.size())



    def bilstm_transduce(self, encoder_frnn, encoder_rrnn, input_char_vecs):
        
        # BiLSTM forward pass
        s_0 = encoder_frnn.initial_state()
        s = s_0
        frnn_outputs = []
        for c in input_char_vecs:
            s = s.add_input(c)
            frnn_outputs.append(s.output())
        
        # BiLSTM backward pass
        s_0 = encoder_rrnn.initial_state()
        s = s_0
        rrnn_outputs = []
        for c in reversed(input_char_vecs):
            s = s.add_input(c)
            rrnn_outputs.append(s.output())
        
        # BiLTSM outputs
        blstm_outputs = []
        for i in xrange(len(input_char_vecs)):
            blstm_outputs.append(dy.concatenate([frnn_outputs[i], rrnn_outputs[len(input_char_vecs) - i - 1]]))
        
        return blstm_outputs

#    @profile
    def predict_next(self):
        (R, bias, W_c, W__a, U__a, v__a) = self.cg_params
        
        # soft attention vector
        att_scores = [v__a * dy.tanh(W__a * self.output_state + U__a * h_input) for h_input in self.biencoder]
        alphas = dy.softmax(dy.concatenate(att_scores))
        c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(self.biencoder)])
        
        # softmax over vocabulary
        h_output = dy.tanh(W_c * dy.concatenate([self.output_state, c]))
        self.logprobs=(dy.log_softmax(R * h_output + bias)).npvalue()
        return self.logprobs

    def predict_next_(self, state):
        (R, bias, W_c, W__a, U__a, v__a) = self.cg_params
        
        # soft attention vector
        att_scores = [v__a * dy.tanh(W__a * state.output() + U__a * h_input) for h_input in self.biencoder]
        alphas = dy.softmax(dy.concatenate(att_scores))
        c = dy.esum([h_input * dy.pick(alphas, j) for j, h_input in enumerate(self.biencoder)])
        
        # softmax over vocabulary
        h_output = dy.tanh(W_c * dy.concatenate([state.output(), c]))
        return (dy.log_softmax(R * h_output + bias)).npvalue()
    
    def get_state(self):
        """The NMT predictor state consists of the decoder network
            state, and (for caching) the current history of consumed words
            """
        return self.consumed #SGNMT
#        return self.s#NEW

    def set_state(self, state):
        """Set the NMT predictor state. """
        self.consumed = state#SGNMT
#        self.s = state#NEW
        return
    
    def reset(self):
        """Empty method. """
        self.consumed = []#SGNMT
        self.s = None#NEW
        return

    def get_unk_probability(self, posterior):
        """Returns the UNK probability defined by NMT. """
        return posterior[utils.UNK_ID] if len(posterior) > utils.UNK_ID else NEG_INF

    def consume(self, pred_id):
        """Feeds back ``pred_id`` to the decoder network. This includes
        embedding of ``pred_id``, running the attention network and update
        the recurrent decoder layer.
        """
        logging.debug(u'nmt consumed: {}'.format(utils.apply_trg_wmap([pred_id])))#SGNMT
        
        self.consumed.append(pred_id)#SGNMT


        inputs_id = [self.BEGIN] + self.consumed#SGNMT
        initial_state = self.decoder.initial_state()#SGNMT
        inputs_emb = [self.VOCAB_LOOKUP[c_id] for c_id in inputs_id]#SGNMT
        states = initial_state.transduce(inputs_emb)#SGNMT
        self.output_state = states[-1]#SGNMT


#        self.consume_next(pred_id)#NEW
        pass

    def consume_next(self, pred_id):
        self.s = self.s.add_input(self.VOCAB_LOOKUP[pred_id])
        self.output_state = self.s.output()

    def consume_next_(self, state, pred_id):
        new_state = state.add_input(self.VOCAB_LOOKUP[pred_id])
        return new_state

class DynetNMTVanillaDecoder(Decoder):
    """It can only be used for pure single system NMT decoding.
        """
    def __init__(self, nmt_model_path, decoder_args):
        """Set up the NMT model used by the decoder.
            
        Args:
            nmt_model_path (string):  Path to the NMT model file
                                      config (dict): NMT configuration
            decoder_args (object): Decoder configuration passed through
                                   from configuration API.
        """
        super(DynetNMTVanillaDecoder, self).__init__(decoder_args)
        self.beam_size = decoder_args.beam
        self.set_up_decoder(nmt_model_path)

    def set_up_decoder(self, nmt_model_path):
        """This method uses the NMT configuration in ``self.config`` to
        initialize the NMT model. This method basically corresponds to
        ``blocks.machine_translation.main``.
        
        Args:
        nmt_model_path (string):  Path to the NMT model file (.npz)
        """
        self.nmt_model = DynetNMTPredictor(nmt_model_path)
    
#    @profile
    def decode(self, src_sentence):
        """Decodes a single source sentence. Note that the
        score breakdowns in returned hypotheses are only on the
        sentence level, not on the word level. For finer grained NMT
        scores you need to use the nmt predictor. ``src_sentence`` is a
        list of source word ids representing the source sentence without
        <S> or </S> symbols. As blocks expects to see </S>, this method
        adds it automatically.
        
        Args:
        src_sentence (list): List of source word ids without <S> or
        </S> which make up the source sentence
        
        Returns:
        list. A list of ``Hypothesis`` instances ordered by their
        score.
        """
        dy.renew_cg()
        logging.debug(u'src_sentence: {}'.format(src_sentence))
#        MAX_PRED_SEQ_LEN = 30*len(src_sentence)
        MAX_PRED_SEQ_LEN = 30
        logging.debug(u'MAX_PRED_SEQ_LEN: {}'.format(MAX_PRED_SEQ_LEN))
        BEGIN = utils.GO_ID
        STOP = utils.EOS_ID
        logging.debug(u'BEGIN: {}, STOP: {}'.format(BEGIN,STOP))
        beam_size = self.beam_size
        self.nmt_model.initialize(src_sentence)
#        ignore_first_eol=True
        states = [self.nmt_model.s] * beam_size
        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = np.full(shape=(1,beam_size),fill_value=BEGIN,dtype = int)
        all_masks = np.ones_like(all_outputs, dtype=float) # whether predicted symbol is self.STOP
        all_costs = np.zeros_like(all_outputs, dtype=float) # the cumulative cost of predictions
        
        for i in range(MAX_PRED_SEQ_LEN):
            if all_masks[-1].sum() == 0:
                logging.debug(u'all_masks: {}'.format(all_masks))
                break
        
            # We carefully hack values of the `logprobs` array to ensure
            # that all finished sequences are continued with `eos_symbol`.
            logprobs = - np.array([self.nmt_model.predict_next_(s) for s in states])
            #            print logprobs
            #            print all_masks[-1, :, None]
            next_costs = (all_costs[-1, :, None] + logprobs * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
            (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
            next_costs[finished, :STOP] = np.inf
            next_costs[finished, STOP + 1:] = np.inf
            
            # indexes - the hypos from prev step to keep, outputs - the next step prediction, chosen cost - cost of predicted symbol
            (indexes, outputs), chosen_costs = self._smallest(next_costs, beam_size, only_first_row=i == 0)
            #            print outputs
            # Rearrange everything
            new_states = (states[ind] for ind in indexes)
            all_outputs = all_outputs[:, indexes]
            all_masks = all_masks[:, indexes]
            all_costs = all_costs[:, indexes]
            
            # Record chosen output and compute new states
            states = [self.nmt_model.consume_next_(s,pred_id) for s,pred_id in zip(new_states, outputs)]
            all_outputs = np.vstack([all_outputs, outputs[None, :]])
            logging.debug(u'all_outputs: {}'.format(all_outputs))
            logging.debug(u'outputs: {}'.format([utils.apply_trg_wmap([c]) for c in outputs]))
            logging.debug(u'chosen_costs: {}'.format(chosen_costs))
            logging.debug(u'outputs != STOP: {}'.format(outputs != STOP))
            all_costs = np.vstack([all_costs, chosen_costs[None, :]])
            mask = outputs != STOP
#            if ignore_first_eol: #and i == 0:
#                mask[:] = 1
            all_masks = np.vstack([all_masks, mask[None, :]])

        all_outputs = all_outputs[1:] # skipping first row of self.BEGIN
        logging.debug(u'outputs: {}'.format(all_outputs))
        all_masks = all_masks[:-1] #? all_masks[:-1] # skipping first row of self.BEGIN and the last row of self.STOP
        logging.debug(u'masks: {}'.format(all_masks))
        all_costs = all_costs[1:] - all_costs[:-1] #turn cumulative cost ito cost of each step #?actually the last row would suffice for us?
        result = all_outputs, all_masks, all_costs
        
        trans, costs = self.result_to_lists(self.nmt_model.vocab, result)
        logging.debug(u'trans: {}'.format(trans))
        hypos = []
        max_len = 0
        for idx in xrange(len(trans)):
            max_len = max(max_len, len(trans[idx]))
            hypo = Hypothesis(trans[idx], -costs[idx])
            hypo.score_breakdown = len(trans[idx]) * [[(0.0,1.0)]]
            hypo.score_breakdown[0] = [(-costs[idx],1.0)]
            hypos.append(hypo)
        
        logging.debug(u'hypos: {}'.format(all_outputs))
        return hypos
            
    @staticmethod
    def _smallest(matrix, k, only_first_row=False):
        """Find k smallest elements of a matrix.
            Parameters
            ----------
            matrix : :class:`np.ndarray`
            The matrix.
            k : int
            The number of smallest elements required.
            Returns
            -------
            Tuple of ((row numbers, column numbers), values).
            """
        #flatten = matrix.flatten()
        if only_first_row:
            flatten = matrix[:1, :].flatten()
        else:
            flatten = matrix.flatten()
        args = np.argpartition(flatten, k)[:k]
        args = args[np.argsort(flatten[args])]
        #        print args
        #        print np.unravel_index(args, matrix.shape)
        #        print flatten[args]
        return np.unravel_index(args, matrix.shape), flatten[args]

    @staticmethod
    def result_to_lists(nmt_vocab, result):
        outputs, masks, costs = [array.T for array in result]
        outputs = [list(output[:int(mask.sum())]) for output, mask in zip(outputs, masks)]
#        words = [u''.join([nmt_vocab.i2w.get(pred_id,UNK_CHAR) for pred_id in output]) for output in outputs]
        words = [[pred_id for pred_id in output] for output in outputs]
        costs = list(costs.T.sum(axis=0))
        return words, costs

    def has_predictors(self):
        """Always returns true. """
        return True

class DynetNMTEnsembleDecoder(Decoder):
    """It can only be used for pure single system NMT decoding.
        """
    def __init__(self, nmt_model_paths, decoder_args):
        """Set up the NMT model used by the decoder.
            
            Args:
            nmt_model_path (list):  List of path to the NMT model file
            decoder_args (object): Decoder configuration passed through
            from configuration API.
            """
        super(DynetNMTEnsembleDecoder, self).__init__(decoder_args)
        self.beam_size = decoder_args.beam
        self.set_up_decoder(nmt_model_paths)

    def set_up_decoder(self, nmt_model_paths):
        """This method uses the NMT configuration in ``self.config`` to
        initialize the NMT model. This method basically corresponds to
        ``blocks.machine_translation.main``.
        
        Args:
        nmt_model_paths (list):  List of paths to the NMT model file
        """
        self.nmt_models = []
        for path in nmt_model_paths:
            self.nmt_models.append(DynetNMTPredictor(path))

    def has_predictors(self):
        """Always returns true. """
        return True

    def decode(self, src_sentence):
        """This is a generalization to NMT ensembles of ``DynetNMTVanillaDecoder``.
                    
        Args:
        src_sentence (list): List of source word ids without <S> or
        </S> which make up the source sentence
        
        Returns:
        list. A list of ``Hypothesis`` instances ordered by their
        score.
        """
        dy.renew_cg()
        logging.debug(u'src_sentence: {}'.format(src_sentence))
        MAX_PRED_SEQ_LEN = 3*len(src_sentence)
        beam_size = self.beam_size
        nmt_models = self.nmt_models
        
        nmt_vocab = nmt_models[0].vocab # same vocab file for all nmt_models!!
#        BEGIN   = nmt_vocab.w2i[BEGIN_CHAR]
        BEGIN = utils.GO_ID
        STOP = utils.EOS_ID
#        STOP   = nmt_vocab.w2i[STOP_CHAR]

        for m in nmt_models:
            m.initialize(src_sentence)
        states = [[m.s] * beam_size for m in nmt_models] # ensemble x beam matrix of states
        # This array will store all generated outputs, including those from
        # previous step and those from already finished sequences.
        all_outputs = np.full(shape=(1,beam_size),fill_value=BEGIN,dtype = int)
        all_masks = np.ones_like(all_outputs, dtype=float) # whether predicted symbol is self.STOP
        all_costs = np.zeros_like(all_outputs, dtype=float) # the cumulative cost of predictions

        for i in range(MAX_PRED_SEQ_LEN):
            if all_masks[-1].sum() == 0:
                break
        
            # We carefully hack values of the `logprobs` array to ensure
            # that all finished sequences are continued with `eos_symbol`.
            logprobs_lst = []
            for j,m in enumerate(nmt_models):
                logprobs_m = - np.array([m.predict_next_(s) for s in states[j]]) # beam_size x vocab_len
                logprobs_lst.append(logprobs_m)
            logprobs = np.sum(logprobs_lst, axis=0)
            next_costs = (all_costs[-1, :, None] + logprobs * all_masks[-1, :, None]) #take last row of cumul prev costs and turn into beam_size X 1 matrix, take logprobs distributions for unfinished hypos only and add it (elem-wise) with the array of prev costs; result: beam_size x vocab_len matrix of next costs
            (finished,) = np.where(all_masks[-1] == 0) # finished hypos have all their cost on the self.STOP symbol
            next_costs[finished, :STOP] = np.inf
            next_costs[finished, STOP + 1:] = np.inf

            # indexes - the hypos from prev step to keep, outputs - the next step prediction, chosen cost - cost of predicted symbol
            (indexes, outputs), chosen_costs = DynetNMTVanillaDecoder._smallest(next_costs, beam_size, only_first_row=i == 0)

            # Rearrange everything
            new_states=[]
            for j,m in enumerate(nmt_models):
                new_states.append([states[j][ind] for ind in indexes])
        
            #        new_states = ((states_m[ind] for ind in indexes) for states_m in states)
            all_outputs = all_outputs[:, indexes]
            all_masks = all_masks[:, indexes]
            all_costs = all_costs[:, indexes]

            # Record chosen output and compute new states
            states = [[m.consume_next_(s,pred_id) for s,pred_id in zip(m_new_states, outputs)] for m,m_new_states in zip(nmt_models, new_states)]
            all_outputs = np.vstack([all_outputs, outputs[None, :]])
            logging.debug(u'all_outputs: {}'.format(all_outputs))
            logging.debug(u'outputs: {}'.format([utils.apply_trg_wmap([c]) for c in outputs]))
            logging.debug(u'chosen_costs: {}'.format(chosen_costs))
            logging.debug(u'outputs != STOP: {}'.format(outputs != STOP))
            all_costs = np.vstack([all_costs, chosen_costs[None, :]])
            mask = outputs != STOP
            #        if ignore_first_eol: # and i == 0:
            #            mask[:] = 1
            all_masks = np.vstack([all_masks, mask[None, :]])

        all_outputs = all_outputs[1:] # skipping first row of self.BEGIN
        logging.debug(u'outputs: {}'.format(all_outputs))
        all_masks = all_masks[:-1] #? all_masks[:-1] # skipping first row of self.BEGIN and the last row of self.STOP
        logging.debug(u'masks: {}'.format(all_masks))
        all_costs = all_costs[1:] - all_costs[:-1] #turn cumulative cost ito cost of each step #?actually the last row would suffice for us?
        result = all_outputs, all_masks, all_costs

        trans, costs = DynetNMTVanillaDecoder.result_to_lists(nmt_vocab,result)
        logging.debug(u'trans: {}'.format(trans))
        hypos = []
        max_len = 0
        for idx in xrange(len(trans)):
            max_len = max(max_len, len(trans[idx]))
            hypo = Hypothesis(trans[idx], -costs[idx])
            hypo.score_breakdown = len(trans[idx]) * [[(0.0,1.0)]]
            hypo.score_breakdown[0] = [(-costs[idx],1.0)]
            hypos.append(hypo)
            self.apply_predictors_count = max_len * self.beam_size
        logging.debug(u'hypos: {}'.format(all_outputs))
        return hypos



