# -*- coding: utf-8 -*-
"""Implementation of the beam search and "synchronized" beam search strategies used for morphological segmentation model (cED+LM)."""

#from memory_profiler import profile
import copy
import logging
from cam.sgnmt import utils
from cam.sgnmt.decoding.beam import BeamDecoder
import numpy as np
from cam.sgnmt.decoding.core import Hypothesis,PartialHypothesis
import dynet as dy
#from guppy import hpy

class BeamDecoderSegm(BeamDecoder):

    def __init__(self, *args, **kwargs):
    
        super(BeamDecoderSegm, self).__init__(*args)
#        self.max_len = 30

#    @profile
    def _expand_hypo_nmt(self, hypo):
        """Get the best beam size expansions of ``hypo`` by one CHAR based on nmt predictor scores only.
        
        Args:
        hypo (PartialHypothesis): Hypothesis to expans
        
        Returns:
        list. List of child hypotheses
        """
        if hypo.score <= self.min_score:
            return []
        self.set_predictor_states(copy.deepcopy(hypo.predictor_states))
#        self.set_predictor_states(hypo.predictor_states)
        if not hypo.word_to_consume is None: # Consume if cheap expand
            self.consume(hypo.word_to_consume)
            hypo.word_to_consume = None
        posterior,score_breakdown = self.apply_predictors()
#        self.apply_predictors()
        hypo.predictor_states = self.get_predictor_states()
#        nmt_only_scores = {k: sum([v[i][0] for i,s in enumerate(v) if self.predictor_names[i]=="nmt"]) for k, v in score_breakdown.items()}
#        nmt_only_scores = np.array([v for k,v in sorted(nmt_only_scores.items(),key=lambda v:v[0])])
#        
        nmt_only_scores = np.array([sum([v[i][0] for i,s in enumerate(v) if self.predictor_names[i]=="nmt"]) for k,v in sorted(score_breakdown.items(),key=lambda t:t[0])])

#        nmt_only_scores = np.array([sum([v[i][0] for i,s in enumerate(v) if self.predictor_names[i]=="nmt"]) for k,v in sorted(self.score_breakdown.items(),key=lambda t:t[0])])

#        logging.debug(u'score_breakdown.items(): {}'.format(score_breakdown.items()))
#        logging.debug(u'next nmt scores: {}'.format(nmt_only_scores))
        top = utils.argmax_n(nmt_only_scores, self.beam_size)
#        char_only_scores = {k: sum([v[i][0] for i,s in enumerate(v) if self.predictor_levels[i]=="c"]) for k, v in score_breakdown.items()}
#        top = utils.argmax_n(char_only_scores, self.beam_size)
        return [hypo.cheap_expand(
                                  trgt_word,
                                  posterior[trgt_word],
                                  score_breakdown[trgt_word]) for trgt_word in top]
#        return [hypo.cheap_expand(
#                                  trgt_word,
#                                  self.posterior[trgt_word],
#                                  self.score_breakdown[trgt_word]) for trgt_word in top]

    def setup_max_len(self, src_sentence):
        if len(src_sentence) > 50: #hack agains hyperlinks in the data
            self.max_len = 0 #

#    @profile
    def decode(self, src_sentence):
        """Decodes a single source sentence using beam search.
        Expands (beam size) hypotheses based on a sum of nmt predictors scores (_expand_hypo_nmt), cuts (beam size) the resulting continuation based on a combined predictors score."""
        dy.renew_cg()
        self.initialize_predictors(src_sentence)
        hypos = self._get_initial_hypos()
        self.setup_max_len(src_sentence)
        logging.debug(u"Source len {}".format(len(src_sentence)))
        logging.debug(u"MAX-ITER: {}".format(self.max_len))
        # Initial expansion
        for hypo in hypos:
            logging.debug(u"INIT {} {}".format(utils.apply_trg_wmap(hypo.trgt_sentence), hypo.score_breakdown))
        it = 0
        while self.stop_criterion(hypos):
            logging.debug(u"ITER: {}, MAX-ITER: {}".format(it,self.max_len))
            if it > self.max_len: # prevent infinite loops
                break
            it = it + 1
            
            next_hypos = []
            next_scores = []
            self.min_score = utils.NEG_INF
            self.best_scores = []
            for hypo in hypos:
                if hypo.get_last_word() == utils.EOS_ID:
                    next_hypos.append(hypo)
                    next_scores.append(self._get_combined_score(hypo))
                    logging.debug(u"BEAM IT {} HYPO {} NO EXPAND".format(it, utils.apply_trg_wmap(hypo.trgt_sentence)))
                    continue
                for next_hypo in self._expand_hypo_nmt(hypo):
                    next_score = self._get_combined_score(next_hypo)
                    if next_score > self.min_score:
                        next_hypos.append(next_hypo)
                        next_scores.append(next_score)
                        self._register_score(next_score)
                    logging.debug(u"BEAM IT {} HYPO {} -> NEXT HYPO {}".format(it, utils.apply_trg_wmap(hypo.trgt_sentence), utils.apply_trg_wmap(next_hypo.trgt_sentence)))
        
            # hypo expansions on this iteraion which will be cut (beam size) based on combined predictors score:
            logging.debug(u"BEAM IT {} NEXT HYPOS BEFORE CUT -> {}".format(it, " && ".join(utils.apply_trg_wmap(h.trgt_sentence) + ", " + str(next_scores[i]) for i,h in enumerate(next_hypos))))
            logging.debug(u"BEAM IT {} Min score: {}".format(it, self.min_score))
                
            if self.hypo_recombination:
                hypos = self._filter_equal_hypos(next_hypos, next_scores)
            else:
                hypos = self._get_next_hypos(next_hypos, next_scores)
            
            # Best (beam size) expansions of the hypo on this iteration...
            logging.debug(u"BEAM IT {} CUT: {}".format(it, " && ".join(utils.apply_trg_wmap(h.trgt_sentence) for h in hypos)))
                
            # ... with detailed scores per char
            for i,hypo in enumerate(hypos):
                logging.debug(u"BEAM IT {} :{}".format(utils.apply_trg_wmap(hypo.trgt_sentence), hypo.score))
                for i,score_char in enumerate(hypo.score_breakdown):
                    logging.debug(u"{}: {}".format(utils.apply_trg_wmap([hypo.trgt_sentence[i]]),", ".join("{:.10f}".format(s) + ":"+ "{:.2f}".format(w) for s,w in score_char)))

#        # final hypos
#        final_scores = []
#        final_hypos = []
#        for hypo in hypos:
#            final_hypos.append(hypo)
#            final_scores.append(hypo.score)
#        hypos = self._get_next_hypos(final_hypos, final_scores)
#
#        # Best final hypos
#        logging.debug(u"BEAM FINAL: {}".format(" && ".join(utils.apply_trg_wmap(h.trgt_sentence) for h in hypos)))

        for hypo in hypos:
            if hypo.get_last_word() == utils.EOS_ID:
                self.add_full_hypo(hypo.generate_full_hypothesis())
        if not self.full_hypos:
            logging.warn("No complete hypotheses found for %s" % src_sentence)
            for hypo in hypos:
                self.add_full_hypo(hypo.generate_full_hypothesis())

        return self.get_full_hypos_sorted()



class SyncBeamDecoderSegm(BeamDecoderSegm):
    """This beam search implementation follows SyncBeamDecoder approach of SGNMT framework:
        ...It is s a two level approach. Hypotheses are not compared after each iteration, but after
        consuming an explicit synchronization symbol. This is useful
        when SGNMT runs on the character level, but it makes more sense
        to compare hypos with same lengths in terms of number of words
        and not characters. The end-of-word symbol </w> can be used as
        synchronization symbol...
        
        Specifically to morphological segmentation model (cED+LM), the 'decode' method of this class works as follows:
        expands (beam size) hypotheses by MORPHEMES based on a sum of nmt predictors scores ('_expand_hypo_nmt' method of SyncBeamDecoderSegm class), cuts (beam size) the resulting continuation based on a combined predictors score.
        """
    
    def __init__(self,
                 decoder_args,
                 hypo_recombination,
                 beam_size,
                 pure_heuristic_scores = False,
                 diversity_factor = -1.0,
                 early_stopping = True,
                 sync_symb = -1,
                 max_word_len = 25):
        """Creates a new beam decoder instance with explicit
            synchronization symbol.
            
            Args:
            decoder_args (object): Decoder configuration passed through
            from the configuration API.
            hypo_recombination (bool): Activates hypo recombination
            beam_size (int): Absolute beam size. A beam of 12 means
            that we keep track of 12 active hypothesis
            pure_heuristic_scores (bool): Hypotheses to keep in the beam
            are normally selected
            according the sum of partial
            hypo score and future cost
            estimates. If set to true,
            partial hypo scores are
            ignored.
            diversity_factor (float): If this is set to a positive
            value we add diversity promoting
            penalization terms to the partial
            hypothesis scores following Li
            and Jurafsky, 2016
            early_stopping (bool): If true, we stop when the best
            scoring hypothesis ends with </S>.
            If false, we stop when all hypotheses
            end with </S>. Enable if you are
            only interested in the single best
            decoding result. If you want to
            create full 12-best lists, disable
            sync_symb (int): Synchronization symbol. If negative, fetch
            '</w>' from ``utils.trg_cmap``
            max_word_len (int): Maximum length of a single word
            """
        super(SyncBeamDecoderSegm, self).__init__(decoder_args,
                                              hypo_recombination,
                                              beam_size,
                                              pure_heuristic_scores,
                                              diversity_factor,
                                              early_stopping)
        self.sync_symb = sync_symb
        self.max_morf_len = max_word_len #default: 25

    def _is_closed(self, hypo):
        """Returns true if hypo ends with </S> or </W>"""
        return hypo.get_last_word() in [utils.EOS_ID, self.sync_symb]
    
    def _all_eos_or_eow(self, hypos):
        """Returns true if the all hypotheses end with </S> or </W>"""
        for hypo in hypos:
            if not self._is_closed(hypo):
                return True
        return False
#    @profile
    def _expand_hypo_nmt(self, input_hypo):
        """Get the best beam size expansions of ``hypo`` by one MORPHEME based on nmt predictor scores only, i.e. expand hypo until all of the beam size best hypotheses end with ``sync_symb`` or EOS. The implementation relies on '_expand_hypo_nmt' of the parent class BeamDecoderSegm which provides best beam size expansions of ``hypo`` by one CHAR based on nmt predictor scores only.
        
        Args:
        hypo (PartialHypothesis): Hypothesis to expand
        
        Return:
        list. List of expanded hypotheses.
        """
        # The input hypo to be expanded
        logging.debug(u"EXPAND: {} {}".format(utils.apply_trg_wmap(input_hypo.trgt_sentence), input_hypo.score))
        
        # Get initial expansions by one char
        hypos = super(SyncBeamDecoderSegm, self)._expand_hypo_nmt(input_hypo)
        # input_hypo_len = len(input_hypo.score_breakdown)
        # Expand until all hypos are closed
        it = 0
        while self._all_eos_or_eow(hypos):
            if it > self.max_morf_len: # prevent infinite loops
                break
            logging.debug(u"SYNC BEAM ITER: {}".format(it))
            it = it + 1
            next_hypos = []
            next_scores = []
            for hypo in hypos:
                # Combined predictors score for the chars in a next morpheme (we look for a best morpheme expansion of the input_hypo)
                next_score = sum([sum([char_scores[i][0] for i,s in enumerate(char_scores) if self.predictor_names[i]=="nmt"]) for char_scores in hypo.score_breakdown])
#                next_score = sum([sum([char_scores[i][0] for i,s in enumerate(char_scores) if self.predictor_levels[i]=="c"]) for char_scores in hypo.score_breakdown])
                logging.debug(u"CONTINUATION: {} -> {}, {}".format(utils.apply_trg_wmap(hypo.trgt_sentence),next_score, hypo.score))
                if self._is_closed(hypo):
                    next_hypos.append(hypo)
                    next_scores.append(next_score)
                    logging.debug(u"NOT EXPAND: {} -> {}, {}".format(utils.apply_trg_wmap(hypo.trgt_sentence),next_score,hypo.score))
                    continue
                for next_hypo in super(SyncBeamDecoderSegm, self)._expand_hypo_nmt(hypo):
                    next_hypos.append(next_hypo)
                    next_score = sum([sum([char_scores[i][0] for i,s in enumerate(char_scores) if self.predictor_names[i]=="nmt"]) for char_scores in next_hypo.score_breakdown])
#                    next_score = sum([sum([char_scores[i][0] for i,s in enumerate(char_scores) if self.predictor_levels[i]=="c"]) for char_scores in next_hypo.score_breakdown])
                    next_scores.append(next_score)
                    logging.debug(u"EXPAND: {} -> {}, {}".format(utils.apply_trg_wmap(next_hypo.trgt_sentence),next_score,next_hypo.score))
            logging.debug(u"BEFORE CUT on ITERATION: {} -> {}".format(it, " && ".join(utils.apply_trg_wmap(h.trgt_sentence) + ", " + str(next_scores[i]) for i,h in enumerate(next_hypos))))

            hypos = self._get_next_hypos(next_hypos, next_scores)
            logging.debug(u"CUT: {}".format(" && ".join(utils.apply_trg_wmap(h.trgt_sentence) for h in hypos)))
    
        # Best final expansion of the initial hypo by morphemes
        for hypo in hypos:
            logging.debug(u"SYNCRESULT {} {}".format(utils.apply_trg_wmap(hypo.trgt_sentence), sum([sum([char_scores[i][0]  for i,s in enumerate(char_scores) if self.predictor_names[i]=="nmt"]) for char_scores in hypo.score_breakdown])))
#            logging.debug(u"SYNCRESULT {} {}".format(utils.apply_trg_wmap(hypo.trgt_sentence), sum([sum([char_scores[i][0]  for i,s in enumerate(char_scores) if self.predictor_levels[i]=="c"]) for char_scores in hypo.score_breakdown])))

        return hypos



