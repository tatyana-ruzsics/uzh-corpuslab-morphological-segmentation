"""This module contains the output handlers. These handlers create 
output files from the n-best lists generated by the ``Decoder``. They
can be activated via --outputs.

This script follows SGNMT output handlers cam.sgnmt.output.py but adapted for morphological segmentation (cED+LM): only TextOutputHandler and modified version of NBestOutputHandler (adapted for MERT optimization) are used.

"""

from abc import abstractmethod
#import pywrapfst as fst
import os
import errno
import logging
from cam.sgnmt import utils
import numpy as np
import codecs

#from six.moves import xrange


class OutputHandler(object):
    """Interface for output handlers. """
    
    def __init__(self):
        """ Empty constructor """
        pass
    
    @abstractmethod
    def write_hypos(self, all_hypos):
        """This method writes output files to the file system. The
            configuration parameters such as output paths should already
            have been provided via constructor arguments.
            
            Args:
            all_hypos (list): list of nbest lists of hypotheses
            
            Raises:
            IOError. If something goes wrong while writing to the disk
            """
        raise NotImplementedError

class TextOutputHandler(OutputHandler):
    """Writes the first best hypotheses to a plain text file """
    
    def __init__(self, path, trg_wmap):
        """Creates a plain text output handler to write to ``path`` """
        super(TextOutputHandler, self).__init__()
        self.path = path
        self.trg_wmap = utils.trg_wmap
        
    def write_hypos(self, all_hypos):
        """Writes the hypotheses in ``all_hypos`` to ``path`` """
        if self.f is not None:
          for hypos in all_hypos:
            self.f.write(utils.apply_trg_wmap(hypos[0].trgt_sentence, self.trg_wmap))
            self.f.write("\n")
            self.f.flush()
        else:
          with codecs.open(self.path, "w", encoding='utf-8') as f:
            for hypos in all_hypos:
              f.write(utils.apply_trg_wmap(hypos[0].trgt_sentence, self.trg_wmap))
              f.write("\n")
              self.f.flush()

    def open_file(self):
      self.f = codecs.open(self.path, "w", encoding='utf-8')

    def close_file(self):
      self.f.close()

    def write_empty_line(self):
      if self.f is not None:
         self.f.write("\n")
         self.f.flush()

                
class NBestOutputHandler(OutputHandler):
    """Produces a n-best file in Moses format. The third part of each 
    entry is used to store the separated unnormalized predictor scores.
    Note that the sentence IDs are shifted: Moses n-best files start 
    with the index 0, but in SGNMT and HiFST we usually refer to the 
    first sentence with 1 (e.g. in lattice directories or --range)
    """
    
    def __init__(self, path, predictor_names, start_sen_id, trg_wmap):
        """Creates a Moses n-best list output handler.
        
        Args:
            path (string):  Path to the n-best file to write
            predictor_names: Names of the predictors whose scores
                             should be included in the score breakdown
                             in the n-best list
            start_sen_id: ID of the first sentence
            trg_wmap (dict): (Inverse) word map for target language
        """
        super(NBestOutputHandler, self).__init__()
        self.path = path
        self.start_sen_id = start_sen_id
        self.trg_wmap = utils.trg_wmap
        self.predictor_names = []
        name_count = {}
        for name in predictor_names:
            if not name in name_count:
                name_count[name] = 1
                final_name = name
            else:
                name_count[name] += 1
                #final_name = "%s%d" % (name, name_count[name])
                final_name = name
            self.predictor_names.append(final_name.replace("_", "0"))
        
    def write_hypos(self, all_hypos):
        """Writes the hypotheses in ``all_hypos`` to ``path`` """
        with codecs.open(self.path, "w", encoding='utf-8') as f:
            n_predictors = len(self.predictor_names)
            idx = self.start_sen_id
            for hypos in all_hypos:
                for hypo in hypos:
                    f.write("%d ||| %s ||| %s ||| %f" %
                            (idx,
                             utils.apply_trg_wmap(hypo.trgt_sentence,
                                                  self.trg_wmap),
#                             ' '.join("%s=%f" % (
#                                  self.predictor_names[i],
#                                  sum([s[i][0] for s in hypo.score_breakdown]))
#                                      for i in xrange(n_predictors)),
                             ' '.join("%s" % (sum([s[i][0] for s in hypo.score_breakdown]))
                                      for i in xrange(n_predictors)),
                             hypo.total_score))
                    f.write("\n")
                idx += 1

