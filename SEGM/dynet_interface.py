"""This module is the interface to the Dynet NMT and RNNLM implementations.
    """
import logging
import os

DYNET_AVAILABLE = True
try:
    from dynet_nmt import DynetNMTPredictor,DynetNMTEnsembleDecoder,DynetNMTVanillaDecoder
    from dynet_rnnlm import DynetRNNLMPredictor
    import dynet as dy
except:
    DYNET_AVAILABLE = False
from dynet_nmt import DynetNMTPredictor,DynetNMTEnsembleDecoder,DynetNMTVanillaDecoder
from dynet_rnnlm import DynetRNNLMPredictor
import dynet as dy


def dynet_get_nmt_predictor(nmt_path):
    """Get the TensorFlow NMT predictor.
        
        Args:
        path (string): Path to NMT model or directory
        
        Returns:
        Predictor. An instance of ``DynetRNNLMPredictor``
        """
    if not DYNET_AVAILABLE:
        logging.fatal("Could not find DyNet nmt pr!")
        return None
    
    logging.info("Loading DyNet nmt predictor")
    return DynetNMTPredictor(nmt_path)

def dynet_get_nmt_vanilla_decoder(args, nmt_paths):
    """Get the Blocks NMT vanilla decoder.
        
        Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_specs (list): List of nmt_path, one
        entry for each model in the ensemble
        
        Returns:
        Predictor. An instance of ``DynetNMTVanillaDecoder``
        """
    if not DYNET_AVAILABLE:
        logging.fatal("Could not find DyNet nmt dec!")
        return None
    
    logging.info("Loading DyNet nmt decoder")
    if len(nmt_paths) == 1:
        return DynetNMTVanillaDecoder(nmt_paths[0], args)
    return DynetNMTEnsembleDecoder(nmt_paths, args)


def dynet_get_rnnlm_predictor(rnnlm_path):
    """Get the TensorFlow RNNLM predictor.
        
        Args:
        path (string): Path to RNNLM model or directory
        
        Returns:
        Predictor. An instance of ``DynetRNNLMPredictor``
        """
    if not DYNET_AVAILABLE:
        logging.fatal("Could not find DyNet lm pr!")
        return None
    
    logging.info("Loading DyNet rnnlm predictor")
    return DynetRNNLMPredictor(rnnlm_path)
