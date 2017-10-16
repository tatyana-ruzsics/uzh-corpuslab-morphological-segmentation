"""This module is derived from the ``sampling`` Blocks module in the SGNMT
    framework but reduced to providing functionality for model
    selection according the acuuracy score on the dev set.
    Additionaly, it contains the default NMT configuration for morphological 
    segmentation model (cED+LM). The code is taken from the `nmt``
    Blocks module in the SGNMT framework but relies on the classes from
    blocks_vanilla_decoder,
    """

from __future__ import print_function

from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch
import logging
import numpy
import operator
import os
import re
import signal
#from subprocess import Popen, PIPE
import time

import subprocess
import codecs
import sys

from cam.sgnmt import utils
from cam.sgnmt.blocks.sparse_search import SparseBeamSearch
from cam.sgnmt.misc.sparse import FlatSparseFeatMap

BLOCKS_AVAILABLE = True
try:
    from blocks.search import BeamSearch # To check if blocks is available
    
    from blocks_vanilla_decoder import BlocksNMTVanillaDecoder,\
        BlocksNMTEnsembleVanillaDecoder
    from blocks_nmt import BlocksNMTPredictor, BlocksUnboundedNMTPredictor
except:
    BLOCKS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AccValidator(SimpleExtension):
    """Implements early stopping based on accuracy score.
    """
    
    def __init__(self,
                 source_sentence,
                 samples,
                 model,
                 data_stream,
                 config,
                 n_best=1,
                 track_n_models=1,
                 normalize=True,
                 store_full_main_loop=False,
                 **kwargs):
        """Creates a new extension which adds model selection based on
            the accuracy score to the training main loop.
            
        Args:
            source_sentence (Variable): Input variable to the sampling
            computation graph
            samples (Variable): Samples variable of the CG
            model (NMTModel): See the model module
            data_stream (DataStream): Data stream to the development
            set
            config (dict): NMT configuration
            n_best (int): beam size
            track_n_models (int): Number of n-best models for which to
            create checkpoints.
            normalize (boolean): Enables length normalization
            store_full_main_loop (boolean): Stores the iteration state
            in the old style of
            Blocks 0.1. Not recommended
            """
        super(AccValidator, self).__init__(**kwargs)
        self.store_full_main_loop = store_full_main_loop
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.config = config
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.normalize = normalize
        self.best_models = []
        self.val_bleu_curve = []
        
        self.src_sparse_feat_map = config['src_sparse_feat_map'] if config['src_sparse_feat_map'] \
            else FlatSparseFeatMap()
        if config['trg_sparse_feat_map']:
            self.trg_sparse_feat_map = config['trg_sparse_feat_map']
            self.beam_search = SparseBeamSearch(
                                                samples=samples,
                                                trg_sparse_feat_map=self.trg_sparse_feat_map)
        else:
            self.trg_sparse_feat_map = FlatSparseFeatMap()
            self.beam_search = BeamSearch(samples=samples)
        
        # Create saving directory if it does not exist
        if not os.path.exists(self.config['saveto']):
            os.makedirs(self.config['saveto'])
        
        if self.config['reload']:
            try:
                bleu_score = numpy.load(os.path.join(self.config['saveto'],
                                                     'val_bleu_scores.npz'))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()
                # Track n best previous bleu scores
                for i, bleu in enumerate(
                    sorted(self.val_bleu_curve, reverse=True)):
                        if i < self.track_n_models:
                            self.best_models.append(ModelInfo(bleu))
                logging.info("BleuScores Reloaded")
            except:
                logging.info("BleuScores not Found")

        self.verbose = self.config.get('val_set_out', None)
        utils.load_trg_wmap(self.config['trg_wmap'])
        self.trg_wmap = utils.trg_wmap
                    
#    def __init__(self, *args, **kwargs):
#        
#        super(AccValidator, self).__init__(*args, **kwargs)
#        self.verbose = self.config.get('val_set_out', None)
#        utils.load_trg_wmap(self.config['trg_wmap'])
#        self.trg_wmap = utils.trg_wmap

    def do(self, which_callback, *args):
        """Decodes the dev set and stores checkpoints in case the BLEU
        score has improved.
        """
        #if self.main_loop.status['iterations_done'] <= \
        #        self.config['val_burn_in']:
        if self.main_loop.status['epochs_done'] <= self.config['val_burn_in']:
            return
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):
        """Evaluate model and store checkpoints. """
        logging.info("Started Validation: ")
        val_start_time = time.time()
        total_cost = 0.0
        if self.verbose:
            ftrans = codecs.open(self.config['val_set_out'], 'w', 'utf-8')
        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            seq = self.src_sparse_feat_map.words2dense(utils.oov_to_unk(
                                                                    line[0], self.config['src_vocab_size']))
            if self.src_sparse_feat_map.dim > 1: # sparse src feats
                input_ = numpy.transpose(
                                         numpy.tile(seq, (self.config['beam_size'], 1, 1)),
                                         (2,0,1))
            else: # word ids on the source side
                input_ = numpy.tile(seq, (self.config['beam_size'], 1))
            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                                        input_values={self.source_sentence: input_},
                                        max_length=3*len(line[0]), eol_symbol=utils.EOS_ID,
                                        ignore_first_eol=True)
                    #            if i < 10:
                    #                logging.info("ID: {}".format(i))
                    #                logging.info("Source: {}".format(line[0]))
                    #                for k, tran in enumerate(trans):
                    #                    logging.info(u"{}".format(utils.apply_trg_wmap(tran,self.trg_wmap)))
                    #                    logging.info("{}".format(costs[k]))
                    # normalize costs according to the sequence lengths
            if self.normalize:
                lengths = numpy.array([len(s) for s in trans])
                costs = costs / lengths
                                
            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans = trans[best]
                    if trans and trans[-1] == utils.EOS_ID:
                        trans = trans[:-1]
                    trans_out = ' '.join([str(w) for w in trans])
                except ValueError:
                    logging.info(
                             "Can NOT find a translation for line: {}".format(i+1))
                    trans_out = '<UNK>'
                    trans = 0
                if j == 0:
                    # Write to subprocess and file if it exists
                    ##print(trans_out, file=mb_subprocess.stdin)
                    if self.verbose:
                        print(utils.apply_trg_wmap(trans,self.trg_wmap), file=ftrans)
            if i != 0 and i % 100 == 0:
                logging.info(
                    "Translated {} lines of validation set...".format(i))
                                        
        logging.info("Total cost of the validation: {}".format(total_cost))
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()
        logging.info("Validation Took: {} minutes".format(
                                                           float(time.time() - val_start_time) / 60.))
        logger.info("{} {} {} {}".format(self.config['bleu_script'], self.config['val_set_out'], self.config['val_set_grndtruth'], self.config['results_out']))
        bleu_score = float(subprocess.check_output("python2.7 {} {} {} {}".format(self.config['bleu_script'], self.config['val_set_out'], self.config['val_set_grndtruth'], self.config['results_out']), shell=True).decode("utf-8"))
        self.val_bleu_curve.append(bleu_score)
        logging.info(bleu_score)
        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
                                       key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False
            
    def save_parameter_values(self, param_values, path):
        ''' This method is copied from blocks.machine_translation.checkpoint '''
        param_values = {name.replace("/", "-"): param
            for name, param in param_values.items()}
        numpy.savez(path, **param_values)
            
    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.config['saveto'])
            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logging.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)
            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))
            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            # fs439: introduce store_full_main_loop and
            # storing best_bleu_params_* files
            if self.store_full_main_loop:
                logging.info("Saving full main loop model {}".format(model.path))
                numpy.savez(model.path,
                            **self.main_loop.model.get_parameter_dict())
            else:
                logging.info("Saving model parameters {}".format(model.path))
                params_to_save = self.main_loop.model.get_parameter_values()
                self.save_parameter_values(params_to_save, model.path)
            numpy.savez(
                        os.path.join(self.config['saveto'], 'val_bleu_scores.npz'),
                        bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


def blocks_get_default_nmt_config():
    """Get default NMT configuration. """
    config = {}
    
    # Model related -----------------------------------------------------------
    
    # Sequences longer than this will be discarded
    config['seq_len'] = 50
    
    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 100#1000
    config['dec_nhids'] = 100#1000
    config['att_nhids'] = -1
    config['maxout_nhids'] = -1
    
    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 300#620
    config['dec_embed'] = 300#620
    
    # Number of layers in encoder and decoder
    config['enc_layers'] = 1
    config['dec_layers'] = 1
    
    # Network layout
    config['dec_readout_sources'] = "sfa"
    config['dec_attention_sources'] = "s"
    
    config['enc_share_weights'] = True
    config['dec_share_weights'] = True
    
    # Skip connections
    config['enc_skip_connections'] = False
    
    # How to derive annotations from the encoder. Comma
    # separated list of strategies.
    # - 'direct': directly use encoder hidden state
    # - 'hierarchical': Create higher level annotations with an
    #                   attentional RNN
    config['annotations'] = "direct"
    
    # Decoder initialisation
    config['dec_init'] = "last"
    
    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = './train'
    
    # Attention
    config['attention'] = 'content'
    
    # External memory structure
    config['memory'] = 'none'
    config['memory_size'] = 500
    
    # Optimization related ----------------------------------------------------
    
    # Batch size
    config['batch_size'] = 20#80
    
    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12
    
    # Optimization step rule
    config['step_rule'] = 'AdaDelta'
    
    # Gradient clipping threshold
    config['step_clipping'] = 1.
    
    # Std of weight initialization
    config['weight_scale'] = 0.01
    
    # Regularization related --------------------------------------------------
    
    # Weight noise flag for feed forward layers
    config['weight_noise_ff'] = 0.0
    
    # Weight noise flag for recurrent layers
    config['weight_noise_rec'] = False
    
    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 1.0
    
    # Vocabulary/dataset related ----------------------------------------------
    
    # Root directory for dataset
    datadir = './data/'
    scriptsdir = '../scripts/'
    
    # Source and target datasets
    config['src_data'] = datadir + 'train.iwords-shuf'#'train.ids.shuf.en'
    config['trg_data'] = datadir + 'train.isegs-shuf'#'train.ids.shuf.fr'
    
    # Monolingual data (for use see --mono_data_integration
    config['src_mono_data'] = datadir + 'mono.iwords'#'mono.ids.en'
    config['trg_mono_data'] = datadir + 'mono.isegs'#'mono.ids.fr'
    
    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 127#30003
    config['trg_vocab_size'] = 81#30003
    
    # Mapping files for using sparse feature word representations
    config['src_sparse_feat_map'] = ""
    config['trg_sparse_feat_map'] = ""
    
    # Early stopping based on bleu related ------------------------------------
    
    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True
    
    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = scriptsdir + 'accuracy.py'
    #config['bleu_script'] = 'perl ' + scriptsdir + 'multi-bleu.perl %s <'
    
    # Validation set source file (ids)
    config['val_set'] = datadir + 'dev.iwords'
    
    # Validation set source file (chars)
    config['val_set_in'] = datadir + 'dev.words'
    
    #Target map from ids to chars
    config['trg_wmap'] = datadir + 'vocab.segs'
    
    # Validation set gold file
    config['val_set_grndtruth'] = datadir + 'dev.segs'#'dev.ids.fr'
    
    # Print validation output to file
    config['output_val_set'] = True
    
    # Validation output file
    config['val_set_out'] = datadir + '/val_out.txt'
    
    # File to save the accuracy results
    config['results_out'] = datadir + '/results_per_epoch.txt'
    
    # Beam-size
    config['beam_size'] = 12
    
    # Timing/monitoring related -----------------------------------------------
    
    # Maximum number of updates
    config['finish_after'] = 10000#1000000
    
    # Reload model from files if exist
    config['reload'] = True
    
    # Save model after this many updates
    config['save_freq'] = 10000#750
    
    # Validate bleu after this many updates
    config['bleu_val_freq'] = 50#6000
    
    # Start bleu validation after this many updates
    config['val_burn_in'] = 1#80000
    
    # fs439: Blocks originally creates dumps of the entire main loop
    # when the BLEU on the dev set improves. This, however, cannot be
    # read to load parameters from, so we create BEST_BLEU_PARAMS*
    # files instead. Set the following parameter to true if you still
    # want to create the old style archives
    config['store_full_main_loop'] = False
    
    # fs439: Fix embeddings when training
    config['fix_embeddings'] = False
    
    return config

def blocks_get_nmt_config_help():
    """Creates a dictionary with help text for the NMT configuration """
    
    config = {}
    config = {}
    config['seq_len'] = "Sequences longer than this will be discarded"
    config['enc_nhids'] = "Number of hidden units in encoder GRU"
    config['dec_nhids'] = "Number of hidden units in decoder GRU"
    config['att_nhids'] = "Dimensionality of attention match vector (-1 to " \
        "use dec_nhids)"
    config['maxout_nhids'] = "Dimensionality of maxout output layer (-1 to " \
        "use dec_nhids)"
    config['enc_embed'] = "Dimension of the word embedding matrix in encoder"
    config['dec_embed'] = "Dimension of the word embedding matrix in decoder"
    config['enc_layers'] = "Number of encoder layers"
    config['dec_layers'] = "Number of decoder layers (NOT IMPLEMENTED for != 1)"
    config['dec_readout_sources'] = "Sources used by readout network: f for " \
        "feedback, s for decoder states, a for " \
            "attention (context vector)"
    config['dec_attention_sources'] = "Sources used by attention: f for " \
        "feedback, s for decoder states"
    config['enc_share_weights'] = "Whether to share weights in deep encoders"
    config['dec_share_weights'] = "Whether to share weights in deep decoders"
    config['enc_skip_connections'] = "Add skip connection in deep encoders"
    config['annotations'] = "Annotation strategy (comma-separated): " \
        "direct, hierarchical"
    config['dec_init'] = "Decoder state initialisation: last, average, constant"
    config['attention'] = "Attention mechanism: none, content, nbest-<n>, " \
        "coverage-<n>, tree, content-<n>"
    config['memory'] = 'External memory: none, stack'
    config['memory_size'] = 'Size of external memory structure'
    config['saveto'] = "Where to save model, same as 'prefix' in groundhog"
    config['batch_size'] = "Batch size"
    config['sort_k_batches'] = "This many batches will be read ahead and sorted"
    config['step_rule'] = "Optimization step rule"
    config['step_clipping'] = "Gradient clipping threshold"
    config['weight_scale'] = "Std of weight initialization"
    config['weight_noise_ff'] = "Weight noise flag for feed forward layers"
    config['weight_noise_rec'] = "Weight noise flag for recurrent layers"
    config['dropout'] = "Dropout ratio, applied only after readout maxout"
    config['src_data'] = "Source dataset"
    config['trg_data'] = "Target dataset"
    config['src_mono_data'] = "Source language monolingual data (for use " \
        "see --mono_data_integration)"
    config['trg_mono_data'] = "Target language monolingual data (for use " \
        "see --mono_data_integration)"
    config['src_vocab_size'] = "Source vocab size, including special tokens"
    config['trg_vocab_size'] = "Target vocab size, including special tokens"
    config['src_sparse_feat_map'] = "Mapping files for using sparse feature " \
        "word representations on the source side"
    config['trg_sparse_feat_map'] = "Mapping files for using sparse feature " \
        "word representations on the target side"
    config['normalized_bleu'] = "Length normalization IN TRAINING"
    config['bleu_script'] = "BLEU script used during training for model selection"
    config['val_set_grndtruth'] = "Validation set gold file"
    config['output_val_set'] = "Print validation output to file"
    config['val_set_out'] = "Validation output file"
    config['beam_size'] = "Beam-size for decoding DURING TRAINING"
    config['finish_after'] = "Maximum number of updates"
    config['reload'] = "Reload model from files if exist"
    config['save_freq'] = "Save model after this many updates"
    config['bleu_val_freq'] = "Validate bleu after this many updates"
    config['val_burn_in'] = "Start bleu validation after this many updates"
    config['store_full_main_loop'] = "Old style archives (not recommended)"
    config['fix_embeddings'] = "Fix embeddings during training"
    config['val_set'] = "Validation set source file (in ids)"
    config['results_out'] = "File to save the accuracy results"
    config['val_set_in'] = "Validation set source file (in chars)"
    config['trg_wmap'] = "Target map from ids to chars"
    return config

def blocks_add_nmt_config(parser):
    """Adds the nmt options to the command line configuration.
        
        Args:
        parser (object): Parser or ArgumentGroup object
        """
    default_config = blocks_get_default_nmt_config()
    nmt_help_texts = blocks_get_nmt_config_help()
    for k in default_config:
        arg_type = type(default_config[k])
        if arg_type == bool:
            arg_type = 'bool'
        parser.add_argument(
                            "--%s" % k,
                            default=default_config[k],
                            type=arg_type,
                            help=nmt_help_texts[k])


def _add_sparse_feat_maps_to_config(nmt_config):
    """Adds the sparse feature map instances to the nmt config """
    new_config = dict(nmt_config)
    if nmt_config['src_sparse_feat_map']:
        new_config['src_sparse_feat_map'] = FileBasedFeatMap(
                                                             nmt_config['enc_embed'],
                                                             nmt_config['src_sparse_feat_map'])
    if nmt_config['trg_sparse_feat_map']:
        new_config['trg_sparse_feat_map'] = FileBasedFeatMap(
                                                             nmt_config['dec_embed'],
                                                             nmt_config['trg_sparse_feat_map'])
    return new_config

def blocks_get_nmt_predictor(args, nmt_path, nmt_config):
    """Get the Blocks NMT predictor. If a target sparse feature map is
        used, we create an unbounded vocabulary NMT predictor. Otherwise,
        the normal bounded NMT predictor is returned
        
        Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_config (dict): NMT configuration
        
        Returns:
        Predictor. The NMT predictor
        """
    if not BLOCKS_AVAILABLE:
        logging.fatal("Could not find Blocks!")
        return None
    nmt_config = _add_sparse_feat_maps_to_config(nmt_config)
    if nmt_path:
        nmt_config['saveto'] = nmt_path
    if nmt_config['trg_sparse_feat_map']:
        return BlocksUnboundedNMTPredictor(
                                           get_nmt_model_path(args.nmt_model_selector,
                                                              nmt_config),
                                           args.gnmt_beta,
                                           nmt_config)
    return BlocksNMTPredictor(get_nmt_model_path(args.nmt_model_selector,
                                                 nmt_config),
                              args.gnmt_beta,
                              args.cache_nmt_posteriors,
                              nmt_config)

def blocks_get_nmt_vanilla_decoder(args, nmt_specs):
    """Get the Blocks NMT vanilla decoder.
        
        Args:
        args (object): SGNMT arguments from ``ArgumentParser``
        nmt_specs (list): List of (nmt_path,nmt_config) tuples, one
        entry for each model in the ensemble
        
        Returns:
        Predictor. An instance of ``BlocksNMTVanillaDecoder``
        """
    if not BLOCKS_AVAILABLE:
        logging.fatal("Could not find Blocks!")
        return None
    nmt_specs_blocks = []
    for nmt_path, nmt_config in nmt_specs:
        nmt_config = _add_sparse_feat_maps_to_config(nmt_config)
        if nmt_path:
            nmt_config['saveto'] = nmt_path
        nmt_specs_blocks.append((get_nmt_model_path(args.nmt_model_selector,
                                                    nmt_config),
                                 nmt_config))
    if len(nmt_specs_blocks) == 1:
        return BlocksNMTVanillaDecoder(nmt_specs_blocks[0][0],
                                       nmt_specs_blocks[0][1],
                                       args)
    return BlocksNMTEnsembleVanillaDecoder(nmt_specs_blocks, args)

PARAMS_FILE_NAME = 'params.npz'
"""Name of the default model file (not checkpoints) """


BEST_BLEU_PATTERN = re.compile('^best_bleu_params_([0-9]+)_BLEU([.0-9]+).npz$')
"""Pattern for checkpoints created in training for model selection """


def get_nmt_model_path_params(nmt_config):
    """Returns the path to the params.npz. This file usually contains
        the latest model parameters.
        
        Args:
        nmt_config (dict):  NMT configuration. We will use the field
        ``saveto`` to get the training directory
        
        Returns:
        string. Path to the params.npz
        """
    return '%s/%s' % (nmt_config['saveto'], PARAMS_FILE_NAME)


def get_nmt_model_path_best_bleu(nmt_config):
    """Returns the path to the checkpoint with the best BLEU score. If
        no checkpoint can be found, back up to params.npz.
        
        Args:
        nmt_config (dict):  NMT configuration. We will use the field
        ``saveto`` to get the training directory
        
        Returns:
        string. Path to the checkpoint file with best BLEU score
        """
    best = 0.0
    best_model = get_nmt_model_path_params(nmt_config)
    for f in os.listdir(nmt_config['saveto']):
        m = BEST_BLEU_PATTERN.match(f)
        if m and float(m.group(2)) > best:
            best = float(m.group(2))
            best_model = '%s/%s' % (nmt_config['saveto'], f)
    return best_model


def get_nmt_model_path_most_recent(nmt_config):
    """Returns the path to the most recent checkpoint. If
        no checkpoint can be found, back up to params.npz.
        
        Args:
        nmt_config (dict):  NMT configuration. We will use the field
        ``saveto`` to get the training directory
        
        Returns:
        string. Path to the most recent checkpoint file
        """
    best = 0
    best_model = get_nmt_model_path_params(nmt_config)
    for f in os.listdir(nmt_config['saveto']):
        m = BEST_BLEU_PATTERN.match(f)
        if m and int(m.group(1)) > best:
            best = int(m.group(1))
            best_model = '%s/%s' % (nmt_config['saveto'], f)
    return best_model


def get_nmt_model_path(nmt_model_selector, nmt_config):
    """Get the path to the NMT model according the given NMT config.
        This switches between the most recent checkpoint, the best BLEU
        checkpoint, or the latest parameters (params.npz). This method
        delegates to ``get_nmt_model_path_*``. This
        method relies on the global ``args`` variable.
        
        Args:
        nmt_model_selector (string): the ``--nmt_model_selector`` arg
        which defines the policy to decide
        which NMT model to load (params,
        bleu, or time)
        nmt_config (dict):  NMT configuration, see ``get_nmt_config()``
        
        Returns:
        string. Path to the NMT model file
        """
    if nmt_model_selector == 'params':
        return get_nmt_model_path_params(nmt_config)
    elif nmt_model_selector == 'bleu':
        return get_nmt_model_path_best_bleu(nmt_config)
    elif nmt_model_selector == 'time':
        return get_nmt_model_path_most_recent(nmt_config)
    logging.fatal("NMT model selector %s not available. Please double-check "
                  "the --nmt_model_selector parameter." % nmt_model_selector)


