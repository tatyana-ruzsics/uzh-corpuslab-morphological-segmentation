# Morphological Segmentation
This repository contains the source code for canonical morphological segmentation presented in Tatyana Ruzsics and Tanja Samardzic ["Neural Sequence-to-sequence Learning of Internal Word  Structure"](http://www.aclweb.org/anthology/K17-2004). In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017). Vancouver, Canada.

#### Installation

The code uses [SGNMT](http://ucam-smt.github.io/sgnmt/html/index.html) framework and depends on the [Blocks](http://blocks.readthedocs.io/en/latest/) and  [srilm-swig](https://github.com/desilinguist/swig-srilm) libraries. Follow the [SGNMT instructions](http://ucam-smt.github.io/sgnmt/html/setup.html) to install these dependencies. The implementation also relies on the adapted version of [Z-MERT](http://cs.jhu.edu/~ozaidan/zmert/). After installation update the enviromental variables LD_LIBRARY_PATH,PYTHONPATH, PATh in the header of the main executable Main.sh file with the location of swig and SRILM. 

#### Running new experiments

The main executable is Main.sh:

```
Main.sh PATHtoDATA PATHtoWorkingDir ResultsFolderName NMT_ENSEMBLES BEAM USE_LENGTH_CONTROL
```


#### Running the experiments in the paper
The data folder contains the [datasets for canonical segmentation](https://github.com/ryancotterell/canonical-segmentation). 
```
Main.sh /data/canonical-segmentation/indonesian/ /experiments/ind results 5 12 -l
Main.sh /data/canonical-segmentation/german/ /experiments/ger results 5 12 -l
Main.sh /data/canonical-segmentation/english/ /experiments/eng results 5 12 -l
```
