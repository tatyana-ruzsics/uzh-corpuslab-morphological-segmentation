#!/bin/bash
# Usage: Main.sh AbsolutePATHtoDATA AbsolutePATHtoWorkingDir ResultsFolderName NMT_ENSEMBLES BEAM USE_LENGTH_CONTROL
# Usage: Main.sh /Users/tatianaruzsics/NN/Segmentation/data/canonical-segmentation/indonesian/ /Users/tatianaruzsics/NN/Segmentation/experiments/ind results 5 3 -l
# Usage: Main.sh /Users/tatianaruzsics/NN/Segmentation/data/canonical-segmentation/german/ /Users/tatianaruzsics/NN/Segmentation/experiments/ger results 5 3 -l
# Usage: Main.sh /Users/tatianaruzsics/NN/Segmentation/data/canonical-segmentation/english/ /Users/tatianaruzsics/NN/Segmentation/experiments/eng results 5 3 -l
###########################################
## POINTERS TO WORKING AND DATA DIRECTORIES
###########################################
#
export DATA=$1
mkdir $DATA
export EXPER=$2
mkdir $EXPER
export SCRIPTS=/Users/tatianaruzsics/NN/Segmentation/scripts
export SEGM=/Users/tatianaruzsics/NN/Segmentation/SEGM
export THEANO_FLAGS="on_unused_input='ignore'"

#LM paths
export LD_LIBRARY_PATH=/Users/tatianaruzsics/NN/Segmentation/swig-srilm:$LD_LIBRARY_PATH
export PYTHONPATH=/Users/tatianaruzsics/NN/Segmentation/swig-srilm:$PYTHONPATH
export PATH=/Users/tatianaruzsics/NN/Segmentation/SRILM/bin/macosx:$PATH

#MERT path
export MERT=/Users/tatianaruzsics/NN/Segmentation/zmert_v1.50


#n is the number of train/test/dev split in data/canonical-segmentation to use

for (( n=0; n<=0; n++ ))


#for n in {0,1}
do
(
# Experiments directories
mkdir $EXPER/exper_data_$n
export EXPER_DATA=$EXPER/exper_data_$n

mkdir -p $EXPER/train/$n
export MODEL=$EXPER/train/$n

export RESULTS_RELAT_DIR_NAME=$3
mkdir -p $EXPER/$RESULTS_RELAT_DIR_NAME
export RESULTS=$EXPER/$RESULTS_RELAT_DIR_NAME
mkdir $RESULTS/$n

export NMT_ENSEMBLES=$4
export BEAM=$5

#
###########################################
## PREPARATION - masking and vocabulary
###########################################
#

# Prepare train set (charcter based - add spaces)
python2.7 $SCRIPTS/SrcTrgSplit.py $DATA/train$n $EXPER_DATA/train.words $EXPER_DATA/train.segs

# Prepare test set (based on types) (charcter based - add spaces)
python2.7 $SCRIPTS/SrcTrgSplit.py $DATA/test$n $EXPER_DATA/test.words $EXPER_DATA/test.segs

# Prepare validation set (based on types) (charcter based - add spaces)
python2.7 $SCRIPTS/SrcTrgSplit.py $DATA/dev$n $EXPER_DATA/dev.words $EXPER_DATA/dev.segs

# Prepare target and source dictionaries

python2.7 $SCRIPTS/build_dict.py -c --num -f $EXPER_DATA/train.segs -d $EXPER_DATA/vocab.segs
python2.7 $SCRIPTS/build_dict.py -c --num -f $EXPER_DATA/train.words -d $EXPER_DATA/vocab.words

# Convert chars to numbers - only needed for training

python2.7 $SCRIPTS/apply_wmap.py -m $EXPER_DATA/vocab.words < $EXPER_DATA/train.words > $EXPER_DATA/train.iwords
python2.7 $SCRIPTS/apply_wmap.py -m $EXPER_DATA/vocab.segs < $EXPER_DATA/train.segs > $EXPER_DATA/train.isegs

python2.7 $SCRIPTS/apply_wmap.py -m $EXPER_DATA/vocab.words < $EXPER_DATA/dev.words > $EXPER_DATA/dev.iwords
python2.7 $SCRIPTS/apply_wmap.py -m $EXPER_DATA/vocab.segs < $EXPER_DATA/dev.segs > $EXPER_DATA/dev.isegs

export SrcVocab=$(echo $(wc -l < $EXPER_DATA/vocab.words))

export TrgVocab=$(echo $(wc -l < $EXPER_DATA/vocab.segs))

export SyncSymbol=$(awk '$1 == "|" { print $2 }' $EXPER_DATA/vocab.segs)
echo "SyncSymbol: $SyncSymbol"

echo "SrcVocabSize: $SrcVocab"
echo "TrgVocabSize: $TrgVocab"

##########################################
# TRAINING NMT
##########################################

#Shuffle

#for (( k=1; k<=$NMT_ENSEMBLES; k++ ))
#do
#
#mkdir $MODEL/$k
#python2.7 $SCRIPTS/shuffle.py -s=$k $EXPER_DATA/train.iwords $EXPER_DATA/train.isegs
#
#python /Users/tatianaruzsics/NN/Segmentation/SEGM-original/train.py --finish_after=20 --reload=False --bleu_val_freq=1 --val_burn_in=10 --reshuffle --saveto=$MODEL/$k --results_out=$MODEL/$k/results_per_epoch.txt --src_vocab_size=$SrcVocab --trg_vocab_size=$TrgVocab  --val_set=$EXPER_DATA/dev.iwords  --val_set_grndtruth=$EXPER_DATA/dev.segs --bleu_script=$SCRIPTS/accuracy.py --trg_data=$EXPER_DATA/train.isegs-shuf --src_data=$EXPER_DATA/train.iwords-shuf --trg_wmap=$EXPER_DATA/vocab.segs --val_set_in=$EXPER_DATA/dev.words --val_set_out=$EXPER_DATA/val_out.txt
#
#done

############################################
# DECODING NMT + EVALUATION on dev
############################################

# sgnmt-based decoder
nmt_predictors="nmt"

nmt_path="--nmt_path=$MODEL/1"

if [ $NMT_ENSEMBLES -gt 1 ]; then
while read num; do nmt_predictors+=",nmt"; done < <(seq $(($NMT_ENSEMBLES-1)))
while read num; do nmt_path+=" --nmt_path$num=$MODEL/$num"; done < <(seq 2 $NMT_ENSEMBLES)
fi

decode_cmd_dev="python $SEGM/decode_segm.py --predictors $nmt_predictors --decoder vanilla --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab $nmt_path --src_wmap=$EXPER_DATA/vocab.words --trg_wmap=$EXPER_DATA/vocab.segs  --src_test=$EXPER_DATA/dev.words --outputs=text --output_path=$RESULTS/$n/dev_out_vanilla.txt --beam=$BEAM"

echo $decode_cmd_dev

eval $decode_cmd_dev

# Evaluate on types and print errors

python2.7 $SCRIPTS/accuracy-err.py $RESULTS/$n/dev_out_vanilla.txt  $EXPER_DATA/dev.segs $RESULTS/$n/Accuracy_vanilla_dev.txt > $RESULTS/$n/Errors_vanilla_dev.txt

###########################################
## DECODING NMT + EVALUATION on test
###########################################

decode_cmd_test="python $SEGM/decode_segm.py --predictors $nmt_predictors --decoder vanilla --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab $nmt_path --src_wmap=$EXPER_DATA/vocab.words --trg_wmap=$EXPER_DATA/vocab.segs  --src_test=$EXPER_DATA/test.words --outputs=text --output_path=$RESULTS/$n/test_out_vanilla.txt --beam=$BEAM"

echo $decode_cmd_test

eval $decode_cmd_test

# Evaluate on types and print errors

python2.7 $SCRIPTS/accuracy-err.py $RESULTS/$n/test_out_vanilla.txt  $EXPER_DATA/test.segs $RESULTS/$n/Accuracy_vanilla_test.txt > $RESULTS/$n/Errors_vanilla_test.txt

##########################################
# LM
##########################################

# Prepare dictionary form morphemes (target side)
python2.7 $SCRIPTS/build_dict.py -m -f $EXPER_DATA/train.segs -d $EXPER_DATA/vocab.morfs
# Prepare mapping file (morf id -> char1_id char2_id ..) and training file for lm with masked morhemes by their ids
python2.7 $SCRIPTS/build_dict.py --morf2char -f $EXPER_DATA/train.segs --chardict $EXPER_DATA/vocab.segs --morfdict $EXPER_DATA/vocab.morfs -d $EXPER_DATA/vocab.m2c -o $EXPER_DATA/train-lm.txt

#run LM

#ngram-count -text $EXPER_DATA/train-lm.txt -lm $EXPER_DATA/morfs.lm -order 3 -write $EXPER_DATA/morfs.lm.counts -unk -interpolate -kndiscount1 -kndiscount2 -kndiscount3
#(ngram-count -text $EXPER_DATA/train-lm.txt -lm $EXPER_DATA/morfs.lm -order 3 -write $EXPER_DATA/morfs.lm.counts -interpolate -kndiscount1 -kndiscount2 -kndiscount3) || (ngram-count -text $EXPER_DATA/train-lm.txt -lm $EXPER_DATA/morfs.lm -order 3 -write $EXPER_DATA/morfs.lm.counts -interpolate -kndiscount1 -kndiscount2 )
(ngram-count -text $EXPER_DATA/train-lm.txt -lm $EXPER_DATA/morfs.lm -order 3 -write $EXPER_DATA/morfs.lm.counts -kndiscount -interpolate) || { echo "Backup to ukn {$n}"; (ngram-count -text $EXPER_DATA/train-lm.txt -lm $EXPER_DATA/morfs.lm -order 3 -write $EXPER_DATA/morfs.lm.counts -ukndiscount -interpolate);}


##########################################
# MERT for NMT & LM + EVALUATION
##########################################

(
##To make sure:
# Change -r in ZMERT_cfg.txt to dev.segs

cp -R $MERT/segm $MERT/segm-$n

cd $MERT/segm-$n

if [[ $6 == "-l" ]]; then # Use length control

    # passed to zmert: commands to decode n-best list from dev file
    echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,word2char_srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.words --trg_wmap=$EXPER_DATA/vocab.segs --src_test=$EXPER_DATA/dev.words --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,wc,word2char_srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.words --trg_wmap=$EXPER_DATA/vocab.segs --src_test=$EXPER_DATA/test.words --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

    nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
    #echo $nmt_w
    while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}wc 1.0\nlm 0.1" > SDecoder_cfg.txt

    while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
    echo -e "${nmt_params}wc\t|||\t1.0\tOpt\t0\t+Inf\t0\t+3\nlm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt
else
    # passed to zmert: commands to decode n-best list from dev file
    echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,word2char_srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.words --trg_wmap=$EXPER_DATA/vocab.segs --src_test=$EXPER_DATA/dev.words --outputs=nbest --nbest=0 --output_path=nbest.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd

    # passed to zmert: commands to decode 1-best list from test file
    echo "python $SEGM/decode_segm.py --predictors $nmt_predictors,word2char_srilm --decoder syncbeam $nmt_path --nmt_config src_vocab_size=$SrcVocab,trg_vocab_size=$TrgVocab --word2char_map=$EXPER_DATA/vocab.m2c --srilm_path=$EXPER_DATA/morfs.lm --srilm_order=3 --max_len_factor=3 --src_wmap=$EXPER_DATA/vocab.words --trg_wmap=$EXPER_DATA/vocab.segs --src_test=$EXPER_DATA/test.words --outputs=text --output_path=test.out --sync_symbol=$SyncSymbol --beam=$BEAM" > SDecoder_cmd_test

    nmt_w=$(echo "scale=2;1/$NMT_ENSEMBLES" | bc)
    #echo $nmt_w
    while read num; do nmt_weights+="nmt$num $nmt_w\n"; done < <(seq $NMT_ENSEMBLES)
    echo -e "cands_file=nbest.txt\ncands_per_sen=12\ntop_n=12\n\n${nmt_weights}lm 0.1" > SDecoder_cfg.txt

    while read num; do nmt_params+="nmt$num\t|||\t${nmt_w}\tFix\t0\t+1\t0\t+1\n"; done < <(seq $NMT_ENSEMBLES)
    echo -e "${nmt_params}lm\t|||\t0.1\tOpt\t0\t+Inf\t0\t+1\nnormalization = none" > params.txt
fi

cp $EXPER_DATA/dev.segs $MERT/segm-$n
cp $EXPER_DATA/test.words $MERT/segm-$n

java -cp ../lib/zmert.jar ZMERT -maxMem 500 ZMERT_cfg.txt

# Evaluate on test

python $SCRIPTS/accuracy-err.py test.out  $EXPER_DATA/test.segs $RESULTS/Accuracy_mert_test.txt > $RESULTS/$n/Errors_mert_test.txt

# copy test out file - for analysis
cp test.out $EXPER/train/$n/test_out_mert.txt

# copy n-best file for dev set with optimal weights - for analysis
cp nbest.out $EXPER/train/$n/nbest_dev_mert.out

cp SDecoder_cfg.txt.ZMERT.final $RESULTS/$n/params-mert-ens.txt) &&

rm -r $MERT/segm-$n

#rm -r $EXPER/exper_data_$n

echo "Process {$n} finished" >> $RESULTS/Accuracy_mert_test.txt

) &

done

