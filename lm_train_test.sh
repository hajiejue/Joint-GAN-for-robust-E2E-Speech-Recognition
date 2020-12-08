#!/bin/bash
stage=0        # start from 0 if you need to start from data preparation
gpu=            # will be deprecated, please use ngpu
ngpu=0          # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=${resume:=none}    # Resume the training from snapshot

# feature configuration
do_delta=false # true when using CNN

# network archtecture
# encoder related
etype=vggblstmp     # encoder architecture type
elayers=8
eunits=320
eprojs=320
subsample=1_2_2_1_1 # skip every n frame from input to nth layers
subsample_type="skip"
# decoder related
dlayers=1
dunits=300
# attention related
atype=location
aact_func=softmax
aconv_chans=10
aconv_filts=100
lsm_type="none"
lsm_weight=0.0
dropout_rate=0.0
# hybrid CTC/attention
mtlalpha=0.5

# minibatch related
batchsize=30
maxlen_in=800  # if input length  > maxlen_in, batchsize is automatically reduced
maxlen_out=150 # if output length > maxlen_out, batchsize is automatically reduced

# optimization related
opt=adadelta
epochs=30

# rnnlm related
model_unit=char
batchsize_lm=64
dropout_lm=0.5
input_unit_lm=256
hidden_unit_lm=650
lm_weight=0.2
fusion=${fusion:=none}

# decoding parameter
lmtype=rnnlm
beam_size=12
nbest=12
penalty=0.0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0.3
recog_model=acc.best # set a model to be used for decoding: 'acc.best' or 'loss.best'


# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1; 
. ./cmd.sh
. ./path.sh
# check gpu option usage
if [ ! -z $gpu ]; then #判断制定的变量是否存在值
    echo "WARNING: --gpu option will be deprecated."
    echo "WARNING: please use --ngpu option."
    if [ $gpu -eq -1 ]; then
        ngpu=0
    else
        ngpu=1
    fi
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',     --resume $resume \
set -e
set -u
set -o pipefail

#dataroot="/home/bliu/SRC/workspace/e2e/data/clean_aishell/"
dataroot="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
##dictroot="/home/bliu/mywork/workspace/e2e/data/lang_1char/"
dictroot="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/"
train_set=train
train_dev=dev
##recog_set="test_mix test_clean"
##recog_set="test_clean_small"
recog_set="test"

# you can skip this and remove --rnnlm option in the recognition (stage 5)
dict=${dictroot}/${train_set}/vocab
embed_init_file=/usr/home/wudamu/Downloads/sgns
#echo "dictionary: ${dict}"
#nlsyms=${dictroot}/non_lang_syms.txt
lmexpdir=checkpoints/train_${lmtype}_2layer_${input_unit_lm}_${hidden_unit_lm}_drop${dropout_lm}_bs${batchsize_lm}
lmdatadir=${lmexpdir}/local/lm_train
if [ ${stage} -le 3 ]; then
    ${cuda_cmd} ${lmexpdir}/train.log \
        python3 lm_train.py \
        --ngpu 1 \
        --input-unit ${input_unit_lm} \
        --lm-type ${lmtype} \
        --unit ${hidden_unit_lm} \
        --dropout-rate ${dropout_lm} \
        --embed-init-file ${embed_init_file} \
        --verbose 1 \
        --batchsize ${batchsize_lm} \
        --outdir ${lmexpdir} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --dict ${dict}
fi
