#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

nlsyms=""
wer=false
bpe=""
bpemodel=""
remove_blank=true
filter=""

. utils/parse_options.sh
dir="/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/decode_asr_train_table3_1/decode_unmatch"
dic="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/train/vocab"
#dir=$1
#dic=$2
echo "dir"
echo "dic"
#if [ $# != 2 ]; then
    #echo "Usage: $0 <data-dir> <dict>";
    #exit 1;
#fi

#dir=$1
#dic=$2
#dir="/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone/playground"
#dic="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/train/vocab"
#echo "dir"
#echo "dic"
#concatjson.py ${dir}/data.*.json > ${dir}/data.json
#concatjson.py ${dir}/data.json > ${dir}/data.json
#json2trn.py ${dir}/data.json ${dic} ${dir}/ref.trn ${dir}/hyp.trn

if $remove_blank; then
    sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp.trn
fi
if [ ! -z ${nlsyms} ]; then
    cp ${dir}/ref.trn ${dir}/ref.trn.org
    cp ${dir}/hyp.trn ${dir}/hyp.trn.org
    filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
    filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
    echo '1 fuck'
fi
if [ ! -z ${filter} ]; then
    sed -i.bak3 -f ${filter} ${dir}/hyp.trn  #  -i∶直接修改读取的档案内容，而不是由萤幕输出。
    sed -i.bak3 -f ${filter} ${dir}/ref.trn
    echo '2 fuck'
fi
    
sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -i rm -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result.txt

if ${wer}; then
    if [ ! -z $bpe ]; then
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/ref.trn | sed -e "s/▁/ /g" > ${dir}/ref.wrd.trn
	spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp.trn | sed -e "s/▁/ /g" > ${dir}/hyp.wrd.trn
    else
	sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/ref.trn > ${dir}/ref.wrd.trn
	sed -e "s/ //g" -e "s/(/ (/" -e "s/<space>/ /g" ${dir}/hyp.trn > ${dir}/hyp.wrd.trn
    fi
    sclite -r ${dir}/ref.wrd.trn trn -h ${dir}/hyp.wrd.trn trn -i rm -o all stdout > ${dir}/result.wrd.txt

    echo "write a WER result in ${dir}/result.wrd.txt"
    grep -e Avg -e SPKR -m 2 ${dir}/result.wrd.txt
fi
