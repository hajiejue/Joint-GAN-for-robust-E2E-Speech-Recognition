#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


# general configuration






dataroot="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
dictroot="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
train_set=train
train_dev=dev
recog_set=test

# you can skip this and remove --rnnlm option in the recognition (stage 5)
dict=${dictroot}/${train_set}/vocab
embed_init_file=${dictroot}/char_embed_vec


expdir="/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone/playground"
echo "expdir: ${expdir}"
#name=asr_clean_syllable_fbank80_drop0.2
name=playground
lmexpdir=checkpoints/train_rnnlm_2layer_256_650_drop0.2_bs64
#fst_path="/home/bliu/mywork/workspace/e2e/data/lang_word/LG_pushed_withsyms.fst"
#nn_char_map_file="/home/bliu/mywork/workspace/e2e/data/lang_word/net_chars.txt"

echo "stage 5: Decoding"
echo "111111"


          
. utils/score_sclite.sh  ${expdir} ${dict}
        
        ##kenlm_path="/home/bliu/mywork/workspace/e2e/src/kenlm/build/text_character.arpa"
        ##rescore_sclite.sh --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${expdir}/${decode_dir}_rescore ${dict} ${kenlm_path}
    ##) &

    ##wait
echo "Finished"

