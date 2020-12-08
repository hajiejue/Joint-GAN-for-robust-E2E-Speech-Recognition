nlsyms=  
expdir="/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone/playground"
dict="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/train/vocab" 
stage=6

if [ ${stage} -le 5 ]; then
     
    score_sclite.sh --nlsyms ${nlsyms} ${expdir} ${dict}
        

    echo "Finished"
fi
