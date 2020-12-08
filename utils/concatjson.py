#!/usr/bin/env python2
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('jsons', type=str, nargs='+',
                       help='json files',default="/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/aishell_char_vggblstmp_e8_subsample1_2_2_1_1_skip_unit320_proj320_d1_unit300_location_softmax_aconvc10_aconvf100_lsm_typenone_lsm_weight0.0_num_save_attention0.5_adadelta_bs30_mli800_mlo150_dropout0.0_fusionnone/playground/data.json")
    args = parser.parse_args()
    
    # logging info
    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    #file_directory ="/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/playground/data.json"
    # make intersection set for utterance keys
    js = {}
    for x in args.jsons:
    #for x in file_directory:
        print(x)
        with open(x, 'r') as f:
            j = json.load(f)
        ks = j['utts'].keys()
        logging.debug(x + ': has ' + str(len(ks)) + ' utterances')
        js.update(j['utts'])
    logging.info('new json has ' + str(len(js.keys())) + ' utterances')
        
    # ensure "ensure_ascii=False", which is a bug
    jsonstring = json.dumps({'utts': js}, indent=4, sort_keys=True, ensure_ascii=False).encode('utf_8')
    print(jsonstring)
