from tarfile import ENCODING
import torch


class Lm_train:
    def __init__(self):
        self.ngpu = 1
        self.input_unit = 256
        self.lm_type = "rnnlm"
        self.unit = 650
        self.dropout_rate = 0.5
        self.verbose = 1
        self.batchsize = 64
        self.outdir = "checkpoints/rnnlm_train"
        self.train_label = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/lm_train/train.txt"
        self.valid_label = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/lm_train/valid.txt"
        #self.dict = "data/lang_syllable/train_units.txt"
        self.dict = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/train/vocab"
        self.embed_init_file = "/usr/home/wudamu/FP/words_vectors/sgns.wiki.bigram-char"
        self.seed = 1
        self.debugmode = 1
        self.bproplen = 35
        self.epoch = 25
        self.gradclip = 5


class Asr_train:
    def __init__(self):
        self.feat_type = "kaldi_magspec"  #'kaldi_magspec'  # fbank not work
        self.delta_order = 0
        self.left_context_width = 0
        self.right_context_width = 0
        self.normalize_type = 1
        self.num_utt_cmvn = 20000
        self.model_unit = "char"
        self.exp_path = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/asr_train_table1_2"
        self.gpu_ids = 0
        self.dataroot = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        self.name = "asr_train_table1_2"
        self.model_unit = "char"
        self.resume = None
        self.dropout_rate = 0.0
        self.etype = "blstmp"
        self.elayers = 4
        self.eunits = 320
        self.eprojs = 320
        self.subsample = "1_2_2_1_1"
        self.subsample_type = "skip"
        self.dlayers = 1
        self.dunits = 300
        self.atype = "location"
        self.aact_fuc = "softmax"
        self.aconv_chans = 10
        self.aconv_filts = 100
        self.adim = 320
        #self.mtlalpha = 0.5
        self.mtlalpha = 0.1
        self.batch_size = 30
        self.maxlen_in = 800
        self.maxlen_out = 150
        self.opt_type = "adadelta"
        self.verbose = 1
        self.lmtype = "rnnlm"
        self.rnnlm = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/local/rnnlm.model.best"
        self.fusion = "None"
        self.epochs = 15
        self.start_epoch = 0
        self.checkpoints_dir = "./checkpoints"
        #self.dict_dir = r"/home/kang/Develop/Robust_e2e_gan/k2k/data/lang_syllable"
        self.dict_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        self.num_workers = 4
        self.fbank_dim = 80  # 40
        self.lsm_type = ""
        self.lsm_weight = 0.0
        self.works_dir = '/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data'

        self.lr = 0.005
        self.eps = 1e-8  # default=1e-8
        self.iters = 0  # default=0
        self.start_epoch = 0  # default=0›
        self.best_loss = float("inf")  # default=float('inf')
        self.best_acc = 0  # default=0
        self.sche_samp_start_iter = 5
        self.sche_samp_final_iter = 15
        self.sche_samp_final_rate = 0.6
        self.sche_samp_rate = 0.0

        self.shuffle_epoch = -1

        self.enhance_type = "blstm"
        self.fbank_opti_type = "frozen"
        #self.num_utt_cmvn = 20000

        self.grad_clip = 5
        self.print_freq = 500
        self.validate_freq = 2000
        self.num_save_attention = 0.5
        self.criterion = "acc"
        self.mtl_mode = "mtl"
        self.eps_decay = 0.01

class Enhance_base_train(Asr_train):
    def __init__(self):
        super(Enhance_base_train, self).__init__()
        self.ngpu = 1
        self.dataroot = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        self.dict_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        #self.feat_type = "fft" #"kaldi_magspec"
        self.feat_type = "kaldi_magspec"  # "kaldi_magspec"
        self.name = "enhance_base_train"
        self.enhance_resume = False
        self.enhance_type = "blstm"
        # choices=['blstm','unet_128', 'unet_256', 'blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm']
        self.enhance_layers = 3  # type=int, help='Number of enhance model layers'
        self.enhance_units = 128  # type=int, help='Number of enhance model hidden units'
        self.enhance_projs = 128  # type=int, help='Number of enhance model projection units'
        self.enhance_nonlinear_type = "sigmoid"
        # type=str, choices=['sigmoid','relu', 'relu6', 'softplus'], help='enhance_nonlinear_type'
        self.enhance_loss_type = "L2"  # type=str, choices=['L2','L1', 'smooth_L1'], help='enhance_loss_type'
        self.enhance_opt_type = "gan_fbank"  # type=str, choices=['gan_fft','gan_fbank'], help='enhance_'
        self.enhance_dropout_rate = 0.0  # type=float, help='enhance_dropout_rate'
        self.enhance_input_nc = 1  # type=int, help='enhance_input_nc'
        self.enhance_output_nc = 1  # type=int, help='enhance_output_nc'
        self.enhance_ngf = 64  # type=int, help='enhance_ngf'
        self.enhance_norm = "batch"  # type=str, help='enhance_norm'
        self.L1_loss_lambda = 1.0  ##  type=float, help='L1_loss_lambda'
        self.num_saved_specgram = 3
        self.exp_path = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/enhance_base_train"
        #self.exp_path_with_fbank = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/with_fbank"
        self.gpu_ids = "0"
        self.print_freq = 500
class Asr_recog(Asr_train):
    def __init__(self):
        super(Asr_recog, self).__init__()
        self.nj = 4
        self.gpu_ids = "0"
        self.ngpu = 1
        self.nbest = 5
        self.beam_size = 5
        #self.resume = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/asr_train/model.acc.best"
        self.resume = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/asr_train_table1_1/model.acc.best"
        self.recog_dir = "data/test"
        self.result_label = "data.json"
        self.penalty = 0.0
        self.maxlenratio = 0.0
        self.minlenratio = 0.0
        self.ctc_weight = 0.1
        self.lmtype = "rnnlm"
        self.verbose = 1
        self.normalize_type = 1
        #self.rnnlm = "checkpoints/train_rnnlm_2layer_256_650_drop0.5_bs64/rnnlm.model.best"
        self.rnnlm = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/rnnlm_train_shi/rnnlm.model.best"
        # self.fstlm_path ${fst_path} \
        # self.nn_char_map_file ${nn_char_map_file} \
        self.lm_weight = 0.2

        self.exp_path = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/xxx"
        #self.works_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"f
        self.works_dir="/usr/home/wudamu/Documents/Robust_e2e_gan-master"
        self.word_rnnlm = None

        self.input_unit = 256
        self.lm_type = "rnnlm"
        self.unit = 650
        self.embed_init_file = "/usr/home/wudamu/FP/words_vectors/sgns.wiki.bigram-char"
        self.feat_type = "kaldi_magspec"
        self.verbose = 1
        self.normalize_type = 1

class Enhance_fbank_train(Asr_train):
    def __init__(self):
        super(Enhance_fbank_train, self).__init__()
        self.ngpu = 1
        self.dataroot = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        self.dict_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        #self.feat_type = "fft" #"kaldi_magspec"
        self.feat_type = "kaldi_magspec"  # "kaldi_magspec"
        self.enhance_resume = False

        self.enhance_type = "blstm"
        # choices=['blstm','unet_128', 'unet_256', 'blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm']
        self.enhance_layers = 3  # type=int, help='Number of enhance model layers'
        self.enhance_units = 128  # type=int, help='Number of enhance model hidden units'
        self.enhance_projs = 128  # type=int, help='Number of enhance model projection units'
        self.enhance_nonlinear_type = "sigmoid"
        # type=str, choices=['sigmoid','relu', 'relu6', 'softplus'], help='enhance_nonlinear_type'
        self.enhance_loss_type = "L2"  # type=str, choices=['L2','L1', 'smooth_L1'], help='enhance_loss_type'
        self.enhance_opt_type = "gan_fbank"  # type=str, choices=['gan_fft','gan_fbank'], help='enhance_opt_type'
        self.enhance_dropout_rate = 0.0  # type=float, help='enhance_dropout_rate'
        self.enhance_input_nc = 1  # type=int, help='enhance_input_nc'
        self.enhance_output_nc = 1  # type=int, help='enhance_output_nc'
        self.enhance_ngf = 64  # type=int, help='enhance_ngf'
        self.enhance_norm = "batch"  # type=str, help='enhance_norm'
        self.L1_loss_lambda = 1.0  ##  type=float, help='L1_loss_lambda'
        self.num_saved_specgram = 3
        self.exp_path = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/enhance_fbank_train_table_2"
        self.gpu_ids = "0"
        self.print_freq = 100
        self.name = "enhance_fbank_train_table_2"

class Joint_recog(Enhance_base_train,Asr_recog):
    def __init__(self):
        super(Joint_recog, self).__init__()
        self.ngpu = 1
        self.dataroot = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        self.dict_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        #self.feat_type = "fft" #"kaldi_magspec"
        self.feat_type = "kaldi_magspec"  # "kaldi_magspec"
        self.enhance_resume = True
        #self.enhance_resume = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/with_fbank/model.loss.best"
        self.enhance_type = "blstm"
        # choices=['blstm','unet_128', 'unet_256', 'blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm']
        self.enhance_layers = 3  # type=int, help='Number of enhance model layers'
        self.enhance_units = 128  # type=int, help='Number of enhance model hidden units'
        self.enhance_projs = 128  # type=int, help='Number of enhance model projection units'
        self.enhance_nonlinear_type = "sigmoid"
        # type=str, choices=['sigmoid','relu', 'relu6', 'softplus'], help='enhance_nonlinear_type'
        self.enhance_loss_type = "L2"  # type=str, choices=['L2','L1', 'smooth_L1'], help='enhance_loss_type'
        self.enhance_opt_type = "gan_fbank"  # type=str, choices=['gan_fft','gan_fbank'], help='enhance_opt_type'
        self.enhance_dropout_rate = 0.0  # type=float, help='enhance_dropout_rate'
        self.enhance_input_nc = 1  # type=int, help='enhance_input_nc'
        self.enhance_output_nc = 1  # type=int, help='enhance_output_nc'
        self.enhance_ngf = 64  # type=int, help='enhance_ngf'
        self.enhance_norm = "batch"  # type=str, help='enhance_norm'
        self.L1_loss_lambda = 1.0  ##  type=float, help='L1_loss_lambda'
        self.num_saved_specgram = 3
        self.exp_path = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/decode_asr_train_table3_1/decode_unmatch"
        self.gpu_ids = "0"
        self.print_freq = 100
        self.name = "decode_asr_train_table3_1/decode_unmatch"
        self.resume =True
        self.rnnlm = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/rnnlm_train_shi/rnnlm.model.best"
        self.word_rnnlm = None

        self.recog_dir = ''
        self.enhance_dir = ''
        self.enhance_out_dir = ''
        self.recog_label = ''
        self.recog_json = ''
        self.result_label = '/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/decode_asr_train_table3_1/decode_unmatch/data.json'
        self.nj = 8

        # search related
        self.nbest = 5
        self.beam_size = 5
        self.penalty = 0.0
        self.maxlenratio = 0.0
        self.minlenratio =0.0
        self.ctc_weight = 0.1
        self.fstlm_path =''
        self.nn_char_map_file = ''

class Enhance_gan_train(Enhance_fbank_train):
    def __init__(self):
        super(Enhance_gan_train, self).__init__()
        self.ngpu = 1
        self.dataroot = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        self.dict_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        #self.feat_type = "fft" #"kaldi_magspec"
        self.feat_type = "kaldi_magspec"  # "kaldi_magspec"
        self.enhance_resume = False
        #self.enhance_resume = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/enhance_gan_train_both_enhance_cmvn/model.loss.best"
        self.enhance_type = "blstm"
        # choices=['blstm','unet_128', 'unet_256', 'blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm']
        self.enhance_layers = 3  # type=int, help='Number of enhance model layers'
        self.enhance_units = 128  # type=int, help='Number of enhance model hidden units'
        self.enhance_projs = 128  # type=int, help='Number of enhance model projection units'
        self.enhance_nonlinear_type = "sigmoid"
        # type=str, choices=['sigmoid','relu', 'relu6', 'softplus'], help='enhance_nonlinear_type'
        self.enhance_loss_type = "L2"  # type=str, choices=['L2','L1', 'smooth_L1'], help='enhance_loss_type'
        self.enhance_opt_type = "gan_fbank"  # type=str, choices=['gan_fft','gan_fbank'], help='enhance_opt_type'
        self.enhance_dropout_rate = 0.0  # type=float, help='enhance_dropout_rate'
        self.enhance_input_nc = 1  # type=int, help='enhance_input_nc'
        self.enhance_output_nc = 1  # type=int, help='enhance_output_nc'
        self.enhance_ngf = 64  # type=int, help='enhance_ngf'
        self.enhance_norm = "batch"  # type=str, help='enhance_norm'
        self.L1_loss_lambda = 1.0  ##  type=float, help='L1_loss_lambda'
        self.num_saved_specgram = 3
        self.exp_path = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/enhance_gan_train_change_param"
        self.gpu_ids = "0"
        self.print_freq = 500
        self.name = "enhance_gan_train_change_param"
        self.gan_loss_lambda = 2.0
        self.netD_type = 'n_layers'
        self.input_nc = 1
        self.ndf = 32
        self.norm_D = 'batch'
        self.n_layers_D = 3
        self.no_lsgan = False
        self.batch_size = 50


class JointTrain(Enhance_gan_train):
    def __init__(self):
        super(JointTrain, self).__init__()
        self.ngpu = 1
        self.dataroot = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        self.dict_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data"
        #self.feat_type = "fft" #"kaldi_magspec"
        self.feat_type = "kaldi_magspec"  # "kaldi_magspec"
        #self.enhance_resume = False
        #self.enhance_resume = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/enhance_gan_train_both_enhance_cmvn/model.loss.best"
        self.enhance_resume = True
        self.enhance_resume = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/enhance_fbank_train_table_2/model.loss.best"
        self.asr_resume = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/asr_mix_train_table3_1"
        self.enhance_type = "blstm"
        # choices=['blstm','unet_128', 'unet_256', 'blstmp','cnnblstmp','cnnblstm', 'vggblstmp','vggblstm']
        self.enhance_layers = 3  # type=int, help='Number of enhance model layers'
        self.enhance_units = 128  # type=int, help='Number of enhance model hidden units'
        self.enhance_projs = 128  # type=int, help='Number of enhance model projection units'
        self.enhance_nonlinear_type = "sigmoid"
        # type=str, choices=['sigmoid','relu', 'relu6', 'softplus'], help='enhance_nonlinear_type'
        self.enhance_loss_type = "L2"  # type=str, choices=['L2','L1', 'smooth_L1'], help='enhance_loss_type'
        self.enhance_opt_type = "gan_fbank"  # type=str, choices=['gan_fft','gan_fbank'], help='enhance_opt_type'
        self.enhance_dropout_rate = 0.0  # type=float, help='enhance_dropout_rate'
        self.enhance_input_nc = 1  # type=int, help='enhance_input_nc'
        self.enhance_output_nc = 1  # type=int, help='enhance_output_nc'
        self.enhance_ngf = 64  # type=int, help='enhance_ngf'
        self.enhance_norm = "batch"  # type=str, help='enhance_norm'
        self.L1_loss_lambda = 1.0  ##  type=float, help='L1_loss_lambda'
        self.num_saved_specgram = 3
        self.exp_path = "/usr/home/wudamu/Documents/Robust_e2e_gan-master/checkpoints/joint_enhance_E2E_ASR (without GAN)_table_3_2"
        self.gpu_ids = "0"
        self.print_freq = 100
        self.name = "joint_enhance_E2E_ASR (without GAN)_table_3_2"
        self.gan_loss_lambda = 2.0
        self.netD_type = 'n_layers'
        self.input_nc = 1
        self.ndf = 32
        self.norm_D = 'batch'
        self.n_layers_D = 3
        self.no_lsgan = False
        self.isGAN = False
        self.joint_resume = False
        self.gpu_ids = "0"
        self.enhance_loss_lambda = 5.0


