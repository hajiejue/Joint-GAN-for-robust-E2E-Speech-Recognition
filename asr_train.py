from __future__ import print_function
import argparse
import os
import math
import random
import shutil
#import psutil
import time 
import itertools
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import fake_opt
from options.train_options import TrainOptions
from model.e2e_model import E2E
from model.feat_model import FbankModel
from data.data_loader import SequentialDataset, SequentialDataLoader
#from data.data_sampler import BucketingSampler, DistributedBucketingSampler
from data.data_loader import BucketingSampler
from utils.visualizer import Visualizer 
from utils import utils 
    
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed) 
                                                     
def main():
    
    opt = fake_opt.Asr_train()
    exp_path = os.path.join(opt.checkpoints_dir, opt.name)
    utils.mkdirs(exp_path)
    opt.exp_path = exp_path
#    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    #cuda_ava = torch.cuda.is_available()
    visualizer = Visualizer(opt)  
    logging = visualizer.get_logger()
    acc_report = visualizer.add_plot_report(['train/acc', 'val/acc'], 'acc.png')
    loss_report = visualizer.add_plot_report(['train/loss', 'val/loss'], 'loss.png')
     
    # data
    logging.info("Building dataset.")
    train_dataset = SequentialDataset(opt, os.path.join(opt.dataroot, 'train_new'), os.path.join(opt.dict_dir, 'train/vocab'),)
    val_dataset = SequentialDataset(opt, os.path.join(opt.dataroot, 'dev_new'), os.path.join(opt.dict_dir, 'train/vocab'),)
    train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size)
    train_loader = SequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader = SequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    opt.idim = train_dataset.get_feat_size() #257
    opt.odim = train_dataset.get_num_classes() #4233
    opt.char_list = train_dataset.get_char_list()
    opt.train_dataset_len = len(train_dataset)
    logging.info('#input dims : ' + str(opt.idim))
    logging.info('#output dims: ' + str(opt.odim))
    logging.info("Dataset ready!")
    
    # Setup a model
    asr_model = E2E(opt)
    fbank_model = FbankModel(opt)
    lr = opt.lr
    eps = opt.eps
    iters = opt.iters
    start_epoch = opt.start_epoch    
    best_loss = opt.best_loss
    best_acc = opt.best_acc
    if opt.resume:
        model_path = os.path.join(opt.works_dir, opt.resume)
        if os.path.isfile(model_path):
            package = torch.load(model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)        
            best_loss = package.get('best_loss', float('inf'))
            best_acc = package.get('best_acc', 0)
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0))
            
            acc_report = package.get('acc_report', acc_report)
            loss_report = package.get('loss_report', loss_report)
            visualizer.set_plot_report(acc_report, 'acc.png')
            visualizer.set_plot_report(loss_report, 'loss.png')
            
            asr_model = E2E.load_model(model_path, 'asr_state_dict') 
            fbank_model = FbankModel.load_model(model_path, 'fbank_state_dict')
            logging.info('Loading model {} and iters {}'.format(model_path, iters))
        else:
            print("no checkpoint found at {}".format(model_path))                
    asr_model.cuda()
    fbank_model.cuda()
    print(asr_model)
  
    # Setup an optimizer
    parameters = filter(lambda p: p.requires_grad, itertools.chain(asr_model.parameters(), fbank_model.parameters()))
    #parameters = filter(lambda p: p.requires_grad, itertools.chain(asr_model.parameters()))
    if opt.opt_type == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, rho=0.95, eps=eps)
    elif opt.opt_type == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(opt.beta1, 0.999))                       
           
    asr_model.train()
    fbank_model.train()
    sample_rampup = utils.ScheSampleRampup(opt.sche_samp_start_iter, opt.sche_samp_final_iter, opt.sche_samp_final_rate)  
    sche_samp_rate = sample_rampup.update(iters)


    fbank_cmvn_file = os.path.join(opt.exp_path, 'fbank_cmvn.npy')
    #fbank_cmvn_file = os.path.join(opt.exp_path, 'cmvn.npy')
    if os.path.exists(fbank_cmvn_file):
        fbank_cmvn = np.load(fbank_cmvn_file)
    else:
        for i, (data) in enumerate(train_loader, start=0):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            fbank_cmvn = fbank_model.compute_cmvn(inputs, input_sizes)
            #fbank_cmvn = FbankModel.compute_cmvn(inputs, input_sizes)
            #if fbank_cmvn is not None:
            if fbank_model.cmvn_processed_num >= fbank_model.cmvn_num:
                fbank_cmvn = fbank_model.compute_cmvn(inputs, input_sizes)
                np.save(fbank_cmvn_file, fbank_cmvn)
                print('save fbank_cmvn to {}'.format(fbank_cmvn_file))
                break
    fbank_cmvn = torch.FloatTensor(fbank_cmvn)

    for epoch in range(start_epoch, opt.epochs):
        if epoch > opt.shuffle_epoch:
            print("Shuffling batches for the following epochs")
            train_sampler.shuffle(epoch)                    
        for i, (data) in enumerate(train_loader, start=(iters*opt.batch_size)%len(train_dataset)):
            utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
            fbank_features = fbank_model(inputs, fbank_cmvn)

            #utt_ids, spk_ids, fbank_features, targets, input_sizes, target_sizes = data
            #loss_ctc, loss_att, acc, context = asr_model(fbank_features, targets, input_sizes, target_sizes, sche_samp_rate)
            loss_ctc, loss_att, acc = asr_model(fbank_features, targets, input_sizes, target_sizes,sche_samp_rate)
            loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att
            optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()          
            # compute the gradient norm to check if it is normal or not 'fbank_state_dict': fbank_model.state_dict(), 
            grad_norm = torch.nn.utils.clip_grad_norm_(asr_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.step()

            iters += 1
            errors = {'train/loss': loss.item(), 'train/loss_ctc': loss_ctc.item(), 
                      'train/acc': acc, 'train/loss_att': loss_att.item()}
            visualizer.set_current_errors(errors)
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {'asr_state_dict': asr_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr,                                    
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                filename='latest'
                utils.save_checkpoint(state, opt.exp_path, filename=filename)
                    
            if iters % opt.validate_freq == 0:
                sche_samp_rate = sample_rampup.update(iters)
                print("iters {} sche_samp_rate {}".format(iters, sche_samp_rate))  
                asr_model.eval()
                fbank_model.eval()
                torch.set_grad_enabled(False)
                num_saved_attention = 0
                for i, (data) in tqdm(enumerate(val_loader, start=0)):
                    utt_ids, spk_ids, inputs, log_inputs, targets, input_sizes, target_sizes = data
                    fbank_features = fbank_model(inputs, fbank_cmvn)
                    #utt_ids, spk_ids, fbank_features, targets, input_sizes, target_sizes = data
                    #loss_ctc, loss_att, acc, context = asr_model(fbank_features, targets, input_sizes, target_sizes, 0.0)
                    loss_ctc, loss_att, acc = asr_model(fbank_features, targets, input_sizes, target_sizes, 0.0)

                    loss = opt.mtlalpha * loss_ctc + (1 - opt.mtlalpha) * loss_att                            
                    errors = {'val/loss': loss.item(), 'val/loss_ctc': loss_ctc.item(), 
                              'val/acc': acc, 'val/loss_att': loss_att.item()}
                    visualizer.set_current_errors(errors)
                    
                    if opt.num_save_attention > 0 and opt.mtlalpha != 1.0:
                        if num_saved_attention < opt.num_save_attention:
                            att_ws = asr_model.calculate_all_attentions(fbank_features, targets, input_sizes, target_sizes)                            
                            for x in range(len(utt_ids)):
                                att_w = att_ws[x]
                                utt_id = utt_ids[x]
                                file_name = "{}_ep{}_it{}.png".format(utt_id, epoch, iters)
                                dec_len = int(target_sizes[x])
                                enc_len = int(input_sizes[x]) 
                                visualizer.plot_attention(att_w, dec_len, enc_len, file_name) 
                                num_saved_attention += 1
                                if num_saved_attention >= opt.num_save_attention:   
                                    break                                                       
                asr_model.train()
                fbank_model.train()
                torch.set_grad_enabled(True)

                visualizer.print_epoch_errors(epoch, iters)  
                acc_report = visualizer.plot_epoch_errors(epoch, iters, 'acc.png') 
                loss_report = visualizer.plot_epoch_errors(epoch, iters, 'loss.png') 
                val_loss = visualizer.get_current_errors('val/loss')
                val_acc = visualizer.get_current_errors('val/acc')  
                filename = None              
                if opt.criterion == 'acc' and opt.mtl_mode != 'ctc':
                    if val_acc < best_acc:
                        logging.info('val_acc {} > best_acc {}'.format(val_acc, best_acc))
                        opt.eps = utils.adadelta_eps_decay(optimizer, opt.eps_decay)
                    else:
                        filename='model.acc.best'                    
                    best_acc = max(best_acc, val_acc)
                    logging.info('best_acc {}'.format(best_acc))  
                elif opt.criterion == 'loss':
                #elif args.criterion == 'loss':
                    if val_loss > best_loss:
                        logging.info('val_loss {} > best_loss {}'.format(val_loss, best_loss))
                        opt.eps = utils.adadelta_eps_decay(optimizer, opt.eps_decay)
                    else:
                        filename='model.loss.best'    
                    best_loss = min(val_loss, best_loss)
                    logging.info('best_loss {}'.format(best_loss))                  
                state = {'asr_state_dict': asr_model.state_dict(), 
                         'opt': opt, 'epoch': epoch, 'iters': iters, 
                         'eps': opt.eps, 'lr': opt.lr,                                    
                         'best_loss': best_loss, 'best_acc': best_acc, 
                         'acc_report': acc_report, 'loss_report': loss_report}
                utils.save_checkpoint(state, opt.exp_path, filename=filename)
                ##filename='epoch-{}_iters-{}_loss-{:.4f}_acc-{:.4f}.pth'.format(epoch, iters, val_loss, val_acc)
                ##utils.save_checkpoint(state, opt.exp_path, filename=filename)                  
                visualizer.reset()

        
          
if __name__ == '__main__':
    main()
