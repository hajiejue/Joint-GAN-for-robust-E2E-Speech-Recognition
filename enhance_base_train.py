from __future__ import print_function
import argparse
import os
import math
import random
import shutil
#import psutil
import time 
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import fake_opt
from options.train_options import TrainOptions
from model.enhance_model import EnhanceModel
from data.mix_data_loader import MixSequentialDataset, MixSequentialDataLoader, BucketingSampler
from utils.visualizer import Visualizer 
from utils import utils 

manualSeed = random.randint(1, 10000)   #生成随机数
random.seed(manualSeed)                 #规定生成随机数的的值
torch.manual_seed(manualSeed)           #规定生成随机数的的值
torch.cuda.manual_seed(manualSeed)      #规定生成随机数的的值
 
def main():
    opt=fake_opt.Enhance_base_train()
    #opt = TrainOptions().parse()        #从train_option 倒入数据
    device = torch.device("cuda:{}".format(opt.gpu_ids[0]) if len(opt.gpu_ids) > 0 and torch.cuda.is_available() else "cpu")  #选择是cpu还是gpu
 
    visualizer = Visualizer(opt)        #可视化工具
    logging = visualizer.get_logger()
    loss_report = visualizer.add_plot_report(['train/loss', 'val/loss'], 'loss.png')

    # data
    logging.info("Building dataset.")
    train_dataset = MixSequentialDataset(opt, os.path.join(opt.dataroot, 'train'), os.path.join(opt.dict_dir, 'train/vocab'),)
    val_dataset   = MixSequentialDataset(opt, os.path.join(opt.dataroot, 'dev'), os.path.join(opt.dict_dir, 'train/vocab'),)
    train_sampler = BucketingSampler(train_dataset, batch_size=opt.batch_size) 
    train_loader = MixSequentialDataLoader(train_dataset, num_workers=opt.num_workers, batch_sampler=train_sampler)
    val_loader = MixSequentialDataLoader(val_dataset, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False)
    opt.idim = train_dataset.get_feat_size()
    opt.odim = train_dataset.get_num_classes()
    opt.char_list = train_dataset.get_char_list()
    opt.train_dataset_len = len(train_dataset)
    logging.info('#input dims : ' + str(opt.idim))
    logging.info('#output dims: ' + str(opt.odim))
    logging.info("Dataset ready!")
  	
    lr = opt.lr
    eps = opt.eps
    iters = opt.iters    
    best_loss = opt.best_loss  
    start_epoch = opt.start_epoch      
    model_path = None
    if opt.enhance_resume:
        model_path = os.path.join(opt.works_dir, opt.enhace_resume)
        if os.path.isfile(model_path):
            package = torch.load(model_path, map_location=lambda storage, loc: storage)
            lr = package.get('lr', opt.lr)
            eps = package.get('eps', opt.eps)        
            best_loss = package.get('best_loss', float('inf'))
            start_epoch = int(package.get('epoch', 0))   
            iters = int(package.get('iters', 0))            
            loss_report = package.get('loss_report', loss_report)
            visualizer.set_plot_report(loss_report, 'loss.png')
            print('package found at {} and start_epoch {} iters {}',format(model_path, start_epoch, iters))
        else:
            print("no checkpoint found at {}".format(model_path))     
    enhance_model = EnhanceModel.load_model(model_path, 'enhance_state_dict', opt)
                          
	# Setup an optimizer
    enhance_parameters = filter(lambda p: p.requires_grad, enhance_model.parameters())    
    if opt.opt_type == 'adadelta':
        enhance_optimizer = torch.optim.Adadelta(enhance_parameters, rho=0.95, eps=opt.eps)
    elif opt.opt_type == 'adam':
        enhance_optimizer = torch.optim.Adam(enhance_parameters, lr=opt.lr, betas=(opt.beta1, 0.999)) 
            
    # Training		                    
    for epoch in range(start_epoch, opt.epochs):
        enhance_model.train()
        if epoch > opt.shuffle_epoch:
            print("Shuffling batches for the following epochs")
            train_sampler.shuffle(epoch)   
        for i, (data) in enumerate(train_loader, start=(iters % len(train_dataset))):
            utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes = data
            loss, enhance_out = enhance_model(clean_inputs=clean_inputs, mix_inputs=mix_inputs, mix_log_inputs=mix_log_inputs, cos_angles=cos_angles, input_sizes=input_sizes)
            enhance_optimizer.zero_grad()  # Clear the parameter gradients
            loss.backward()          
            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(enhance_model.parameters(), opt.grad_clip)
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                enhance_optimizer.step()
                
            iters += 1
            errors = {'train/loss': loss.item()}
            visualizer.set_current_errors(errors)
            if iters % opt.print_freq == 0:
                visualizer.print_current_errors(epoch, iters)
                state = {'enhance_state_dict': enhance_model.state_dict(), 'opt': opt,                                             
                         'epoch': epoch, 'iters': iters, 'eps': eps, 'lr': lr,                                    
                         'best_loss': best_loss, 'loss_report': loss_report}
                filename='latest'
                utils.save_checkpoint(state, opt.exp_path, filename=filename)
                    
            if iters % opt.validate_freq == 0:
                enhance_model.eval()
                torch.set_grad_enabled(False)
                num_saved_specgram = 0
                for i, (data) in tqdm(enumerate(val_loader, start=0)):
                    utt_ids, spk_ids, clean_inputs, clean_log_inputs, mix_inputs, mix_log_inputs, cos_angles, targets, input_sizes, target_sizes = data
                    #loss, enhance_out = enhance_model(clean_inputs, mix_inputs, mix_log_inputs, cos_angles, input_sizes)
                    loss, enhance_out = enhance_model(clean_inputs=clean_inputs, mix_inputs=mix_inputs, mix_log_inputs=mix_log_inputs,cos_angles=cos_angles, input_sizes=input_sizes)

                    errors = {'val/loss': loss.item()}
                    visualizer.set_current_errors(errors)
                
                    if opt.num_saved_specgram > 0:
                        if num_saved_specgram < opt.num_saved_specgram:
                            enhanced_outs = enhance_model.calculate_all_specgram(mix_inputs, mix_log_inputs, input_sizes)
                            print(num_saved_specgram)
                            for x in range(len(utt_ids)):
                                enhanced_out = enhanced_outs[x].data.cpu().numpy()
                                enhanced_out[enhanced_out <= 1e-7] = 1e-7
                                enhanced_out = np.log10(enhanced_out)
                                clean_input = clean_inputs[x].data.cpu().numpy()
                                clean_input[clean_input <= 1e-7] = 1e-7
                                clean_input = np.log10(clean_input)
                                mix_input = mix_inputs[x].data.cpu().numpy()
                                mix_input[mix_input <= 1e-7] = 1e-7
                                mix_input = np.log10(mix_input)
                                utt_id = utt_ids[x]
                                file_name = "{}_ep{}_it{}.png".format(utt_id, epoch, iters)
                                input_size = int(input_sizes[x])
                                visualizer.plot_specgram(clean_input, mix_input, enhanced_out, input_size, file_name)
                                num_saved_specgram += 1
                                if num_saved_specgram >= opt.num_saved_specgram:
                                    break                                                                                    
                enhance_model.train()
                torch.set_grad_enabled(True)  
				
                visualizer.print_epoch_errors(epoch, iters)               
                loss_report = visualizer.plot_epoch_errors(epoch, iters, 'loss.png')                     
                train_loss = visualizer.get_current_errors('train/loss')
                val_loss = visualizer.get_current_errors('val/loss')
                filename = None
                if val_loss > best_loss:
                    print('val_loss {} > best_loss {}'.format(val_loss, best_loss))
                    eps = utils.adadelta_eps_decay(enhance_optimizer, opt.eps_decay)
                else:
                    filename='model.loss.best'                                
                best_loss = min(val_loss, best_loss)
                print('best_loss {}'.format(best_loss))  
                
                state = {'enhance_state_dict': enhance_model.state_dict(), 'opt': opt,                                             
                         'epoch': epoch, 'iters': iters, 'eps': eps, 'lr': lr,                                    
                         'best_loss': best_loss, 'loss_report': loss_report}
                ##filename='epoch-{}_iters-{}_loss-{:.6f}-{:.6f}.pth'.format(epoch, iters, train_loss, val_loss)
                utils.save_checkpoint(state, opt.exp_path, filename=filename)
                visualizer.reset()      
      
if __name__ == '__main__':
    main()
