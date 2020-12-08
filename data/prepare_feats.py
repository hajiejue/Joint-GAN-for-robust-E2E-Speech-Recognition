import os
import sys
from numpy.core.fromnumeric import argmax
from data import kaldi_io
import librosa
import numpy as np
import scipy.signal
import torchaudio
import math
import random
from random import choice
from multiprocessing import Pool

import scipy.io.wavfile as wav

import pdb


def load_audio(path):
    sound, _ = torchaudio.load_wav(path, normalization=False)
    sound = sound.numpy()

    if len(sound.shape) > 1:
        if sound.shape[0] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=0)  # multiple channels, average
    sound = sound / (2 ** 15)

    return sound
# def load_audio(path):
#     path = os.path.join(root_path,path)
#     print(path)
#     rate, sig = wav.read(path)
#     sig = sig.astype(np.float32)
#     return sig
def MakeMixture(speech, noise, db):

    if speech is None or noise is None:
        return None
    if np.sum(np.square(noise)) < 1.0e-6:
        return None
    spelen = speech.shape[0]
    exnoise = noise
    #print(" I am mixfeature")
    while exnoise.shape[0] < spelen:

        exnoise = np.concatenate([exnoise, noise], 0)

    noise = exnoise
    noilen = noise.shape[0]
    elen = noilen - spelen - 1

    if elen > 1:

        s = round(random.randint(0, elen - 1))
    else:
        s = 0
    e = s + spelen
    noise = noise[s:e]
    try:
        noi_pow = np.sum(np.square(noise))
        if noi_pow > 0:
            noi_scale = math.sqrt(np.sum(np.square(speech)) / (noi_pow * (10 ** (db / 10.0))))
        else:
            print(noi_pow, np.square(noise), "error")
            return None
    except:
        return None
    nnoise = noise * noi_scale
    print("nnoise:",nnoise)
    mixture = speech + nnoise
    mixture = mixture.astype("float32")
    return mixture
def make_feature(wav_path_list, noise_wav_list, feat_dir, thread_num, argument=False, repeat_num=1):
    mag_ark_scp_output = "ark:| copy-feats --compress=true ark:- ark,scp:{0}/feats{1}.ark,{0}/feats{1}.scp".format(feat_dir, thread_num)
    ang_ark_scp_output = "ark:| copy-feats --compress=true ark:- ark,scp:{0}/angles{1}.ark,{0}/angles{1}.scp".format(feat_dir, thread_num)
    print("I am  make_featrue")
    if argument:

        fwrite = open(os.path.join(feat_dir, "db" + str(thread_num)), "a")
    f_mag = kaldi_io.open_or_fd(mag_ark_scp_output, "wb")
    f_ang = kaldi_io.open_or_fd(ang_ark_scp_output, "wb")
    for num in range(repeat_num):
        for tmp in wav_path_list:
            uttid, wav_path = tmp
            clean = load_audio(wav_path)
            print("i in line 88")
            y = None
            while y is None:
                if argument:
                    noise_path = choice(noise_wav_list)
                    n = load_audio(noise_path[1])
                    db = np.random.uniform(low=0, high=20)
                    y = MakeMixture(clean, n, db)
                    uttid_new = uttid + "__mix{}".format(num)
                    # print(uttid_new + " " + str(db) + "\n")
                    fwrite.write(uttid_new + " " + str(db) + "\n")
                    print("i in line 99")
                else:
                    y = clean
                    uttid_new = uttid
                    print("i in line 103")
            # STFT
            if y is not None:
                D = librosa.stft(y, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
                spect = np.abs(D)
                angle = np.angle(D)
                print("I in line 109")
                print("spect",spect)
                kaldi_io.write_mat(f_mag, spect.transpose((1, 0)), key=uttid_new)
                kaldi_io.write_mat(f_ang, angle.transpose((1, 0)), key=uttid_new)
            else:
                print(noise_path, tmp, "error")
    f_mag.close()
    f_ang.close()
    if argument:
        fwrite.close()
def main():
    # 输入参数
    data_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/test"
    feat_dir = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/test"
    noise_repeat_num = 1
    # 建立clean数据的路径
    clean_feat_dir = os.path.join(feat_dir, "clean_for_test")
    if not os.path.exists(clean_feat_dir):
        os.makedirs(clean_feat_dir)
    # 建立mix数据的路径
    mix_feat_dir = os.path.join(feat_dir, "mix_for_test")
    if not os.path.exists(mix_feat_dir):
        os.makedirs(mix_feat_dir)
    clean_wav_list = []
    clean_wav_scp = os.path.join(data_dir, "wav_for_test.scp")
    with open(clean_wav_scp, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip().replace("\n", "")
            uttid, wav_path = line.split()
            clean_wav_list.append((uttid, wav_path))

    print("clean_wav_list", len(clean_wav_list))
    noise_wav_list = []
    #noise_wav_scp = os.path.join("data/noise", "noise.scp")
    noise_wav_scp = "/usr/home/wudamu/FP/kaldi-trunk/egs/aishell/s5/data/noise/noise.scp"
    with open(noise_wav_scp, "r", encoding="utf-8") as fid:
        for line in fid:
            line = line.strip().replace("\n", "")
            wav_path = line.split()
            noise_wav_list.append(wav_path)
    print("noise_wav_list", len(noise_wav_list))
    print(noise_wav_list)
    # 使用八个线程
    threads_num = 4

    wav_num = len(clean_wav_list)
    print("Parent process %s." % os.getpid())
    p = Pool()
    start = 0
    for i in range(threads_num):
        end = start + int(wav_num / threads_num)
        if i == (threads_num - 1):
            end = wav_num
        wav_path_tmp_list = clean_wav_list[start:end]
        start = end
        p.apply_async(make_feature, args=(wav_path_tmp_list, noise_wav_list, clean_feat_dir, i, False))
        #make_feature(clean_wav_list, noise_wav_list, clean_feat_dir, i, False)
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")
    command_line = "cat {}/feats*.scp > {}/clean_feats.scp".format(clean_feat_dir, data_dir)
    os.system(command_line)
    command_line = "cat {}/angles*.scp > {}/clean_angles.scp".format(clean_feat_dir, data_dir)
    os.system(command_line)
    wav_num = len(clean_wav_list)
    print("Parent process %s." % os.getpid())
    p = Pool()
    for i in range(threads_num):
        wav_path_tmp_list = clean_wav_list[int(i * wav_num / threads_num) : int((i + 1) * wav_num / threads_num)]
        p.apply_async(make_feature, args=(wav_path_tmp_list, noise_wav_list, mix_feat_dir, i, True, noise_repeat_num))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")
    command_line = "cat {}/feats*.scp > {}/mix_feats.scp".format(mix_feat_dir, data_dir)
    os.system(command_line)
    command_line = "cat {}/angles*.scp > {}/mix_angles.scp".format(mix_feat_dir, data_dir)
    os.system(command_line)
    command_line = "cat {}/db* > {}/db.scp".format(mix_feat_dir, data_dir)
    os.system(command_line)
if __name__ == "__main__":

    main()
