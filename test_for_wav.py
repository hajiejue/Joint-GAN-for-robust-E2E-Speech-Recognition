import numpy as np
import librosa
import torch
import torchaudio
import os
import scipy.io.wavfile
def rewav(path,uttid,feats,ang,hop_length = 256,win_length = 512,window = scipy.signal.hamming,rate = 16000):
    if isinstance(feats,torch.Tensor):
        feats_numpy = feats.numpy()
        ang_numpy = ang.numpy()
    else:
        feats_numpy = feats
        ang_numpy = ang
    feat_mat = feats_numpy*np.cos(ang_numpy)+1j*feats_numpy*np.sin(ang_numpy)
    #feat_mat = feat_mat[feat_mat]
    #feat_mat = feat_mat.T
    x = librosa.core.istft(feat_mat,hop_length = hop_length,win_length = win_length,window = window)
    #x = x*65535
    path_wav = os.path.join(path,"remix_"+uttid+".wav")
    scipy.io.wavfile.write(path_wav,rate,data = x)
def load_audio(path):
    sound, _ = torchaudio.load(path, normalization=True)
    sound = sound.numpy()
    if sound.shape[0] == 1:
        sound = sound.squeeze()
    else:
        sound = sound.mean(axis=0)  # multiple channels, average
    #sound = sound / 65536.
    return sound
path = "/usr/home/wudamu/FP/Download/data_aishell/wav/train/S0002/BAC009S0002W0122.wav"
path_sav = "/usr/home/shi/projects/data_aishell/data/wavfile"
audio = load_audio(path)
D = librosa.stft(audio, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
spect = np.abs(D)
angle = np.angle(D)
rewav(path_sav,'11000',spect,angle)