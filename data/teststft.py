from scipy import signal
import numpy as np
# 产生一个测试信号，振幅为2的正弦波，其频率在3kHZ缓慢调制，振幅以指数形式下降的白噪声
import librosa
import scipy.signal
import scipy.io.wavfile as wav
def load_audio(path):
    rate, x = wav.read(path)
    #x, sr = librosa.load(path, sr=8000)
    #ppp = type(x)
    return x

path = '/usr/home/wudamu/FP/Download/data_aishell/wav/S0131/train/S0131/BAC009S0131W0195.wav'
clean = load_audio(path)
# 计算并绘制STFT的大小
array =clean.astype(np.float)
D = librosa.stft(array, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
spect = np.abs(D)
angle = np.angle(D)