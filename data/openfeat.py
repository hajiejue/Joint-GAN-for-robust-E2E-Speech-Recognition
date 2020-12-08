import numpy as np
import os
import scipy.io.wavfile
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
rate = 16000
with_noise = np.load("with_noise.npy")
no_noise = np.load("no_noise.npy")
with_noise = librosa.stft(with_noise, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
no_noise = librosa.stft(no_noise, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)


X = no_noise
Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=44100, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()
spect_with_noise = np.abs(with_noise)
angle_with_noise = np.angle(with_noise)
feat_mat_with_noise = spect_with_noise * np.cos(angle_with_noise) + 1j*spect_with_noise*np.sin(angle_with_noise)


X = with_noise
Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=44100, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()


spect_with_no = np.abs(no_noise)
angle_with_no = np.angle(no_noise)
feat_mat_no = spect_with_no * np.cos(angle_with_no) + 1j*spect_with_no*np.sin(angle_with_no)


stft_rewav_with_noise = librosa.core.istft(feat_mat_with_noise,hop_length=256, win_length=512, window=scipy.signal.hamming)
stft_rewav_with_no = librosa.core.istft(feat_mat_no,hop_length=256, win_length=512, window=scipy.signal.hamming)
scipy.io.wavfile.write('stft_rewav_with_noise.wav', rate, stft_rewav_with_noise)
scipy.io.wavfile.write('stft_rewav_no.wav', rate, stft_rewav_with_no)
