import librosa
import numpy as np
import skimage.measure


def mel_spectrogram(sr_desired, name_song, window_size, hop_length):
    "This function calculates the mel spectrogram in dB with Librosa library"
    y, sr = librosa.load(name_song, sr=None)
    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=window_size,
        hop_length=hop_length,
        n_mels=80,
        fmin=80,
        fmax=16000,
    )
    S_to_dB = librosa.power_to_db(S, ref=np.max)  # convert S in dB
    return S_to_dB  # S_to_dB is the spectrogam in dB


def max_pooling(spectrogram_padded, pooling_factor):
    x_prime = skimage.measure.block_reduce(
        spectrogram_padded, (1, pooling_factor), np.max
    )
    return x_prime


song = "t2"
song_path = "/home/richie/Desktop/test_songs/" + song + ".mp3"

window_size = 2048  # (samples/frame)
hop_length = 1024  # overlap 50% (samples/frame)
sr_desired = 44100

mel = mel_spectrogram(sr_desired, song_path, window_size, hop_length)
mel_max_pooling = max_pooling(mel, 6)

np.save("/home/richie/Desktop/sslm/" + song + "_mls.npy", mel_max_pooling)
