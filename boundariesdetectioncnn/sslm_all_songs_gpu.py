import librosa
import librosa.display
import numpy as np
import scipy
from scipy.spatial import distance
import os
import pathlib
from multiprocessing import Pool
import cupy as cp
import cucim.skimage.measure
import torch
import torchaudio
from nnAudio import features



# window_size = 2048  # (samples/frame)
# hop_length = 1024  # overlap 50% (samples/frame)
# sr_desired = 44100
# p1 = 2  # pooling factor
# p2 = 3
# pm = 6
# L_sec_near = 14  # lag near context in seconds
# L_near = round(
#     L_sec_near * sr_desired / hop_length
# )  # conversion of lag L seconds to frames

audio_path = "/home/richie/Desktop/test_songs/"

im_path_base = "/home/richie/Desktop/sslm/"
im_path_mel = os.path.join(im_path_base, "mls/")
im_path_SSLM_MFCCs_euclidean = os.path.join(im_path_base, "SSLM_MFCCs_euclidean/")
im_path_SSLM_MFCCs_cosine = os.path.join(im_path_base, "SSLM_MFCCs_cosine/")
im_path_SSLM_Chromas_euclidean = os.path.join(im_path_base, "SSLM_Chromas_euclidean/")
im_path_SSLM_Chromas_cosine = os.path.join(im_path_base, "SSLM_Chromas_cosine/")


def split_filename(filename, full_name=True):
    path = pathlib.Path(filename)
    if full_name:
        return path.name, path.suffix
    return path.stem, path.suffix


def load_audio(input_file, sr_desired):
    "This function loads the audio file and resamples it to sr_desired"
    waveform, sample_rate = torchaudio.load(input_file)
    if sample_rate != sr_desired:
        transform = torchaudio.transforms.Resample(sample_rate, sr_desired)
        waveform = transform(waveform)
        sample_rate = sr_desired
    # convert to mono
    if waveform.shape[0] == 2:
        # waveform = (waveform[:, 0] + waveform[:, 1]) / 2
        waveform = torch.mean(waveform, dim=0)
    return waveform, sample_rate


def save_data(save_path, input_data):
    # check file exists
    if os.path.exists(save_path):
        print(save_path, " exists")
        return
    cp.save(save_path, input_data)


def lag_near_frames(lag_near_sec=14, sr=44100, hop_length=1024):
    return round(lag_near_sec * sr / hop_length)


def stft_cuda(input_signal, window_size=2048, hop_length=1024):
    # stft_trans = features.STFT(n_fft=window_size, hop_length=hop_length)
    # stft_tensor = stft_trans(input_signal)
    stft_tensor = torch.stft(input=input_signal.cuda(), n_fft=window_size, hop_length=hop_length).cuda()
    stft_cuda = torch.abs(stft_tensor)
    return stft_cuda


def melspectrogram_cuda(input_signal, sr=44100, window_size=2048, hop_length=1024):
    mel_trans = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=window_size, hop_length=hop_length, n_mels=80, fmin=80, fmax=16000)
    mel_cuda = mel_trans(input_signal.cuda()).cuda()
    mel_db_trans = torchaudio.transforms.AmplitudeToDB('power', cp.max)
    mel_db_cuda = mel_db_trans(mel_cuda)

    return mel_db_cuda


def max_pooling(input_signal, pooling_factor):
    output_singal = cucim.skimage.measure.block_reduce(
        input_signal, (1, pooling_factor), cp.max
    )
    return output_singal


def mls(input_signal, sr=44100, window_size=2048, hop_length=1024, pooling_factor=6):
    mel = melspectrogram_cuda(input_signal, sr, window_size, hop_length)
    output_singal = max_pooling(mel, pooling_factor)
    return output_singal


def sslm(
    input_signal,
    sr=44100,
    window_size=2048,
    hop_length=1024,
    first_pooling_factor=2,
    second_pooling_factor=3,
    lag=30,
    feature_type='chromas',
    distance_type="cosine"
):
    """SSLM extraction, including MFCCs and Chromas.

    Args:
        input_signal ([type]): Audio signal
        sr (int, optional): Desired samplerate. Defaults to 44100.
        window_size (int, optional): (samples/frame). Defaults to 2048.
        hop_length (int, optional): overlap 50% (samples/frame). Defaults to 1024.
        first_pooling_factor (int, optional): First layer max pooling factor. Defaults to 2, set to 6 to get a better performance.
        second_pooling_factor (int, optional): Second layer max pooling factor, when first layer set to 6, this should to set to 0. Defaults to 3.
        lag (int, optional): Conversion of lag L seconds to frames. Defaults to 30.
        feature_type (str, optional): Feature type, 'chromas' or 'mfccs'.
        distance_type (str, optional): Distance type, 'cosine' or 'euclidean'.

    Returns:
        Array: SSLM feature
    """
    input_waveform = (
        stft_cuda(input_signal, window_size, hop_length)
        if feature_type == 'chromas' and distance_type == 'cosine'
        else melspectrogram_cuda()(input_signal, sr, window_size, hop_length)
    )

    input_waveform_cp = cp.asarray(input_waveform)
    # input_waveform_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(input_waveform_cuda))

    padding_factor = lag
    """"This part pads a mel spectrogram gived the spectrogram a lag parameter
    to compare the first rows with the last ones and make the matrix circular"""
    pad = cp.full(
        (input_waveform_cp.shape[0], padding_factor), -70
    )  # matrix of 80x30frames of -70dB corresponding to padding
    S_padded = cp.concatenate(
        (pad, input_waveform_cp), axis=1
    )  # padding 30 frames with noise at -70dB at the beginning

    """This part max-poolend the spectrogram in time axis by a factor of p"""
    x_prime = S_padded
    if first_pooling_factor > 1:
        x_prime = max_pooling(S_padded, first_pooling_factor)

    """"This part calculates a circular Self Similarity Lag Matrix given
    the mel spectrogram padded and max-pooled"""
    before_bagging = (
        librosa.feature.chroma_stft(
            S=x_prime, sr=sr, n_fft=window_size, hop_length=hop_length
        )
        if feature_type == 'chromas'
        else scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    )
    before_bagging = before_bagging[1:, :]

    # before_bagging = x_prime
    # if type == 'chromas_cosine':
    #     before_bagging = librosa.feature.chroma_stft(S=x_prime, sr=sr, n_fft=window_size, hop_length=hop_length)
    #     before_bagging = before_bagging[1:, :]
    # elif type == 'mfccs':
    #     #MFCCs calculation from DCT-Type II
    #     before_bagging = scipy.fftpack.dct(x_prime, axis=0, type=2, norm='ortho')
    #     before_bagging = before_bagging[1:,:] #0 componen ommited

    # Bagging frames
    m = 2  # baggin parameter in frames
    x = [cp.roll(before_bagging, n, axis=1) for n in range(m)]
    x_hat = cp.concatenate(x, axis=0)

    # Cosine distance calculation: D[N/p,L/p] matrix
    distances = cp.zeros(
        (x_hat.shape[1], padding_factor // first_pooling_factor)
    )  # D has as dimensions N/p and L/p
    for i in range(x_hat.shape[1]):  # iteration in columns of x_hat
        for l in range(padding_factor // first_pooling_factor):
            if i - (l + 1) < 0:
                cosine_dist = 1
            elif i - (l + 1) < padding_factor // first_pooling_factor:
                cosine_dist = 1
            else:
                if distance_type == 'cosine':
                    cosine_dist = distance.cosine(x_hat[:, i], x_hat[:, i - (l + 1)])
                elif distance_type == 'euclidean':
                    cosine_dist = distance.euclidean(
                        x_hat[:, i], x_hat[:, i - (l + 1)]
                    )  # cosine distance between columns i and i-L
                else:
                    cosine_dist = 0

                if cosine_dist == float('nan'):
                    cosine_dist = 0
            distances[i, l] = cosine_dist

    # Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1
    epsilon = cp.zeros(
        (distances.shape[0], padding_factor // first_pooling_factor)
    )  # D has as dimensions N/p and L/p
    for i in range(
        padding_factor // first_pooling_factor, distances.shape[0]
    ):  # iteration in columns of x_hat
        for l in range(padding_factor // first_pooling_factor):
            epsilon[i, l] = cp.quantile(
                cp.concatenate((distances[i - l, :], distances[i, :])), kappa
            )

    # We remove the padding done before
    distances = distances[padding_factor // first_pooling_factor :, :]
    epsilon = epsilon[padding_factor // first_pooling_factor :, :]
    # x_prime = x_prime[:, padding_factor // first_pooling_factor:]

    # Self Similarity Lag Matrix
    sslm = scipy.special.expit(1 - distances / epsilon)  # aplicación de la sigmoide
    sslm = cp.transpose(sslm)
    if second_pooling_factor > 1:
        sslm = max_pooling(sslm, second_pooling_factor)

    # Check if SSLM has nans and if it has them, substitute them by 0
    sslm[cp.isnan(sslm)] = 0

    return sslm


def mls_sslm_extraction(audio_file, sr_desired=44100):
    if not os.path.exists(audio_file):
        print("audio: " + audio_file + " not exist")
        return

    file_name, file_suffix = split_filename(audio_file, False)
    if file_suffix == ".mp3" or file_suffix == ".wav":
        # check if features already exist
        mls_file_path = os.path.join(im_path_mel, file_name + ".npy")
        sslm_chromas_cosine_file_path = os.path.join(
            im_path_SSLM_Chromas_cosine, file_name + ".npy"
        )
        sslm_chromas_euclidean_file_path = os.path.join(
            im_path_SSLM_Chromas_euclidean, file_name + ".npy"
        )
        sslm_mfccs_cosine_file_path = os.path.join(
            im_path_SSLM_MFCCs_cosine, file_name + ".npy"
        )
        sslm_mfccs_euclidean_file_path = os.path.join(
            im_path_SSLM_MFCCs_euclidean, file_name + ".npy"
        )

        mls_exist = os.path.exists(mls_file_path)
        sslm_chromas_cosine_exist = os.path.exists(sslm_chromas_cosine_file_path)
        sslm_chromas_euclidean_exist = os.path.exists(sslm_chromas_euclidean_file_path)
        sslm_mfccs_cosine_exist = os.path.exists(sslm_mfccs_cosine_file_path)
        sslm_mfccs_euclidean_exist = os.path.exists(sslm_mfccs_euclidean_file_path)

        if (
            mls_exist
            and sslm_chromas_cosine_exist
            and sslm_chromas_euclidean_exist
            and sslm_mfccs_cosine_exist
            and sslm_mfccs_euclidean_exist
        ):
            print("audio: " + audio_file + " all features already processed.")
            return

        print("==========process start audio: ", file_name, " ==========")
        l_frames = lag_near_frames()
        y, sr = load_audio(audio_file, sr_desired)
        # get mls
        if not mls_exist:
            sslm_mls = mls(input_signal=y)
            save_data(os.path.join(im_path_mel, file_name + ".npy"), sslm_mls)
            print("audio: " + audio_file + " mls finished.")

        # get sslm
        if not sslm_chromas_cosine_exist:
            sslm_chromas_cosine = sslm(
                input_signal=y,
                sr=sr,
                lag=l_frames,
                feature_type='chromas',
                distance_type='cosine'
            )
            save_data(
                os.path.join(im_path_SSLM_Chromas_cosine, file_name + ".npy"),
                sslm_chromas_cosine,
            )
            print("audio: " + audio_file + " sslm_chromas_cosine finished.")

        if not sslm_chromas_euclidean_exist:
            sslm_chromas_euclidean = sslm(
                input_signal=y,
                sr=sr,
                lag=l_frames,
                feature_type='chromas',
                distance_type='euclidean'
            )
            save_data(
                os.path.join(im_path_SSLM_Chromas_euclidean, file_name + ".npy"),
                sslm_chromas_euclidean,
            )
            print("audio: " + audio_file + " sslm_chromas_euclidean finished.")

        if not sslm_mfccs_cosine_exist:
            sslm_mfccs_cosine = sslm(
                input_signal=y,
                sr=sr,
                lag=l_frames,
                feature_type='mfccs',
                distance_type='cosine'
            )
            save_data(
                os.path.join(im_path_SSLM_MFCCs_cosine, file_name + ".npy"),
                sslm_mfccs_cosine,
            )
            print("audio: " + audio_file + " sslm_mfccs_cosine finished.")

        if not sslm_mfccs_euclidean_exist:
            sslm_mfccs_euclidean = sslm(
                input_signal=y,
                sr=sr,
                lag=l_frames,
                feature_type='mfccs',
                distance_type='euclidean'
            )
            save_data(
                os.path.join(im_path_SSLM_MFCCs_euclidean, file_name + ".npy"),
                sslm_mfccs_euclidean,
            )
            print("audio: " + audio_file + " sslm_mfccs_euclidean finished.")

        print("==========process end audio: ", file_name, " ==========")


# def batch_mls_sslm_extraction(audio_path, task_num=5):
#     p = Pool(task_num)
#     for root, dirs, files in os.walk(audio_path):
#         for file in files:
#             audio_file = os.path.join(root, file)
#             p.apply_async(mls_sslm_extraction, args=(audio_file,))
#     p.close()
#     p.join()


def batch_mls_sslm_extraction(audio_path):
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            audio_file = os.path.join(root, file)
            mls_sslm_extraction(audio_file)


batch_mls_sslm_extraction(audio_path)
