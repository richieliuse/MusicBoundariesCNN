import librosa
import librosa.display
import librosa.core.audio
import numpy as np
import skimage.measure
import scipy
# from scipy.spatial import distance
import os
import pathlib
from multiprocessing import Pool
from pyinstrument import Profiler
import cupy as cp
import cucim.skimage.measure
# from fastdist import fastdist
from numba import jit, njit, prange, guvectorize
import audioread

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


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def to_mono(y):
    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim-1)))

    return y


def ar_load(path, offset, duration, dtype):
    y = []
    with audioread.audio_open(path) as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels

        s_start = int(np.round(sr_native * offset)) * n_channels

        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + (int(np.round(sr_native * duration)) * n_channels)

        n = 0

        for frame in input_file:
            frame = buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)

            if n < s_start:
                # offset is after the current frame
                # keep reading
                continue

            if s_end < n_prev:
                # we're off the end.  stop reading
                break

            if s_end < n:
                # the end is in this frame.  crop.
                frame = frame[: s_end - n_prev]

            if n_prev <= s_start <= n:
                # beginning is in this frame
                frame = frame[(s_start - n_prev):]

            # tack on the current frame
            y.append(frame)

    if y:
        y = np.concatenate(y)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)

    y = to_mono(y)

    return y, sr_native


def split_filename(filename, full_name=True):
    path = pathlib.Path(filename)
    if full_name:
        return path.name, path.suffix
    return path.stem, path.suffix


def load_audio(input_audio, sr_desired):
    "This function loads the audio file and resamples it to sr_desired"
    file_name, file_suffix = split_filename(input_audio)
    y, sr = ar_load(path=input_audio, offset=0.0, duration=None, dtype=np.float32) if file_suffix == ".mp3" else librosa.load(input_audio, sr=sr_desired)

    # y, sr = librosa.load(input_audio, sr=sr_desired)
    if sr != sr_desired:
        y = librosa.core.resample(y, sr, sr_desired)
        sr = sr_desired
    return y, sr


def save_data(save_path, input_data):
    # check file exists
    if os.path.exists(save_path):
        print(save_path, " exists")
        return
    np.save(save_path, input_data)


def lag_near_frames(lag_near_sec=13.93, sr=44100, hop_length=1024):
    return round(lag_near_sec * sr / hop_length)


def fourier_transform(y, window_size=2048, hop_length=1024):
    stft = np.abs(librosa.stft(y=y, n_fft=window_size, hop_length=hop_length))
    return stft


def mel_spectrogram(y, sr=44100, window_size=2048, hop_length=1024):
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


def max_pooling(input_signal, pooling_factor):
    output_singal = skimage.measure.block_reduce(
        input_signal, (1, pooling_factor), np.max
    )
    return output_singal


def max_pooling_cuda(input_signal, pooling_factor):
    output_singal = cucim.skimage.measure.block_reduce(
        input_signal, (1, pooling_factor), cp.max
    )
    return output_singal


def mls(input_signal, sr=44100, window_size=2048, hop_length=1024, pooling_factor=6):
    mel = mel_spectrogram(input_signal, sr, window_size, hop_length)
    output_singal = max_pooling(mel, pooling_factor)
    return output_singal


# @njit
# def compute_distances(x_hat, padding_factor, first_pooling_factor, distance_type):
#     # Cosine distance calculation: D[N/p,L/p] matrix
#     distances = np.zeros(
#         (x_hat.shape[1], padding_factor // first_pooling_factor)
#     )  # D has as dimensions N/p and L/p
#     for i in range(x_hat.shape[1]):  # iteration in columns of x_hat
#         for l in range(padding_factor // first_pooling_factor):
#             if i - (l + 1) < 0:
#                 cosine_dist = 1
#             elif i - (l + 1) < padding_factor // first_pooling_factor:
#                 cosine_dist = 1
#             else:
#                 if distance_type == 'cosine':
#                     cosine_dist = distance.cosine(x_hat[:, i], x_hat[:, i - (l + 1)])
#                 elif distance_type == 'euclidean':
#                     cosine_dist = distance.euclidean(x_hat[:, i], x_hat[:, i - (l + 1)])  # cosine distance between columns i and i-L
#                 else:
#                     cosine_dist = 0

#                 if cosine_dist == float('nan'):
#                     cosine_dist = 0
#             distances[i, l] = cosine_dist
#     return distances


@njit(fastmath=True, parallel=True)
def eucl_naive(A, B):
    assert A.shape[1] == B.shape[1]
    C = np.empty((A.shape[0], B.shape[0]), A.dtype)

    # workaround to get the right datatype for acc
    init_val_arr = np.zeros(1, A.dtype)
    init_val = init_val_arr[0]

    for i in prange(A.shape[0]):
        for j in range(B.shape[0]):
            acc = init_val
            for k in range(A.shape[1]):
                acc += (A[i, k] - B[j, k]) ** 2
            C[i, j] = np.sqrt(acc)
    return C


@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", target='parallel')
def fast_cosine_gufunc(u, v, result):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = 1 - udotv / (u_norm * v_norm)
    result[:] = ratio


@guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", target='parallel')
def fast_euclidean_gufunc(u, v, result):
    m = u.shape[0]
    udotv = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

        udotv += abs(u[i] - v[i]) ** 2

    dist = udotv ** (1 / 2)
    result[:] = dist


# @jit(nopython=True, fastmath=True)
@njit(fastmath=True, parallel=True)
def compute_dist_cosine(u, v):
    n = u.size
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(n):
        num += u[i] * v[i]
        u_norm += abs(u[i]) ** 2
        v_norm += abs(v[i]) ** 2

    denom = (u_norm * v_norm) ** (1 / 2)
    result = num / denom
    return result


# @jit(nopython=True, fastmath=True)
@njit(fastmath=True, parallel=True)
def compute_dist_euclidean(u, v):
    n = u.size
    dist = 0
    for i in range(n):
        dist += abs(u[i] - v[i]) ** 2
    result = dist ** (1 / 2)
    return result


# @jit
def compute_distances(x_hat, padding_factor, first_pooling_factor, distance_type):
    # Cosine distance calculation: D[N/p,L/p] matrix
    distances = np.zeros(
        (x_hat.shape[1], padding_factor // first_pooling_factor)
    )  # D has as dimensions N/p and L/p
    tmp_r = np.zeros(1)
    for i in range(x_hat.shape[1]):  # iteration in columns of x_hat
        for j in range(padding_factor // first_pooling_factor):
            t_dist = 0
            # if i - (l + 1) < 0:
            #     t_dist = 1
            if i - (j + 1) < padding_factor // first_pooling_factor:
                t_dist = 1
            else:
                # t_dist = 1 - compute_dist_cosine(x_hat[:, i], x_hat[:, i - (j + 1)]) if distance_type == 'cosine' else compute_dist_euclidean(x_hat[:, i], x_hat[:, i - (j + 1)])
                if distance_type == 'cosine':
                    t_dist = fast_cosine_gufunc(x_hat[:, i], x_hat[:, i - (j + 1)], tmp_r)[0]
                else:
                    t_dist = fast_euclidean_gufunc(x_hat[:, i], x_hat[:, i - (j + 1)], tmp_r)[0]

                if t_dist == float('nan'):
                    t_dist = 0
            distances[i, j] = t_dist
    return distances


@jit(nopython=True, fastmath=True)
def compute_epsilon(distances, padding_factor, first_pooling_factor, kappa):
    epsilon = np.zeros(
        (distances.shape[0], padding_factor // first_pooling_factor)
    )  # D has as dimensions N/p and L/p
    for i in range(
        padding_factor // first_pooling_factor, distances.shape[0]
    ):  # iteration in columns of x_hat
        for j in range(padding_factor // first_pooling_factor):
            epsilon[i, j] = np.quantile(
                np.concatenate((distances[i - j, :], distances[i, :])), kappa
            )

    return epsilon


def sslm(
    input_signal,
    sr=44100,
    window_size=2048,
    hop_length=1024,
    first_pooling_factor=2,
    second_pooling_factor=3,
    lag=30,
    feature_type='chromas',
    distance_type="cosine",
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
    orignal_signal = (
        fourier_transform(input_signal, window_size, hop_length)
        if feature_type == 'chromas' and distance_type == 'cosine'
        else mel_spectrogram(input_signal, sr, window_size, hop_length)
    )
    # if feature_type == 'chromas' and distance_type == 'cosine':
    #     orignal_signal = fourier_transform(input_signal, window_size, hop_length)
    # else:
    #     orignal_signal = mel_spectrogram(input_signal, sr, window_size, hop_length)

    padding_factor = lag
    """"This part pads a mel spectrogram gived the spectrogram a lag parameter
    to compare the first rows with the last ones and make the matrix circular"""
    pad = np.full(
        (orignal_signal.shape[0], padding_factor), -70
    )  # matrix of 80x30frames of -70dB corresponding to padding
    S_padded = np.concatenate(
        (pad, orignal_signal), axis=1
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
    x = [np.roll(before_bagging, n, axis=1) for n in range(m)]
    x_hat = np.concatenate(x, axis=0)

    # Cosine distance calculation: D[N/p,L/p] matrix
    distances = compute_distances(
        x_hat, padding_factor, first_pooling_factor, distance_type
    )

    # Threshold epsilon[N/p,L/p] calculation
    kappa = 0.1
    # distances_cp = cp.asarray(distances)
    epsilon = compute_epsilon(distances, padding_factor, first_pooling_factor, kappa)

    # We remove the padding done before
    distances = distances[padding_factor // first_pooling_factor:, :]
    epsilon = epsilon[padding_factor // first_pooling_factor:, :]
    # x_prime = x_prime[:, padding_factor // first_pooling_factor:]

    # Self Similarity Lag Matrix
    # distances = cp.asnumpy(distances_cp)
    # epsilon = cp.asnumpy(epsilon_cp)
    # expit = cp.ElementwiseKernel(
    #     'float64 x',
    #     'float64 y',
    #     'y = 1 / (1 + exp(-x))',
    #     'expit'
    # )
    # distances_cp = cp.asarray(distances)
    # epsilon_cp = cp.asarray(epsilon)
    # sslm = expit(1 - distances_cp / epsilon_cp)

    sslm = scipy.special.expit(1 - distances / epsilon)  # aplicación de la
    # sigmoide
    sslm = np.transpose(sslm)
    if second_pooling_factor > 1:
        sslm = max_pooling(sslm, second_pooling_factor)

    # Check if SSLM has nans and if it has them, substitute them by 0
    sslm[np.isnan(sslm)] = 0

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
                distance_type='cosine',
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
                distance_type='euclidean',
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
                distance_type='cosine',
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
                distance_type='euclidean',
            )
            save_data(
                os.path.join(im_path_SSLM_MFCCs_euclidean, file_name + ".npy"),
                sslm_mfccs_euclidean,
            )
            print("audio: " + audio_file + " sslm_mfccs_euclidean finished.")

        print("==========process end audio: ", file_name, " ==========")


def batch_mls_sslm_extraction(audio_path, task_num=5):
    p = Pool(task_num)
    for root, dirs, files in os.walk(audio_path):
        for file in files:
            audio_file = os.path.join(root, file)
            p.apply_async(mls_sslm_extraction, args=(audio_file,))
    p.close()
    p.join()


profiler = Profiler()
profiler.start()

# batch_mls_sslm_extraction(audio_path)
mls_sslm_extraction("/home/richie/Desktop/test_songs/The Weeknd - Blinding Lights.mp3")

profiler.stop()
print(profiler.output_text(unicode=True, color=True))
