# import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import signal
import librosa
import pandas as pd
import os

# from torchvision import transforms
# from torch.utils.data import DataLoader

# import sys
# import os
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from models.model_CNN_MLS_4SSLM import CNN_Fusion

# from data.extract_labels_from_txt import ReadDataFromtxt
# from data.dataloaders import (
#     SSMDataset,
#     normalize_image,
#     padding_MLS,
#     padding_SSLM,
#     borders,
# )


# "Create array of labels"
# def gaussian(x, mu, sig):
#     return np.exp(-np.power((x - mu)/sig, 2.)/2)

# def borders(image, label):
#     """This function transforms labels in sc to gaussians in frames"""
#     pooling_factor = 6
#     num_frames = image.shape[2]
#     repeated_label = []
#     for i in range(len(label)-1):
#         if label[i] == label[i+1]:
#             repeated_label.append(i)
#     label = np.delete(label, repeated_label, 0) #labels in seconds
#     label = label/pooling_factor #labels in frames

#     #Pad frames we padded in images also in labels but in seconds
#     sr = 44100
#     hop_length = 1024
#     window_size = 2048
#     padding_factor = 50
#     label_padded = [label[i] + padding_factor*hop_length/sr for i in range(label.shape[0])]
#     vector = np.arange(num_frames)
#     new_vector = (vector*hop_length + window_size/2)/sr
#     sigma = 0.1
#     gauss_array = 0
#     for mu in (label_padded[1:]):
#         gauss_array += gaussian(new_vector, mu, sigma)
#     for i in range(len(gauss_array)):
#         if gauss_array[i] > 1:
#             gauss_array[i] = 1
#     return image, gauss_array


def padding_MLS(image):
    """This function pads 30frames at the begining and end of an image"""
    sr = 44100
    hop_length = 1024
    window_size = 2048
    padding_factor = 50
    y = voss(padding_factor * hop_length - 1)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=window_size,
        hop_length=hop_length,
        n_mels=80,
        fmin=80,
        fmax=16000,
    )
    S_to_dB = librosa.power_to_db(S, ref=np.max)
    pad_image = S_to_dB[np.newaxis, :, :]

    # Pad MLS
    S_padded = np.concatenate((pad_image, image), axis=-1)
    S_padded = np.concatenate((S_padded, pad_image), axis=-1)
    return S_padded


def padding_SSLM(image):
    """This function pads 30frames at the begining and end of an image"""
    padding_factor = 50

    # Pad SSLM
    pad_image = np.full((image.shape[1], padding_factor), 1)
    pad_image = pad_image[np.newaxis, :, :]
    S_padded = np.concatenate((pad_image, image), axis=-1)
    S_padded = np.concatenate((S_padded, pad_image), axis=-1)
    return S_padded


def normalize(array):
    """This function normalizes a matrix along x axis (frequency)"""
    normalized = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        normalized[i, :] = (array[i, :] - np.mean(array[i, :])) / np.std(array[i, :])
    return normalized


def normalize_image(image):
    """This function normalies an image"""
    image = np.squeeze(image)  # remove
    image = normalize(image)
    # image = (image-np.min(image))/(np.max(image)-np.min(image))
    image = np.expand_dims(image, axis=0)
    return image


def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.

    nrows: number of values to generate
    rcols: number of random sources to add

    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values


epochs = "100"  # load model trained this number of epochs
# Model loading
output_channels = 32  # mapas características de salida de la capa 1 de la CNN
model = CNN_Fusion(output_channels, output_channels)
model.load_state_dict(
    torch.load(
        "/home/richie/Projects/GitHub/MusicBoundariesCNN/pretrained_weights/mel_4sslm_combined/mel_sslm_chromas_eucl_sslm_chromas_cosine_mfccs_eucl_mfccs_cosine_large/saved_model_"
        + epochs
        + "epochs.bin",
        map_location=torch.device('cpu'),
    )
)
model.eval()

# batch_size = 1

# transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

song_id = "t3"

im_path_base = "/home/richie/Desktop/sslm/"
im_path_mel = os.path.join(im_path_base, "mls/")
im_path_L_MFCCs = os.path.join(im_path_base, "SSLM_MFCCs_euclidean/")
im_path_L_MFCCs2 = os.path.join(im_path_base, "SSLM_MFCCs_cosine/")
im_path_L_MFCCs3 = os.path.join(im_path_base, "SSLM_Chromas_euclidean/")
im_path_L_MFCCs4 = os.path.join(im_path_base, "SSLM_Chromas_cosine/")

# labels_path = "/data/alpaca/datasets/annotations/salami/"


# mels_dataset = SSMDataset(
#     im_path_mel, labels_path, transforms=[padding_MLS, normalize_image, borders]
# )
# mels_trainloader = DataLoader(mels_dataset, batch_size=batch_size, num_workers=0)

# sslms_dataset = SSMDataset(
#     im_path_L_MFCCs, labels_path, transforms=[padding_SSLM, normalize_image, borders]
# )
# sslms_trainloader = DataLoader(sslms_dataset, batch_size=batch_size, num_workers=0)

# sslms_dataset2 = SSMDataset(
#     im_path_L_MFCCs2, labels_path, transforms=[padding_SSLM, normalize_image, borders]
# )
# sslms_trainloader2 = DataLoader(sslms_dataset2, batch_size=batch_size, num_workers=0)

# sslms_dataset3 = SSMDataset(
#     im_path_L_MFCCs3, labels_path, transforms=[padding_SSLM, normalize_image, borders]
# )
# sslms_trainloader3 = DataLoader(sslms_dataset3, batch_size=batch_size, num_workers=0)

# sslms_dataset4 = SSMDataset(
#     im_path_L_MFCCs4, labels_path, transforms=[padding_SSLM, normalize_image, borders]
# )
# sslms_trainloader4 = DataLoader(sslms_dataset4, batch_size=batch_size, num_workers=0)

hop_length = 1024
sr = 44100
window_size = 2024
pooling_factor = 6
lamda = 6 / pooling_factor
padding_factor = 50
lamda = round(lamda * sr / hop_length)
# lamda = 2
delta = 0.15
# beta = 1

image_mel = np.load(im_path_mel + song_id + ".npy")
image_sslm = np.load(im_path_L_MFCCs + song_id + ".npy")
image_sslm2 = np.load(im_path_L_MFCCs2 + song_id + ".npy")
image_sslm3 = np.load(im_path_L_MFCCs3 + song_id + ".npy")
image_sslm4 = np.load(im_path_L_MFCCs4 + song_id + ".npy")

image_mel = image_mel[np.newaxis, :, :]
image_sslm = image_sslm[np.newaxis, :, :]
image_sslm2 = image_sslm2[np.newaxis, :, :]
image_sslm3 = image_sslm3[np.newaxis, :, :]
image_sslm4 = image_sslm4[np.newaxis, :, :]

image_mel = padding_MLS(image_mel)
image_mel = normalize_image(image_mel)

image_sslm = padding_SSLM(image_sslm)
image_sslm = normalize_image(image_sslm)

image_sslm2 = padding_SSLM(image_sslm2)
image_sslm2 = normalize_image(image_sslm2)

image_sslm3 = padding_SSLM(image_sslm3)
image_sslm3 = normalize_image(image_sslm3)

image_sslm4 = padding_SSLM(image_sslm4)
image_sslm4 = normalize_image(image_sslm4)

image_mel = np.expand_dims(image_mel, 0)  # creating dimension corresponding to batch
image_sslm = np.expand_dims(image_sslm, 0)
image_sslm2 = np.expand_dims(image_sslm2, 0)
image_sslm3 = np.expand_dims(image_sslm3, 0)
image_sslm4 = np.expand_dims(image_sslm4, 0)

if image_mel.shape[3] != image_sslm.shape[3]:
    image_mel = image_mel[:, :, :, 1:]

image_mel = torch.Tensor(image_mel)
image_sslm = torch.Tensor(image_sslm)
image_sslm2 = torch.Tensor(image_sslm2)
image_sslm3 = torch.Tensor(image_sslm3)
image_sslm4 = torch.Tensor(image_sslm4)

image_sslm = torch.cat((image_sslm, image_sslm2, image_sslm3, image_sslm4), 1)


pred = model(image_mel, image_sslm)
pred = pred.view(-1, 1)
pred = torch.sigmoid(pred)
pred_with_confidence = pred.detach().numpy()
pred_new = pred_with_confidence[:, 0]

peak_position = signal.find_peaks(pred_new, height=delta, distance=lamda)[
    0
]  # array of peaks
peaks_positions = ((peak_position - padding_factor) * pooling_factor * hop_length) / sr
for i in range(len(peaks_positions)):  # if the 1st element is <0 convert it to 0
    if peaks_positions[i] < 0:
        peaks_positions[i] = 0

pred_positions = np.array(
    (np.copy(peaks_positions[:-1]), np.copy(peaks_positions[1:]))
).T
repeated_list = []
for j in range(pred_positions.shape[0]):
    if pred_positions[j, 0] == pred_positions[j, 1]:
        repeated_list.append(j)
pred_positions = np.delete(pred_positions, repeated_list, 0)
