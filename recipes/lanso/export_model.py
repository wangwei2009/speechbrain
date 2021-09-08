#!/usr/bin/env/python3
"""Recipe for training a speech enhancement system with spectral masking.

To run this recipe, do the following:
> python train.py train.yaml --data_folder /path/to/save/mini_librispeech

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Szu-Wei Fu 2020
 * Chien-Feng Liao 2020
 * Peter Plantinga 2021
"""
import sys
import os
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from train import SpeakerBrain
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
from tqdm import tqdm
import time
from models.export_onnx_mnn import export_onnx_mnn

def save_audio(filename: str, audio: np.ndarray, fs=16000):
    """Save loaded audio to file using the configured audio parameters"""
    if not filename.endswith(".wav"):
        filename = filename + ".wav"
    audio = (audio * np.iinfo(np.int16).max).astype(np.int16)

    wavfile.write(filename, fs, audio)   # audio should be (Nsamples, Nchannels)

def save_prob_to_wav(audio: np.ndarray, prob_vec: np.ndarray, hop_len: int, predict_win: int, filename: str, delay=0):
    """combine mono audio with prob_vec to stereo data

    Args:
        audio (np.ndarray): [description]
        prob_vec (np.ndarray): [description]
        win_len (int): [description]
        filename (str): [description]
        delay (int, optional): [description]. Defaults to 0.
    """
    from scipy.io import wavfile
    audio = np.squeeze(audio)
    output_wavdata = np.zeros((len(audio), 2))
    output_wavdata[:, 0] = np.squeeze(audio)
    frame_num = int(len(audio)/hop_len) - predict_win

    print("frame_len:{}".format(frame_num))
    print("frame_len:{}".format(frame_num))

    prob_interp = np.zeros(len(audio))
    print(audio.shape)
    print(prob_vec.shape)
    print(prob_interp.shape)

    for n in range(frame_num-2):
        prob_interp[n*hop_len:(n+1)*hop_len] = np.ones(hop_len) * prob_vec[n]

    if delay >0:   # delay channel-2
        output_wavdata[delay:, 1] = prob_interp[:len(output_wavdata[delay:, 1])]
    else:          # delay channel-2
        output_wavdata[:len(prob_interp[delay*-1:]), 1] = prob_interp[delay*-1:]

    save_audio(filename, output_wavdata)

def loadtxt(filename:str) -> np.array:
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            print(line)

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    print(hparams_file)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    se_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file
    )

    ind2lab = label_encoder.ind2lab
    print(ind2lab)
    lab2ind = label_encoder.lab2ind
    print(lab2ind)

    words_wanted=['小蓝小蓝', '管家管家', '物业物业', 'unknown']

    wav = '/home/wangwei/work/corpus/kws/lanso/LS-ASR-data_16k_trim_command_folder_distance_rnd3/dev/物业物业/05/193-05-164.wav'
    wav = '/home/wangwei/work/corpus/kws/lanso/LS-ASR-data_16k_trim_command_folder_distance_rnd3_save14/dev/物业物业/05_noisy/20-05-151_noisy_0.wav'
    wav_data = sb.dataio.dataio.read_audio(wav)
    noisy_wavs = wav_data.reshape(1, -1)
    print("noisy_wavs.shape:{}".format(noisy_wavs.shape))

    input_buffer = torch.zeros((1, 151, 40)).to(se_brain.device)

    noisy_feats = se_brain.modules.compute_features(noisy_wavs)
    print("noisy_feats.shape:{}".format(noisy_feats.shape))
    input_buffer[:, -noisy_feats.shape[1]:, :] = noisy_feats
    print("noisy_feats.shape:{}".format(noisy_feats.shape))
    noisy_feats_npy = input_buffer.numpy()
    # print("noisy_feats_npy:{}".format(noisy_feats_npy[0, 3, :]))
    print("noisy_feats_npy.shape:{}".format(noisy_feats_npy.shape))
    # print(noisy_feats_npy[0, :, 1])
    plt.figure()
    plt.pcolor(noisy_feats_npy[0].transpose())
    plt.colorbar()
    plt.savefig('noisy_feats_npy.png')
    # np.save('noise_floor.npy', noisy_feats_npy[:, -151:, :])

    # se_brain.on_evaluate_start(min_key="ErrorRate")
    se_brain.on_evaluate_start()
    se_brain.on_stage_start(sb.Stage.TEST, epoch=None)

    print("Epoch loaded: {}".format(hparams['epoch_counter'].current))

    se_brain.modules.eval()

    # # data = np.loadtxt('fbank_flatten.txt', delimiter=',')
    # data = np.loadtxt('/home/wangwei/work/kws/kws_lanso/bin/fbank.txt', delimiter=',')
    # data = data.reshape(1, -1, 40)
    # print("data.shape:{}".format(data.shape))
    # data_torch = torch.from_numpy(data).to(se_brain.device)
    # # print(data_torch[0, :, 1])
    # # noisy_feats[:, 0, :] = torch.from_numpy(np.random.rand(40)).to(se_brain.device)/100
    # input_buffer[:, -data.shape[1]:, :] = data_torch
    # # noisy_feats[:, 0, :] = noisy_feats[:, 1, :]
    # fbank = input_buffer.numpy()
    # print("fbank.shape:{}".format(fbank.shape))
    # plt.figure()
    # plt.pcolor(fbank[0].transpose())
    # plt.colorbar()
    # plt.savefig('fbank.png')

    noisy_feats = se_brain.modules.mean_var_norm(input_buffer, torch.ones([1]).to(se_brain.device))

    dummy_input = torch.rand((1, 1, 151, 40)).to(se_brain.device)

    print("noisy_feats.shape:{}".format(noisy_feats.shape))
    dummy_input[0, :, :noisy_feats.shape[1], :] = noisy_feats

    # noisy_feat_npy = noisy_feats[0, :].detach().cpu().numpy()
    # np.savetxt('noisy_feat_wuye.txt', noisy_feat_npy, delimiter=',')

    # data = np.loadtxt('fbank.txt', delimiter=',')
    # data = data.reshape(1, -1, 40)
    # data_torch = torch.from_numpy(data).to(se_brain.device)
    # dummy_input[0, :, 1:, :] = data_torch

    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='tdnn4_save8_seed46.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='tdnn4_save16_seed833_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='crdnn_save163_seed88_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='tdnn4_save16_seed8911_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='tdnn5_save19_reverb_seed921_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/tdnn5_save20_seed931_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/bcresnet_save20_seed98_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/bcresnet_M_save20_seed99_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/bcresnet_M_save20_seed991_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/bcresnet_L_save20_seed982_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/bcresnet_S_save20_seed1013_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/tdnn4_save21_seed1023_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/tdnn7_save21_seed1031_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/tdnn4_save21_seed1024_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='model_export/tdnn4_save21_seed1026_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save21_seed1029_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save21_seed1028_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save21_seed10210_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save21_seed10211_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save22_seed1071_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save22_seed1071_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/bcresnet_S_save21_seed1013_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save16_seed8911_kw2_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save22_seed1081_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save22_seed1082_latest.mnn')
    export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/bcresnet1_seed1141_latest.mnn')
    # export_onnx_mnn(se_brain.modules.embedding_model, dummy_input, output_model='/home/wangwei/work/kws/kws_lanso/bin/tdnn4_save22_seed108_best.mnn')