import os
import numpy as np
import librosa
import argparse
import parselmouth
from omegaconf import OmegaConf
from tqdm import tqdm


def compute_f0(path, save, hps):
    x, sr = librosa.load(path, sr=hps.audio.sampling_rate)
    lpad = 1024 // 160
    rpad = lpad
    f0 = parselmouth.Sound(x, sr).to_pitch_ac(
        time_step=160 / sr,
        voicing_threshold=0.5,
        pitch_floor=30,
        pitch_ceiling=1000).selected_array['frequency']
    f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
    for index, pitch in enumerate(f0):
        f0[index] = round(pitch, 1)
    np.save(save, f0, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-p", "--pit", help="pit", dest="pit")
    args = parser.parse_args()
    print(args.wav)
    print(args.pit)
    os.makedirs(args.pit, exist_ok=True)
    wavPath = args.wav
    pitPath = args.pit

    hps = OmegaConf.load(f"./configs/nsf_bigvgan.yaml")

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            os.makedirs(f"./{pitPath}/{spks}", exist_ok=True)
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            for file in tqdm(os.listdir(f"./{wavPath}/{spks}"), spks):
                if file.endswith(".wav"):
                    # print(file)
                    file = file[:-4]
                    compute_f0(f"{wavPath}/{spks}/{file}.wav", f"{pitPath}/{spks}/{file}.pit", hps)
