import os
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import wavfile

def resample_wave(wav_in, wav_out, sample_rate):
    wav, _ = librosa.load(wav_in, sr=sample_rate)
    wav = wav / np.abs(wav).max() * 0.6
    wav = wav / max(0.01, np.max(np.abs(wav))) * 32767 * 0.6
    wavfile.write(wav_out, sample_rate, wav.astype(np.int16))

def process_file(file, wavPath, spks, outPath, sr):
    if file.endswith(".wav"):
        file = file[:-4]
        resample_wave(f"{wavPath}/{spks}/{file}.wav", f"{outPath}/{spks}/{file}.wav", sr)

def process_files_with_thread_pool(wavPath, spks, outPath, sr, thread_num=None):
    files = [f for f in os.listdir(f"./{wavPath}/{spks}") if f.endswith(".wav")]

    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = {executor.submit(process_file, file, wavPath, spks, outPath, sr): file for file in files}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing files'):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = 'please enter embed parameter ...'
    parser.add_argument("-w", "--wav", help="wav", dest="wav")
    parser.add_argument("-o", "--out", help="out", dest="out")
    parser.add_argument("-t", "--thread_count", help="thread count to process, set 0 to use all cpu cores", dest="thread_count", type=int, default=1)
    args = parser.parse_args()
    print(args.wav)
    print(args.out)

    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out
    args.sr = 32000  # 32kHz

    for spks in os.listdir(wavPath):
        if os.path.isdir(f"./{wavPath}/{spks}"):
            if not os.path.exists(f"./{outPath}/{spks}"):
                os.makedirs(f"./{outPath}/{spks}")
            print(f">>>>>>>>>>{spks}<<<<<<<<<<")
            if args.thread_count == 0:
                process_num = os.cpu_count()
            else:
                process_num = args.thread_count
            process_files_with_thread_pool(wavPath, spks, outPath, args.sr, process_num)
