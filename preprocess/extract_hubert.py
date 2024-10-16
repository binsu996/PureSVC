import sys
import os
import ray
from tqdm import tqdm
import librosa
import torch
import argparse
import numpy as np
from pathlib import Path
from .hubert import hubert_model


def load_audio(file: str, sr: int = 16000):
    x, sr = librosa.load(file, sr=sr)
    return x


class HuBertExtractor:
    def __init__(self):
        # wget https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt -O pretrained_models/hubert-soft-0d54a1f4.pt
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = hubert_model.hubert_soft(
            "pretrained_models/hubert-soft-0d54a1f4.pt")
        model.eval()
        model.half()
        model.to(device)
        self.device = device
        self.model = model

    def extract(self, x):
        if isinstance(x,(str,Path)):
            x = load_audio(x)
        x = torch.from_numpy(x).to(self.device)
        x = x[None, None, :].half()
        with torch.no_grad():
            y = self.model.units(x).squeeze()
        return y

    def __call__(self, wavPath, vecPath):
        vec = self.get_hubert(wavPath).data.cpu().float().numpy()
        # print(vec.shape)   # [length, dim=256] hop=320
        np.save(vecPath, vec, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", help="wav", dest="wav", required=True)
    parser.add_argument("-v", "--vec", help="vec", dest="vec", required=True)
    parser.add_argument("-n", default=16, type=int)
    args = parser.parse_args()
    print(args.wav)
    print(args.vec)

    ray.init()

    os.makedirs(args.vec, exist_ok=True)
    wavPath = Path(args.wav)
    vecPath = Path(args.vec)

    hubert_array = [HuBertExtractor.remote() for i in range(args.n)]

    jobs = []
    idx = 0
    n_inst = len(hubert_array)
    for path_wav in wavPath.glob("**/*.wav"):
        path_vec = vecPath/path_wav.relative_to(wavPath).with_suffix(".vec")
        path_vec.parent.mkdir(exist_ok=True, parents=True)
        inst = hubert_array[idx % n_inst]
        jobs.append(inst.do.remote(path_wav, path_vec))
        idx += 1
    ray.get(jobs)
