from pathlib import Path
import numpy as np
import librosa


class CustomDataset:
    def __init__(self, list_file_path, base_folder, attr_names, sr=None):
        with open(list_file_path, 'r') as f:
            lines = f.readlines()
            stems = [x for x in lines if x != ""]
            self.stems = stems
        self.base_folder = base_folder
        self.attr_names = attr_names
        self.sr = sr

    def __len__(self):
        return len(self.stems)

    def load_attr(self, file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if file_path.suffix == ".npy":
            return np.load(file_path)
        elif file_path.suffix in [".mp3", ".wav"]:
            return librosa.load(str(file_path), sr=self.sr)
        else:
            raise NotImplementedError(f"unsupport file {file_path.suffix}")

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration()

        data = {}
        stem = self.stems[index]
        for attr, ext in self.attr_names.items():
            attr_file = Path(self.base_folder)/attr/f"{stem}{ext}"
            data[attr] = self.load_attr(attr_file)
        return data