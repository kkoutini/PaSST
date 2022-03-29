# %%
import sys

import h5py
import pandas as pd
import numpy as np
import csv
import os

# %%
from numpy import dtype

if len(sys.argv) > 1:
    FSD50K_base = sys.argv[1] # the path to of FSD50K base as downloaded from zalando.
else:
    FSD50K_base = "/home/khaled/shared/FSD50K/"  # the path to of FSD50K base as downloaded from zalando.
    print("Pass the path to FSD50K: python convert_to_mp3.py path/to/fsd50k")

base_dir = "../../audioset_hdf5s/" # the path to store hdf file.



####
balanced_csv = FSD50K_base + "FSD50K.ground_truth/dev.csv"
eval_csv = FSD50K_base + "FSD50K.ground_truth/eval.csv"
class_idx_csv = FSD50K_base + "FSD50K.ground_truth/vocabulary.csv"
mp3_path = "/home/khaled/shared/FSD50K/mp3/"

# %%

df = pd.read_csv(class_idx_csv, header=None, index_col=0)
classes_list = list(df[1].values)
assert sorted(classes_list) == classes_list
id_to_ix = {id: i for i, id in enumerate(classes_list)}
ix_to_id = {i: id for i, id in enumerate(classes_list)}

# %%

# Load labels
df = pd.read_csv(balanced_csv)

train = df[df.split == "train"]
val = df[df.split == "val"]

eval = pd.read_csv(eval_csv)


# %%
def get_labels(df):
    y = np.zeros((len(df), 200), dtype=np.int32)

    for i, target in enumerate(df.labels.values):
        for t in target.split(","):
            y[i, id_to_ix[t]] = 1
    return df.fname.values, y


# %%

for set_name, df, prefix in [("train", train, "FSD50K.dev_audio/"), ("val", val, "FSD50K.dev_audio/"),
                             ("eval", eval, "FSD50K.eval_audio/")]:
    print("now working on ", set_name, prefix, "len=", len(df))
    # files, y = torch.load(read_file+".pth")
    files, y = get_labels(df)
    y = np.packbits(y, axis=-1)
    packed_len = y.shape[1]
    print(files[0], "classes: ", packed_len, y.dtype)
    available_size = len(files)
    dt = h5py.vlen_dtype(np.dtype('uint8'))
    save_file = "FSD50K." + set_name
    if os.path.isfile(base_dir + "mp3/" + save_file + "_mp3.hdf"):
        print(base_dir + "mp3/" + save_file + "_mp3.hdf", "exists!\n\n\n contiue")
        continue
    with h5py.File(base_dir + "mp3/" + save_file + "_mp3.hdf", 'w') as hf:
        audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
        waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
        target = hf.create_dataset('target', shape=((available_size, packed_len)), dtype=y.dtype)
        for i, file in enumerate(files):
            if i % 1000 == 0:
                print(f"{i}/{available_size}")
            f = f"{file}.mp3"
            a = np.fromfile(mp3_path + prefix + f, dtype='uint8')
            audio_name[i] = f
            waveform[i] = a
            target[i] = y[i]

    print(a.shape)
    print("Done!", prefix)
