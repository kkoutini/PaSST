# %%
import h5py
import pandas as pd
import numpy as np
import csv
import os


# %%
base_dir = "../../audioset_hdf5s/"
balanced_csv= base_dir+ "metadata/balanced_train_segments.csv"
eval_csv= base_dir+ "metadata/eval_segments.csv"
mp3_path = "../../mp3_audio/"


# %%

def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.
    source: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/utils/utilities.py#L59
    Args:
      csv_path: str
    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]   # Remove heads

    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=np.bool)
    audio_names = []
 
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        audio_name = 'Y{}.mp3'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        label_ids = items[3].split('"')[1].split(',')

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1

    meta_dict = {'audio_name': np.array(audio_names), 'target': targets}
    return meta_dict

# Load label
with open(base_dir+'metadata/class_labels_indices.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)

labels = []
ids = []    # Each label has a unique id such as "/m/068hy"
for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)

classes_num = len(labels)

lb_to_ix = {label : i for i, label in enumerate(labels)}
ix_to_lb = {i : label for i, label in enumerate(labels)}

id_to_ix = {id : i for i, id in enumerate(ids)}
ix_to_id = {i : id for i, id in enumerate(ids)}

# %%

def check_available(balanced_csv,balanced_audio_path,prefix=None):
    meta_csv = read_metadata(balanced_csv,classes_num,id_to_ix)
    audios_num = len(meta_csv['audio_name'])
    found=0
    notfound=0
    available_files=[]
    available_targets=[]
    if prefix is None:
        prefix = os.path.basename(balanced_csv)[:-4]
    for n in range(audios_num):
        audio_path =  meta_csv['audio_name'][n]
        #print(balanced_audio_path + f"{prefix}/{audio_path}")
        if os.path.isfile(balanced_audio_path + f"{prefix}/{audio_path}" ):
            found+=1
            available_files.append(meta_csv['audio_name'][n])
            available_targets.append(meta_csv['target'][n])
        else:
            notfound+=1
    print(f"Found {found} . not found {notfound}")
    return available_files,available_targets
# %%

# %%

# %%



for read_file,prefix in [(balanced_csv,"balanced_train_segments/"), (eval_csv,"eval_segments/"),]:
    print("now working on ",read_file,prefix)
    #files, y = torch.load(read_file+".pth")
    files, y = check_available(read_file, mp3_path)
    y = np.packbits(y, axis=-1)
    packed_len = y.shape[1]
    print(files[0], "classes: ",packed_len, y.dtype)
    available_size = len(files)
    f = files[0][:-3]+"mp3"
    a = np.fromfile(mp3_path+prefix + "/"+f, dtype='uint8')

    dt = h5py.vlen_dtype(np.dtype('uint8'))
    save_file = prefix.split("/")[0]
    with h5py.File(base_dir+ "mp3/" + save_file+"_mp3.hdf", 'w') as hf:
        audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
        waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
        target = hf.create_dataset('target', shape=((available_size, packed_len)), dtype=y.dtype)
        for i,file in enumerate(files):
            if i%1000==0:
                print(f"{i}/{available_size}")
            f = file[:-3] + "mp3"
            a = np.fromfile(mp3_path + prefix  + f, dtype='uint8')
            audio_name[i]=f
            waveform[i] = a
            target[i] = y[i]

    print(a.shape)
    print("Done!" , prefix)


# %%
print("working on unbalanced...")



all_x,all_y = None, None
for idx in  range(41):
    print("working on ",idx)
    tmp_csv = base_dir+ f"metadata/unbalanced_partial_csvs/unbalanced_train_segments_part{idx:02}.csv"
    prefix = f"unbalanced_train_segments/unbalanced_train_segments_part{idx:02}"
    x,y = check_available(tmp_csv,mp3_path,prefix=prefix)
    x = np.array([f"{idx:02}/"+one for one in x])
    y=np.packbits(y, axis=-1)
    print("x,y",x.shape, y.shape)
    if all_x is None:
        all_x = x
        all_y = y
    else:
        all_x = np.concatenate((all_x,x))
        all_y = np.concatenate((all_y,y))
    print(f"done {idx}! all x,y",all_x.shape, all_y.shape)



print("now working on packing  unbalanced")
prefix = "unbalanced_train_segments/unbalanced_train_segments_part"
files = all_x
y = all_y
packed_len = y.shape[1]
print(files[0], "classes: ",packed_len, y.dtype)
available_size = len(files)
f = files[0][:-3]+"mp3"
a = np.fromfile(mp3_path+prefix + f, dtype='uint8')

dt = h5py.vlen_dtype(np.dtype('uint8'))
save_file = prefix.split("/")[0]
with h5py.File(base_dir+ "mp3/" + save_file+"_mp3.hdf", 'w') as hf:
    audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
    waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
    target = hf.create_dataset('target', shape=((available_size, packed_len)), dtype=y.dtype)
    for i,file in enumerate(files):
        if i%1000==0:
            print(f"{i}/{available_size}")
        f = file[:-3] + "mp3"
        a = np.fromfile(mp3_path + prefix  + f, dtype='uint8')
        audio_name[i]=f
        waveform[i] = a
        target[i] = y[i]

print(a.shape)
print("Done!" , prefix)


