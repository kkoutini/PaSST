import os
import tarfile
import multiprocessing
import glob
import h5py
import numpy as np

from torch.hub import download_url_to_file

# global constants
openmicurl = "https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz?download=1"
download_target = "openmic-2018-v1.0.0.tgz"
extract_target = download_target.replace(".tgz", "")
dataset_path = os.path.join(extract_target, "openmic-2018/")
mp3_path = os.path.join(dataset_path, "mp3/")
hdf5s_dir = "../../audioset_hdf5s/"
train_files_csv = os.path.join(dataset_path, "partitions/split01_train.csv")
test_files_csv = os.path.join(dataset_path, "partitions/split01_test.csv")


def download(force=False):
    if force or not os.path.isfile(download_target):
        print("Downloading OpenMIC from zenodo...")
        download_url_to_file(openmicurl, download_target)
    else:
        print(f"{download_target} already exists. Skipping download!")


def untar():
    my_tar = tarfile.open(download_target)
    print(f"Extracting openmic from {download_target} to {extract_target}")

    my_tar.extractall(extract_target)


def process_folder(fol="balanced_train_segments"):
    print("now working on ", fol)
    os.makedirs(mp3_path + fol, exist_ok=True)
    all_files = list(glob.glob(os.path.join(dataset_path, "audio/") + "/*/*.ogg"))  # openmic format
    print(f"it has {len(all_files)}")
    print(all_files[:5])
    global all_num
    all_num = len(all_files)
    cmds = [(i, file, mp3_path + fol + "/" + os.path.basename(file)[:-3]) for i, file in enumerate(all_files)]
    print(cmds[0])
    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(process_one, cmds)


def process_one(i, f1, f2):
    if i % 100 == 0:
        print(f"{i}/{all_num} \t", f1)
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {f1} -codec:a mp3 -ar 32000 {f2}mp3")


def make_mp3():
    process_folder("audio")


def read_metadata(csv_path, classes_num, id_to_ix, openmicf):
    """Read metadata of AudioSet from a csv file.
    source: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/utils/utilities.py#L59
    Args:
      csv_path: str
    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    audios_num = len(lines)
    # class + mask
    targets = np.zeros((audios_num, classes_num * 2), dtype=np.float32)
    audio_names = []
    notfound = set()
    for i, row in enumerate(lines):
        audio_name = '{}.mp3'.format(row)  # Audios are started with an extra 'Y' when downloading
        audio_names.append(audio_name)
        t = openmicf.Y_true[id_to_ix[row]] #id_to_ix[row]["true"]
        m = openmicf.Y_mask[id_to_ix[row]].astype(int)#id_to_ix[row]["mask"]
        # Target
        targets[i, :classes_num] = t
        targets[i, classes_num:] = m

    print(notfound)
    print("original_targets", len(targets))
    mask = targets.astype(np.int).sum(1) > 0
    print(len(mask), mask.sum())
    print("after: ", len(targets[mask]))
    meta_dict = {'audio_name': np.array(audio_names)[mask], 'target': targets[mask]}
    return meta_dict


def get_files_labels(balanced_csv, balanced_audio_path, d_files, openmicf, prefix=None, zip_contents=None, classes_num=20):
    meta_csv = read_metadata(balanced_csv, classes_num, d_files, openmicf)
    # fname,labels,mids
    audios_num = len(meta_csv['audio_name'])
    found = 0
    notfound = 0
    available_files = []
    available_targets = []
    for n in range(audios_num):
        audio_path = meta_csv['audio_name'][n]
        # print(balanced_audio_path + f"{prefix}/{audio_path}")
        if n == 0:
            print("checking: ", balanced_audio_path + f"{prefix}/{audio_path}")
        if os.path.isfile(balanced_audio_path + f"{prefix}/{audio_path}"):
            found += 1
            available_files.append(meta_csv['audio_name'][n])
            available_targets.append(meta_csv['target'][n])
        else:
            notfound += 1
    print(f"Found {found} . not found {notfound}")
    return available_files, available_targets


def pack():
    d_files = dict()
    opmic = np.load(os.path.join(dataset_path, "openmic-2018.npz"))
    opmic.allow_pickle = True

    for i, sid in enumerate(opmic.f.sample_key):

        d_files[sid] = i #{"mask": opmic.f.Y_mask[i].astype(int),
                        #"true": opmic.f.Y_true[i]}
    print("len=",len(d_files))

    for read_file, prefix in [(train_files_csv, "audio/"), (test_files_csv, "audio/")]:
        print("now working on ", read_file, prefix)
        # files, y = torch.load(read_file+".pth")
        files, y = get_files_labels(read_file, mp3_path, d_files=d_files, openmicf=opmic.f, prefix=prefix)
        y = np.array(y)
        # y = np.packbits(y, axis=-1)
        packed_len = y.shape[1]
        print(files[0], "classes: ", packed_len, y.dtype)
        available_size = len(files)
        f = files[0]
        a = np.fromfile(mp3_path + prefix + "/" + f, dtype='uint8')

        dt = h5py.vlen_dtype(np.dtype('uint8'))
        save_file = read_file.rsplit("/", 1)[1].replace("split01", "openmic")
        os.makedirs(hdf5s_dir + "mp3/" ,exist_ok=True)
        if os.path.isfile(hdf5s_dir + "mp3/" + save_file + "_mp3.hdf"):
            print(hdf5s_dir + "mp3/" + save_file + "_mp3.hdf", "exists!\n\n\n contiue")
            continue
        with h5py.File(hdf5s_dir + "mp3/" + save_file + "_mp3.hdf", 'w') as hf:
            audio_name = hf.create_dataset('audio_name', shape=((available_size,)), dtype='S20')
            waveform = hf.create_dataset('mp3', shape=((available_size,)), dtype=dt)
            target = hf.create_dataset('target', shape=((available_size, packed_len)), dtype=y.dtype)
            for i, file in enumerate(files):
                if i % 1000 == 0:
                    print(f"{i}/{available_size}")
                f = file
                a = np.fromfile(mp3_path + prefix + f, dtype='uint8')
                audio_name[i] = f
                waveform[i] = a
                target[i] = y[i]
        print("Saved h5py file into ", hdf5s_dir + "mp3/" + save_file + "_mp3.hdf")
        print(a.shape)
        print("Done!", prefix)


def preprocess():
    download()
    untar()
    make_mp3()
    pack()


preprocess()