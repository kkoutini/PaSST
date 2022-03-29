import multiprocessing
import glob
import os
import sys

if len(sys.argv) > 1:
    FSD50K_base = sys.argv[1] # the path to of FSD50K base as downloaded from zalando.
else:
    FSD50K_base = "/home/khaled/shared/FSD50K/"  # the path to of FSD50K base as downloaded from zalando.
    print("Pass the path to FSD50K: python convert_to_mp3.py path/to/fsd50k")



outputp = FSD50K_base + "/mp3/"  # the path to the output mp3.

all_num = 0


def process_folder(fol="balanced_train_segments"):
    print("now working on ", fol)
    os.makedirs(outputp + fol, exist_ok=True)
    all_files = list(glob.glob(FSD50K_base + fol + "/*.wav"))
    print(f"it has {len(all_files)}")
    global all_num
    all_num = len(all_files)
    cmds = [(i, file, outputp + fol + "/" + os.path.basename(file)[:-3]) for i, file in enumerate(all_files)]
    print(cmds[0])
    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(process_one, cmds)


def process_one(i, f1, f2):
    if i % 100 == 0:
        print(f"{i}/{all_num} \t", f1)
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {f1} -codec:a mp3 -ar 32000 {f2}mp3")


print("We will convert the following folders to mp3: ")
folders = ['FSD50K.eval_audio', 'FSD50K.dev_audio']

print(folders)

for fol in folders:
    process_folder(fol)
