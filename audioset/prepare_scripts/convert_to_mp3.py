import argparse
import multiprocessing
import glob
import os

# Replace this with the dataset downloaded using PANN scripts
source_path = "/share/cp/datasets/full_audioset/audioset201906/audios/"

# Replace with the output directory
out_path = "./mp3_audio/"

all_num = 0


def process_folder(fol="balanced_train_segments"):
    print("now working on ", fol)
    os.makedirs(out_path + fol, exist_ok=True)
    all_files = list(glob.glob(source_path + fol + "/*.wav"))
    print(f"it has {len(all_files)}")
    global all_num
    all_num = len(all_files)
    cmds = [(i, file, out_path + fol + "/" + os.path.basename(file)[:-3]) for i, file in enumerate(all_files)]
    print(cmds[0])
    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(process_one, cmds)


def process_one(i, f1, f2):
    if i % 100 == 0:
        print(f"{i}/{all_num} \t", f1)
    os.system(f"ffmpeg  -hide_banner -nostats -loglevel error -n -i {f1} -codec:a mp3 -ar 32000 {f2}mp3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=False, default=None,
                        help='Path of folder containing the wave files, expected format to be as downloaded '
                             'using the PANNs script, containing balanced_train_segments, eval_segments, '
                             'unbalanced_train_segments folders.')
    parser.add_argument('--out', type=str, required=False, default=None,
                        help='Directory to save out the converted mp3s.')

    args = parser.parse_args()

    source_path = args.source or source_path
    out_path = args.out or out_path
    folders = ['balanced_train_segments', 'eval_segments'] + ["unbalanced_train_segments/" + x for x in sorted(
        os.listdir(source_path + "unbalanced_train_segments/"))]

    print("I will work on these folders:")
    print(folders)

    for fol in folders:
        process_folder(fol)
