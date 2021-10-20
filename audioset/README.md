# Experiments on Audioset
Audioset has around 2M segments. The total size of the dataset with wav files with `32khz` sampling rate is around 1.2 TB. In our setup, this results in a huge IO bottleneck that slows down the training process significantly.
Therefore, we encode the dataset to mp3, pack the mp3 into HDF5 format and decode the mp3s on the fly, If you have enough cpu cores (10-16 dataloading workers) you should not notice any slowdowns.

in the `dataset.py` file we read the samples from the hdf files. Decode the mp3, do wave form augmentations and return the raw waveform of the model.
`AudioSetDataset` is the main class where reading from the hdf files.



## Preparing the dataset
###  Downloading Audioset
We used the scripts provided by [PANNS](https://github.com/qiuqiangkong/audioset_tagging_cnn) to download the dataset.
###  Converting to mp3
Once the Datasets are downloaded we convert all the files to mp3 using the script:
`prepare_scripts/convert_to_mp3.py`.

```bash
python convert_to_mp3.py --source pann_download_folder --out mp3_folder
```

this will significantly reduce the size of the dataset and overcome the IO bottleneck in our setup. The trade-off is that more cpu is needed during training to decode the mp3s. 
We use the [av](https://pypi.org/project/av/) (check `decode_mp3` in ` dataset.py`) library to decode the mp3 in the data loading workers, this is much faster than calling ffmpeg.
As a result, approximetly 10 decoding threads should be enough keep a 2080ti busy.

you can test how much time it take to load and decode one epoch on your system:
```bash
python python ex_audioset.py test_loaders_train_speed
```

This step is not necessary if you have a more powerful setup and the `decode_mp3` also supports other ffmpeg codecs.

###  packing to HDF5 files

Finally, you need to pack the mp3 files into a single HDF5 file using `create_h5pymp3_dataset.py`.
you just need to set the paths in the script to match your local paths. The script goes through the csv files and check if the corresponding mp3 file exists, then it will store it in h5py file.
The output of this step should be 3 files `balanced_train_segments_mp3.hdf`, `eval_segments_mp3.hdf` and `unbalanced_train_segments_mp3.hdf`.
Each of these files. Make sure the paths match the default config in `dataset.py`