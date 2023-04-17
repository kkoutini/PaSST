# Experiments on OpenMIC-2018

[OpenMIC-2018](https://github.com/cosmir/openmic-2018) ([zenodo](https://zenodo.org/record/1432913#.W6dPeJNKjOR)) is a dataset for polyphonic instruments identification.



## Preparing the dataset
Use `openmic2008/prepare_scripts/download_preprocess.py` to download and pack the dataset:
```shell
cd openmic2008/prepare_scripts/
python download_preprocess.py
```
When the script completes, you should have two files inside `audioset_hdf5s/mp3/`
 `openmic_train.csv_mp3.hdf` and `openmic_test.csv_mp3.hdf`
these files contains the mp3s of the dataset and the labels.



## Fine-tuning pretrained PaSST on the openmic2008

Similar to audioset you can use:
```shell
# Example call with all the default config:
python ex_openmic.py with  trainer.precision=16  -p 
```

```shell
# with 2 gpus:
DDP=2 python ex_openmic.py with  trainer.precision=16  -p 
```
