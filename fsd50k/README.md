# Experiments on FSD50K
 The FSD50K dataset ([Zenodo](https://zenodo.org/record/4060432))  consists of 51K audio clips annotated
with 200 sound event classes taken from the Audioset ontology. The dataset contains 100 hours of audio and is the
second largest publicly available general purpose sound event
recognition dataset after Audioset. Furthermore, the FSD50K
evaluation set is of high quality, with each evaluation label being double-checked and assessed by two to five independent annotators 

# Setup
1. Download the dataset from [Zenodo](https://zenodo.org/record/4060432) and unzip it.
2. Convert wav files to mp3s:
```shell
cd fsd50k/prepare_scripts/

python convert_to_mp3.py path/to/fsd50k
 ```
this will create a folder inside the FSD50K directory with the mp3 files.
3. Pack the mp3 to HDF5 files:
```shell
cd fsd50k/prepare_scripts/
python create_h5pymp3_dataset.py path/to/fsd50k
 ```
Now you should have inside `../../audioset_hdf5s/mp3/` three new files: `FSD50K.eval_mp3.hdf`, `FSD50K.val_mp3.hdf`, `FSD50K.train_mp3.hdf`.


# Runing Experiments

Similar to the runs on Audioset, PaSST-S:

```shell
# Example call with all the default config:
python ex_fsd50k.py with  trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "FSD50k PaSST-S"
```

```shell
# Example call without overlap:
python ex_fsd50k.py with  passt_s_swa_p16_s16_128_ap473 models.net.s_patchout_t=10  models.net.s_patchout_f=1 trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "FSD50k PaSST-S"
```

# Runing Experiments

