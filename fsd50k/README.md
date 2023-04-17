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
python ex_fsd50k.py with  trainer.precision=16  -p
```

```shell
# Example call without overlap:
python ex_fsd50k.py with  passt_s_swa_p16_s16_128_ap473 models.net.s_patchout_t=10  models.net.s_patchout_f=1 trainer.precision=16  -p
```


# Pre-trained models

Pre-trained models on FSD50K can be found here [here](https://github.com/kkoutini/PaSST/releases/tag/v0.0.5). 

In order to use the pre-trained models, for fine-tuning or inference, using a minimal dependencies, refer to the [PaSST-HEAR](https://github.com/kkoutini/passt_hear21), as an example after installing passt_hear21 :

```python
from hear21passt.base import get_basic_model,get_model_passt
import torch
# model wrapper, includes Melspectrogram and the default pre-trained transformer
model = get_basic_model(mode="logits")
# replace the transformer with one that outputs 200 classes
model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476",  n_classes=200)

# load the pre-trained model state dict with mAP of .655 on FSD50K
state_dict = torch.load('/home/khaled/fsd50k-passt-s-f128-p16-s10-ap.655.pt')
# load the weights into the transformer
model.net.load_state_dict(state_dict)

# example inference
model.eval()
model = model.cuda()
with torch.no_grad():
    # audio_wave has the shape of [batch, seconds*32000] sampling rate is 32k
    logits=model(audio_wave) 
```


Using the model with no patch overlap PaSST-S-N `fsd50k-passt-s-n-f128-p16-s16-ap.642.pt`:
```python
# replace the transformer with one that outputs 200 classes
model.net = get_model_passt(arch="passt_s_p16_s16_128_ap468", fstride=16,
                                     tstride=16,  n_classes=200)

# load the pre-trained model state dict with mAP of .642 on FSD50K with no patch overlap
state_dict = torch.load('/home/khaled/fsd50k-passt-s-n-f128-p16-s16-ap.642.pt')
# load the weights into the transformer
model.net.load_state_dict(state_dict)

```