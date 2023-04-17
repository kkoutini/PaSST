# Experiments on ESC-50 Environmental Sound Classification
[ESC-50](https://github.com/karolpiczak/ESC-50) consist of 2000 5-second recordings to be classified to 50 semantical classes.

## Setting up the fine-tuning experiments
- Download the prerpocessed (resampled) dataset [esc50.zip](https://github.com/kkoutini/PaSST/releases/download/v.0.0.6/esc50.zip) from the [releases page](https://github.com/kkoutini/PaSST/releases/tag/v.0.0.6) and 
unpack the zip file into a directory (the default path is `./audioset_hdf5s/`). The `base_dir` config in the dataset file ([here](https://github.com/kkoutini/PaSST/blob/main/esc50/dataset.py#L35)) should point to the extracted contents of the dataset zip file.
- Running the experiments using the common configurations (similar to Audioset)
```shell
python3 ex_esc50.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5  basedataset.fold=1 -p
```
## Pre-trained models

Pre-trained models on ESC-50 can be found here [here](https://github.com/kkoutini/PaSST/releases/tag/v.0.0.6). 

In order to use the pre-trained models, for fine-tuning or inference, using a minimal dependencies, refer to the [PaSST-HEAR](https://github.com/kkoutini/passt_hear21), as an example after installing passt_hear21 :
```python
from hear21passt.base import get_basic_model,get_model_passt
import torch
# model wrapper, includes Melspectrogram and the default transformer
model = get_basic_model(mode="logits")
# replace the transformer with one that outputs 50 classes
model.net = get_model_passt(arch="passt_s_swa_p16_128_ap476",  n_classes=50)

# load the pre-trained model state dict
state_dict = torch.load('/home/khaled/esc50-passt-s-n-f128-p16-s10-fold1-acc.967.pt')
# load the weights into the transformer
model.net.load_state_dict(state_dict)

# example inference
model.eval()
model = model.cuda()
with torch.no_grad():
    # audio_wave has the shape of [batch, seconds*32000] sampling rate is 32k
    logits=model(audio_wave) 
```

