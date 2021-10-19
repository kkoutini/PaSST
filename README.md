# PaSST: Efficient Training of Audio Transformers with Patchout


This is the implementation for [Efficient Training of Audio Transformers with Patchout](https://arxiv.org/abs/2110.05069)

Patchout significantly reduces the training time and GPU memory requirements to train transformers on audio spectrograms, while improving their performance.

<p align="center"><img src="https://github.com/kkoutini/PaSST/blob/main/figures/speed_mem_map.png?raw=true" alt="Illustration of AST." width="300"/></p>

Patchout works by dropping out some of the input patches during training. In either a unstructured way, or complete time-frames or frequency bins.  

<p align="center"><img src="https://github.com/kkoutini/PaSST/blob/main/figures/passt_diag.png?raw=true" alt="Illustration of AST." width="300"/></p>


# Setting up the experiments environment
This repo uses forked versions of sacred for configuration and logging, and pytorch-lightning for training.

For setting up [Mamba](https://github.com/mamba-org/mamba) is recommended and faster then `conda`:

```shell
conda install mamba -n base -c conda-forge
```
Now you can import the environment from `environment.yml`
```shell
mamba env create -f environment.yml
```
Now you have an environment named `ba3l`. Now install the forked versions of `sacred` and `pl-lightning` and `ba3l`.
```shell
# dependencies
conda activate ba3l
pip install https://github.com/kkoutini/sacred/archive/ba3l.zip
pip install https://github.com/kkoutini/pytorch-lightning/archive/ba3l.zip
pip install https://github.com/kkoutini/ba3l/archive/master.zip

```

In order to check the environment we used in our runs, please check the `environment.yml` and `pip_list.txt` files.
 Which were exported using:
```shell
conda env export --no-builds | grep -v "prefix" > environment.yml
pip list > pip_list.txt
```
# Pre-trained models
Please check the [releases page](releases/), to download pre-trained models. 
In general, you can get a pretrained model on Audioset using 
```python
from models.passt import get_model
model  = get_model(arch="passt_s_swa_p16_128_ap476", pretrained=True, n_classes=527, in_channels=1,
                   fstride=10, tstride=10,input_fdim=128, input_tdim=998,
                   u_patchout=0, s_patchout_t=40, s_patchout_f=4)
```
this will get automatically download pretrained PaSST on audioset with with mAP of ```0.476```. the model was trained with ```s_patchout_t=40, s_patchout_f=4``` but you can change these to better fit your task/ computational needs.

# Contact
The repo will be updated, in the mean time if you have any questions or problems feel free to open an issue on GitHub, or contact the authors directly.
