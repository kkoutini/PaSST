# PaSST: Efficient Training of Audio Transformers with Patchout


This is the implementation for [Efficient Training of Audio Transformers with Patchout](https://arxiv.org/abs/2110.05069)

Patchout significantly reduces the training time and GPU memory requirements to train transformers on audio spectrograms, while improving their performance.

<p align="center"><img src="https://github.com/kkoutini/PaSST/blob/main/.github/speed_mem_map.png?raw=true" width="600"/></p>

Patchout works by dropping out some of the input patches during training. In either a unstructured way (randomly, similar to dropout), or entire time-frames or frequency bins of the extracted patches (similar to SpecAugment), which corresponds to rows/columns in step 3 of the figure below.  

![PaSST architecture](.github/passt_diag.png)


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

# Training on Audioset
Download and prepare the dataset as explained in the [audioset page](audioset/)
The base PaSST model can be trained for example like this:
```bash
python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p -m mongodb_server:27000:audioset21_balanced -c "PaSST base"
```
you can override any of the configuration using the [sacred syntax](https://sacred.readthedocs.io/en/stable/command_line.html).
In order to see the available options either use [omniboard](https://github.com/vivekratnavel/omniboard) or use:
```shell
 python ex_audioset.py print_config
 ```
In short:
 - All the configuration options under `trainer` are pytorch lightning trainer [api](https://pytorch-lightning.readthedocs.io/en/1.4.1/common/trainer.html#trainer-class-api). 
 - `models.net` are the passt options.
 - `models.mel` are the preprocessing options.

For example using only unstructured patchout of 400:
```bash
python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384  models.net.u_patchout=400  models.net.s_patchout_f=0 models.net.s_patchout_t=0 -p -m mongodb_server:27000:audioset21_balanced -c "Unstructured PaSST base"
```

Multi-gpu training can be enabled by setting the environment variable `DDP`, for example with 2 gpus:

```shell
 DDP=2 python ex_audioset.py with trainer.precision=16  models.net.arch=passt_deit_bd_p16_384 -p -m mongodb_server:27000:audioset21_balanced -c "PaSST base 2 GPU"
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
