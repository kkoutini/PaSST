import pytorch_lightning as pl

import warnings
from abc import ABC

import torch.distributed as dist
from munch import DefaultMunch

try:
    # loading for pyTorch 1.3
    from torch.utils.data import IterableDataset
except ImportError:
    # loading for pyTorch 1.1
    import torch

    warnings.warn(
        "Your version of pyTorch %s does not support `IterableDataset`,"
        " please upgrade to 1.2+" % torch.__version__,
        ImportWarning,
    )
    EXIST_ITER_DATASET = False
else:
    EXIST_ITER_DATASET = True

try:
    from apex import amp

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


class Ba3lModule(pl.LightningModule):
    def __init__(self, experiment):
        super(Ba3lModule, self).__init__()
        self.experiment = experiment
        self.run =  experiment.current_run
        self.config = DefaultMunch.fromDict(experiment.current_run.config)
        for key,model in experiment.current_run.config['models'].items():
            setattr(self, key, experiment.current_run.get_command_function("models."+key+"."+model['instance_cmd'])())
        self.save_hyperparameters(self.config)
        

