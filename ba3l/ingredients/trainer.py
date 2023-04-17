import inspect
import os

from ba3l.util.functions import get_default_kwargs_dict
from .datasets import Datasets, raise_
from .models import Models
from sacred import Ingredient

from typing import Sequence, Optional, List

from sacred.utils import PathType, optional_kwargs_decorator


class Trainer(Ingredient):
    """
    The class that annotates the main Trainer of Ba3l experiment


    """

    TRAINER_STRING_PREFIX = "get_trainer"
