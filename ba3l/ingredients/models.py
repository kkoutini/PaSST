import inspect
import os

from .ingredient import Ingredient

from typing import Sequence, Optional, List

from sacred.utils import PathType

import inspect
import os
from functools import partial

from ba3l.util.functions import get_default_kwargs_dict
from sacred.config import CMD
from .ingredient import Ingredient

from typing import Sequence, Optional, List

from sacred.utils import PathType, optional_kwargs_decorator
from munch import DefaultFactoryMunch, Munch


class Models(Ingredient):
    """
    The class that annotates the models of Ba3l experiment


    """

    __instance = None

    @classmethod
    def get_instance(cls):
        if Models.__instance is None:
            Models.__instance = Models()
        return Models.__instance

    def __init__(
        self,
        name: Optional[str] = None,
        ingredients: Sequence[Ingredient] = (),
        interactive: bool = False,
        base_dir: Optional[PathType] = None,
        save_git_info: bool = True,
    ):
        """
        Create a new experiment with the given name and optional ingredients.

        Parameters
        ----------
        name
            Optional name of this experiment, defaults to the filename.
            (Required in interactive mode)

        ingredients : list[sacred.Ingredient], optional
            A list of ingredients to be used with this experiment.

        interactive
            If set to True will allow the experiment to be run in interactive
            mode (e.g. IPython or Jupyter notebooks).
            However, this mode is discouraged since it won't allow storing the
            source-code or reliable reproduction of the runs.

        base_dir
            Optional full path to the base directory of this experiment. This
            will set the scope for automatic source file discovery.

        additional_host_info
            Optional dictionary containing as keys the names of the pieces of
            host info you want to collect, and as
            values the functions collecting those pieces of information.

        save_git_info:
            Optionally save the git commit hash and the git state
            (clean or dirty) for all source files. This requires the GitPython
            package.
        """

        caller_globals = inspect.stack()[1][0].f_globals
        if name is None:
            name = "models"

        super().__init__(
            path=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            _caller_globals=caller_globals,
            save_git_info=save_git_info,
        )

        self.get_models_command = None
        self.current_run = None
        # self.command(print_config, unobserved=True)


def raise_(ex):
    raise ex


class Model(Ingredient):
    """
    The class that annotates a Dateset of Ba3l experiment
    a Dataset can be


    """

    MODEL_STRING_PREFIX = "get_instance"

    def __init__(
        self,
        name: str,
        ingredients: Sequence[Ingredient] = (),
        interactive: bool = False,
        base_dir: Optional[PathType] = None,
        save_git_info: bool = True,
    ):
        """
        Create a new experiment with the given name and optional ingredients.

        Parameters
        ----------
        name
            Optional name of this experiment, defaults to the filename.
            (Required in interactive mode)

        ingredients : list[sacred.Ingredient], optional
            A list of ingredients to be used with this experiment.

        interactive
            If set to True will allow the experiment to be run in interactive
            mode (e.g. IPython or Jupyter notebooks).
            However, this mode is discouraged since it won't allow storing the
            source-code or reliable reproduction of the runs.

        base_dir
            Optional full path to the base directory of this experiment. This
            will set the scope for automatic source file discovery.

        additional_host_info
            Optional dictionary containing as keys the names of the pieces of
            host info you want to collect, and as
            values the functions collecting those pieces of information.

        save_git_info:
            Optionally save the git commit hash and the git state
            (clean or dirty) for all source files. This requires the GitPython
            package.
        """

        caller_globals = inspect.stack()[1][0].f_globals
        if name is None:
            name = "model"
        self.name = name.rsplit(".", 1)[-1]
        super().__init__(
            path=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            _caller_globals=caller_globals,
            save_git_info=save_git_info,
        )

        self.get_instance_command = None
        self.current_run = None
        self.get_instance = lambda: raise_(
            NotImplementedError(
                "Use dataset.dataset_name.dataset to annotate the  "
                "get_dataset function!."
            )
        )

    @optional_kwargs_decorator
    def instance(
        self, function=None, prefix=None, unobserved=False, static_args={}, **extra_args
    ):
        """
        Decorator to define a new model.

        The name of the model is used to get an instance of the model, it will register a command


        The command can be given a prefix, to restrict its configuration space
        to a subtree. (see ``capture`` for more information)

        A command can be made unobserved (i.e. ignoring all observers) by
        passing the unobserved=True keyword argument.
        :param function: the function to return a Dataset Object
        :param prefix: sacred configuration prefix
        :param unobserved: sacred unobserved
        :param static_args: static Args to be passed to the function, these arg need not to be serlizable and
         are not stored in the config
        :param extra_args: explicit arguments to be add to the config, you can these to override the function default
        values, for example wraping a config with CMD, then the parameter will be filled with excuting the command
        specified by CMD string value. CMD string have special context
        :return:


        """
        self.add_default_args_config(function, prefix, extra_args, static_args=static_args)
        captured_f = self.capture(function, prefix=prefix, static_args=static_args)
        captured_f.unobserved = unobserved
        self.commands[Model.MODEL_STRING_PREFIX] = captured_f
        self.get_instance = captured_f
        self.add_config(get_instance=CMD("get_instance"))
        return captured_f

    def __getattr__(self, k):
        if k == "get_instance":
            return self.__getattribute__("get_instance")
        super().__getattribute__(k)
        # @todo maybe run commands from here after running
