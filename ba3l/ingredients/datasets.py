import inspect
import os
from functools import partial

from ba3l.util.functions import get_default_kwargs_dict
from sacred.config import CMD
from .ingredient import Ingredient

from typing import Sequence, Optional, List

from sacred.utils import PathType, optional_kwargs_decorator
from munch import DefaultFactoryMunch, Munch

def raise_(ex):
    raise ex


class Dataset(Ingredient):
    """
    The class that annotates a Dateset of Ba3l experiment
    a Dataset can be


    """

    DATASET_STRING_PREFIX = "get_dataset"
    ITER_STRING_PREFIX = "get_iterator"

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
            name = "dataset"
        self.name = name.rsplit(".", 1)[-1]
        super().__init__(
            path=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            _caller_globals=caller_globals,
            save_git_info=save_git_info,
        )

        self.get_dataset_command = None
        self.get_dataset_iterator_command = None
        self.current_run = None
        self.get_dataset = lambda: raise_(
            NotImplementedError(
                "Use dataset.dataset_name.dataset to annotate the  "
                "get_dataset function!."
            )
        )
        self.get_iter = lambda: raise_(
            NotImplementedError(
                "Use dataset.dataset_name.iter to annotate the  " "get_iter function!."
            )
        )

    @optional_kwargs_decorator
    def dataset(
        self, function=None, prefix=None, unobserved=False, static_args={}, **extra_args
    ):
        """
        Decorator to define a new Dataset.

        The name of the dataset is used to get an instance of the dataset, it will register a command

        Datasets are sacred commands.

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
        self.commands[Dataset.DATASET_STRING_PREFIX] = captured_f
        self.get_dataset = captured_f
        self.add_config(dataset=CMD("get_dataset"))
        return captured_f

    @optional_kwargs_decorator
    def iter(
        self, function=None, prefix=None, unobserved=False, static_args={}, **extra_args
    ):
        """
        Decorator to define a new Iterator.

        The name of the iterator is used to get an instance of the iterator, it will register a command

        iterator are sacred commands.

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
        """
        self.add_default_args_config(function, prefix, extra_args, static_args=static_args)

        captured_f = self.capture(function, prefix=prefix, static_args=static_args)
        captured_f.unobserved = unobserved
        self.commands[Dataset.ITER_STRING_PREFIX] = captured_f
        self.get_iter = captured_f
        return captured_f

    # def get_dataset(self):
    #     assert self.current_run is not None, "Can only be called during a run."
    #     return self.commands[Datasets.DATASET_STRING_PREFIX + name]()
    #     # return self.current_run.get_command_function(
    #     #     self.path + "." + Datasets.DATASET_STRING_PREFIX + name)()
    #     #

    def __getattr__(self, k):
        if k == "iterator":
            return self.__getattribute__("iter")
        if k == "get_iterator":
            return self.__getattribute__("get_iter")
        super().__getattribute__(k)
        # @todo maybe run commands from here after running


class Datasets(Ingredient, Munch):
    """
    The class that encapsulates all the datasets in an experiment


    """

    __instance = None

    @classmethod
    def get_instance(cls):
        if Datasets.__instance is None:
            Datasets.__instance = Datasets()
        return Datasets.__instance

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
            name = "datasets"

        Ingredient.__init__(
            self,
            path=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            _caller_globals=caller_globals,
            save_git_info=save_git_info,
        )

        self.get_datasets_list_command = None
        self.get_dataset_command = None
        self.get_dataset_iterator_command = None
        self.current_run = None
        self.get_dataset = None

        # self.command(get_dataset_iterator_command, unobserved=True)

    def __getattr__(self, k):
        """ Gets key if it exists, otherwise returns the default value."""
        try:
            return Munch.__getattr__(self, k)
        except AttributeError:
            return self.__getitem__(k)

    def __setattr__(self, k, v):
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def __getitem__(self, k):
        """ Gets key if it exists, otherwise returns the default value."""
        try:
            return Munch.__getitem__(self, k)
        except KeyError:
            self[k] = Dataset(
                self.path + "." + k,
                base_dir=self.base_dir,
                save_git_info=self.save_git_info,
            )
            assert self
            self.ingredients.append(self[k])
            return self[k]

    def __hash__(self):
        return Ingredient.__hash__(self)

    def get_datasets(self, config_conditions={}, return_datasets_names=False):
        """ Return all the datasets whose configuration matches config_conditions."""
        results = []
        for dataset in self.ingredients:
            all_ok = True
            for cond_k, cond_v in config_conditions.items():
                if (
                    self.current_run.get_config_path_value(dataset.path + "." + cond_k)
                    != cond_v
                ):
                    all_ok = False
                    break
            if all_ok:
                if return_datasets_names:
                    results.append((dataset))
                results.append(dataset)
        return results
