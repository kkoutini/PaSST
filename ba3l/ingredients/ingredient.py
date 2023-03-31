import inspect
import os
from functools import partial

from ba3l.util.functions import get_default_kwargs_dict
from sacred import Ingredient as sacred_Ingredient

from typing import Sequence, Optional, List

from sacred.utils import PathType, optional_kwargs_decorator
from munch import DefaultFactoryMunch, Munch


def raise_(ex):
    raise ex


class Ingredient(sacred_Ingredient):
    """
    The class that annotates a Dateset of Ba3l experiment
    a Dataset can be


    """

    def __init__(
        self,
        path: str,
        ingredients: Sequence[sacred_Ingredient] = (),
        interactive: bool = False,
        _caller_globals: Optional[dict] = None,
        base_dir: Optional[PathType] = None,
        save_git_info: bool = True,
    ):
        """
        The Base Ingredient of all Ba3l ingredients

         Parameters
         ----------
         path
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

        _caller_globals = _caller_globals or inspect.stack()[1][0].f_globals
        if path is None:
            path = "Ingredient"

        super().__init__(
            path=path,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            _caller_globals=_caller_globals,
            save_git_info=save_git_info,
        )

        self.current_run = None
        self.last_default_configuration_position = 0

    def add_default_args_config(self, function, prefix, extra_args={}, static_args={}):
        """
        adds the default parameters of a function to the ingredient config at lowest priority!
        Default args config is meant remove the need to declare all the configurations manually.
        :param f: the function
        """
        # @todo get the doc of the params as well
        config_candidate = {**get_default_kwargs_dict(function), **extra_args}
        # remove "static_args" from config
        for k in static_args:
            config_candidate.pop(k, None)
        if prefix is not None:
            for pr in prefix.split('.')[::-1]:
                config_candidate={pr: config_candidate}
        self.configurations.insert(self.last_default_configuration_position, self._create_config_dict(config_candidate, None))
        self.last_default_configuration_position += 1

    @optional_kwargs_decorator
    def command(
        self, function=None, prefix=None, unobserved=False, add_default_args_config=True, static_args={}, **extra_args
    ):
        """
        Decorator to define a new Command.

        a command is a function whose parameters are filled automatically by sacred.

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
        if add_default_args_config:
            self.add_default_args_config(function, prefix, extra_args, static_args=static_args)
        captured_f = self.capture(function, prefix=prefix, static_args=static_args)
        captured_f.unobserved = unobserved
        self.commands[function.__name__] = captured_f
        return captured_f
