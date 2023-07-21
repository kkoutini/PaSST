import inspect
from importlib import import_module

from ba3l.ingredients.datasets import Datasets
from ba3l.ingredients.models import Models, Model
#from ba3l.trainer import Trainer
from ba3l.util.sacred_logger import SacredLogger
from sacred import Experiment as Sacred_Experiment, Ingredient
from typing import Sequence, Optional, List

from sacred.commandline_options import CLIOption
from sacred.config import CMD
from sacred.host_info import HostInfoGetter
from sacred.utils import PathType, optional_kwargs_decorator
from pytorch_lightning import loggers as pl_loggers
from ba3l.util.functions import get_default_kwargs_dict


def ingredients_recursive_apply(ing, fn):
    fn(ing)
    for kid in ing.ingredients:
        ingredients_recursive_apply(kid, fn)

def config_recursive_apply(conf, fn):
    for k,v in conf.items():
        if isinstance(v, dict):
            config_recursive_apply(v,fn)
        else:
            fn(k,v)


def get_loggers(use_tensorboard_logger=False, use_sacred_logger=False):
    loggers = []
    if use_sacred_logger:
        loggers.append( SacredLogger(expr))
    if use_tensorboard_logger:
        loggers.append(pl_loggers.TensorBoardLogger(sacred_logger.name))
    
    return loggers



class Experiment(Sacred_Experiment):
    """
    Main Ba3l Experiment class overrides sacred experiments.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        ingredients: Sequence[Ingredient] = (),
        datasets: Optional[Ingredient] = None,
        models: Optional[Ingredient] = None,
        interactive: bool = False,
        base_dir: Optional[PathType] = None,
        additional_host_info: Optional[List[HostInfoGetter]] = None,
        additional_cli_options: Optional[Sequence[CLIOption]] = None,
        save_git_info: bool = True,
    ):
        """
        Create a new experiment with the given name and optional ingredients. (from Sacred)


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
        if models is None:
            models = Models.get_instance()
        self.models = models
        if datasets is None:
            datasets = Datasets.get_instance()
        self.datasets = datasets
        if ingredients is None:
            ingredients = []
        ingredients = list(ingredients) + [models, datasets]
        caller_globals = inspect.stack()[1][0].f_globals
        self.last_default_configuration_position = 0
        super().__init__(
            name=name,
            ingredients=ingredients,
            interactive=interactive,
            base_dir=base_dir,
            additional_host_info=additional_host_info,
            additional_cli_options=additional_cli_options,
            save_git_info=save_git_info,
            caller_globals=caller_globals
        )


    def get_run_identifier(self):
        return str(self.current_run.db_identifier) \
            + "_" + str(self.current_run._id)


    def get_dataloaders(self, filter={}):
        results = {}
        for ds in self.datasets.get_datasets(filter):
            results[ds.name] = ds.get_iterator()
        if len(results) == 1:
            for k, v in results.items():
                return v
        return results

    def get_train_dataloaders(self):
        return self.get_dataloaders(dict(train=True))

    def get_val_dataloaders(self):
        return self.get_dataloaders(dict(validate=True))

    def _create_run(
        self,
        command_name=None,
        config_updates=None,
        named_configs=(),
        info=None,
        meta_info=None,
        options=None,
        dry_run=False,
    ):
        if self.current_run is not None:
            # @todo replace with logger
            print("Warning: multiple runs are not yet supported")


        run = super()._create_run(
            command_name,
            config_updates,
            named_configs,
            info,
            meta_info,
            options,
            dry_run=False,
        )
        # self.current_run=run
        # def update_current_run(ing):
        #     ing.current_run = run
        #
        # ingredients_recursive_apply(self, update_current_run)

        return run

    @optional_kwargs_decorator
    def command(
            self, function=None, prefix=None, unobserved=False, add_default_args_config=True, static_args={},
            **extra_args
    ):
        """
        Decorator to define a new Command.

        a command is a function whose parameters are filled automatically by sacred.

        The command can be given a prefix, to restrict its configuration space
        to a subtree. (see ``capture`` for more information)

        A command can be made unobserved (i.e. ignoring all observers) by
        passing the unobserved=True keyword argument.
        :param add_default_args_config: wether to add the default arguments of the function to the config automatically.
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
        add_default_args_config = (not unobserved) and add_default_args_config
        if add_default_args_config:
            self.add_default_args_config(function, prefix, extra_args, static_args=static_args)
        captured_f = self.capture(function, prefix=prefix, static_args=static_args)
        captured_f.unobserved = unobserved
        self.commands[function.__name__] = captured_f
        return captured_f


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
        # respect the prefix for the added default parameters
        if prefix is not None:
            for pr in prefix.split('.')[::-1]:
                config_candidate={pr: config_candidate}

        self.configurations.insert(self.last_default_configuration_position, self._create_config_dict(config_candidate, None))
        self.last_default_configuration_position += 1
