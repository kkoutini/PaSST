from pytorch_lightning.utilities import rank_zero_only
try:
    from pytorch_lightning.loggers import Logger as LightningLoggerBase
    from pytorch_lightning.loggers.logger import rank_zero_experiment
except ImportError:
    from pytorch_lightning.loggers import LightningLoggerBase
    from pytorch_lightning.loggers.base import rank_zero_experiment

from warnings import warn


from logging import getLogger

try:
    import sacred
except ImportError:
    raise ImportError("Missing sacred package.  Run `pip install sacred`")


logger = getLogger(__name__)


class SacredLogger(LightningLoggerBase):
    def __init__(self, sacred_experiment):
        """Initialize a sacred logger.
        :param sacred.experiment.Experiment sacred_experiment: Required. Experiment object with desired observers
        already appended.
        source: https://github.com/expectopatronum/pytorch-lightning/blob/9fcb238ec03e3f0b0378fd058119f1563a11650c/pytorch_lightning/logging/sacred.py
        """
        super().__init__()
        self.sacred_experiment = sacred_experiment
        self.experiment_name = sacred_experiment.path
        self._run_id = None
        warn('SacredLogger is deprecated', DeprecationWarning, stacklevel=2)

    @property
    def experiment(self):
        return self.sacred_experiment

    @property
    def run_id(self):
        if self._run_id is not None:
            return self._run_id
        self._run_id = self.sacred_experiment.get_run_identifier()
        return self._run_id

    @rank_zero_only
    def log_hyperparams(self, params):
        # probably not needed bc. it is dealt with by sacred
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(f"Discarding metric with string value {k}={v}")
                continue
            self.experiment.log_scalar(k, v, step)

    @property
    def name(self):
        return self.experiment_name

    @property
    def version(self):
        return self.run_id

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
