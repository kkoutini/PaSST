
import importlib
import io
import os
import sys

# check if ipywidgets is installed before importing tqdm.auto
# to ensure it won't fail and a progress bar is displayed
from typing import Optional, Union


from pytorch_lightning.callbacks import Callback,ProgressBarBase , ProgressBar as PlProgressBar
from pytorch_lightning.callbacks.progress import tqdm



class ProgressBar(PlProgressBar):
    r"""
    This is the default progress bar used by Lightning. It prints to `stdout` using the
    :mod:`tqdm` package and shows up to four different bars:

    - **sanity check progress:** the progress during the sanity check run
    - **main progress:** shows training + validation progress combined. It also accounts for
      multiple validation runs during training when
      :paramref:`~pytorch_lightning.trainer.trainer.Trainer.val_check_interval` is used.
    - **validation progress:** only visible during validation;
      shows total progress over all validation datasets.
    - **test progress:** only active when testing; shows total progress over all test datasets.

    For infinite datasets, the progress bar never ends.

    If you want to customize the default ``tqdm`` progress bars used by Lightning, you can override
    specific methods of the callback class and pass your custom implementation to the
    :class:`~pytorch_lightning.trainer.trainer.Trainer`:

    Example::

        class LitProgressBar(ProgressBar):

            def init_validation_tqdm(self):
                bar = super().init_validation_tqdm()
                bar.set_description('running validation ...')
                return bar

        bar = LitProgressBar()
        trainer = Trainer(callbacks=[bar])

    Args:
        refresh_rate:
            Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display. By default, the
            :class:`~pytorch_lightning.trainer.trainer.Trainer` uses this implementation of the progress
            bar and sets the refresh rate to the value provided to the
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.progress_bar_refresh_rate` argument in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.
        process_position:
            Set this to a value greater than ``0`` to offset the progress bars by this many lines.
            This is useful when you have progress bars defined elsewhere and want to show all of them
            together. This corresponds to
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.process_position` in the
            :class:`~pytorch_lightning.trainer.trainer.Trainer`.

    """


    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        has_main_bar = False
        bar = tqdm(
            desc='Validating',
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout
        )
        return bar



    def on_epoch_start(self, trainer, pl_module):
        ProgressBarBase.on_epoch_start(self, trainer, pl_module)
        self.main_progress_bar = self.init_train_tqdm()
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float('inf'):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_val_batches = 0
        total_batches = total_train_batches + total_val_batches
        reset(self.main_progress_bar, total_batches)
        self.main_progress_bar.set_description(f'Epoch {trainer.current_epoch}')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        ProgressBarBase.on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.train_batch_idx, self.total_train_batches):
            self._update_bar(self.main_progress_bar)
            self.main_progress_bar.set_postfix(trainer.progress_bar_dict)

    def on_validation_start(self, trainer, pl_module):
        ProgressBarBase.on_validation_start(self, trainer, pl_module)
        if trainer.sanity_checking:
            reset(self.val_progress_bar, sum(trainer.num_sanity_val_batches))
        else:
            if self.main_progress_bar is not None:
                self.main_progress_bar.close()
            self.val_progress_bar = self.init_validation_tqdm()
            reset(self.val_progress_bar, self.total_val_batches)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        ProgressBarBase.on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        if self._should_update(self.val_batch_idx, self.total_val_batches):
            self._update_bar(self.val_progress_bar)
            #self._update_bar(self.main_progress_bar)

    def on_validation_end(self, trainer, pl_module):
        ProgressBarBase.on_validation_end(self, trainer, pl_module)
        if self.main_progress_bar is not None:
            self.main_progress_bar.set_postfix(trainer.progress_bar_dict)
        self.val_progress_bar.close()



def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """ The tqdm doesn't support inf values. We have to convert it to None. """
    if x == float('inf'):
        return None
    return x


def reset(bar: tqdm, total: Optional[int] = None) -> None:
    """ Resets the tqdm bar to 0 progress with a new total, unless it is disabled. """
    if not bar.disable:
        bar.reset(total=convert_inf(total))
