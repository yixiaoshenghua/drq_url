"""A `dowel.logger.LogOutput` for tensorboard.

It receives the input data stream from `dowel.logger`, then add them to
tensorboard summary operations through PyTorch's TensorBoard.

Note:
TensorBoard does not support logging parametric distributions natively.
We add this feature by sampling data from a `scipy.stats` distribution.
"""
import functools
import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from torch.utils.tensorboard import SummaryWriter
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from dowel import Histogram
from dowel import LoggerWarning
from dowel import LogOutput
from dowel import TabularInput
from dowel.utils import colorize


class TensorBoardOutput(LogOutput):
    """TensorBoard output for logger.

    Args:
        log_dir(str): The save location of the tensorboard event files.
        x_axis(str): The name of data used as x-axis for scalar tabular.
            If None, x-axis will be the number of dump() is called.
        additional_x_axes(list[str]): Names of data to used be as additional
            x-axes.
        flush_secs(int): How often, in seconds, to flush the added summaries
            and events to disk.
        histogram_samples(int): Number of samples to generate when logging
            random distribution.
    """

    def __init__(self,
                 log_dir,
                 x_axis=None,
                 additional_x_axes=None,
                 flush_secs=120,
                 histogram_samples=1e3):
        additional_x_axes = additional_x_axes or []

        # Create TensorBoard writer with PyTorch's SummaryWriter
        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self._x_axis = x_axis
        self._additional_x_axes = additional_x_axes
        self._default_step = 0
        self._histogram_samples = int(histogram_samples)
        self._waiting_for_dump = []
        

        self._warned_once = set()
        self._disable_warnings = False

    @property
    def types_accepted(self):
        """Return the types that the logger may pass to this output."""
        return (TabularInput,)

    def record(self, data, prefix=''):
        """Add data to tensorboard summary.

        Args:
            data: The data to be logged by the output.
            prefix(str): A prefix placed before a log entry in text outputs.

        """
        if isinstance(data, TabularInput):
            self._waiting_for_dump.append(
                functools.partial(self._record_tabular, data))
        else:
            raise ValueError('Unacceptable type.')

    def _record_tabular(self, data, step):
        if self._x_axis:
            nonexist_axes = []
            for axis in [self._x_axis] + self._additional_x_axes:
                if axis not in data.as_dict:
                    nonexist_axes.append(axis)
            if nonexist_axes:
                self._warn('{} {} exist in the tabular data.'.format(
                    ', '.join(nonexist_axes),
                    'do not' if len(nonexist_axes) > 1 else 'does not'))

        for key, value in data.as_dict.items():
            if isinstance(value,
                          np.ScalarType) and self._x_axis in data.as_dict:
                if self._x_axis is not key:
                    x = data.as_dict[self._x_axis]
                    self._record_kv(key, value, x)

                for axis in self._additional_x_axes:
                    if key is not axis and axis in data.as_dict:
                        x = data.as_dict[axis]
                        self._record_kv(f'{key}/{axis}', value, x)
            else:
                self._record_kv(key, value, step)
            data.mark(key)

    def _record_kv(self, key, value, step):
        if isinstance(value, str):
            self._writer.add_text(key, value, step)
        elif isinstance(value, np.ScalarType):
            self._writer.add_scalar(key, value, step)
        elif isinstance(value, plt.Figure):
            self._writer.add_figure(key, value, step)
        elif isinstance(value, np.ndarray) and value.ndim == 5:
            # PyTorch requires [batch, time, channels, height, width]
            self._writer.add_video(key, value, step, fps=15)
        elif isinstance(value, (rv_frozen, multi_rv_frozen)):
            shape = (self._histogram_samples,) + value.mean().shape
            samples = value.rvs(shape)
            self._writer.add_histogram(key, samples, step)
        elif isinstance(value, Histogram):
            self._writer.add_histogram(key, np.asarray(value), step)

    def dump(self, step=None):
        """Flush summary writer to disk."""
        current_step = step or self._default_step
        
        # Process all waiting tabular inputs
        for p in self._waiting_for_dump:
            p(current_step)
        self._waiting_for_dump.clear()

        # Flush TensorBoard writer
        self._writer.flush()
        
        # Auto-increment default step if not provided
        if step is None:
            self._default_step += 1

    def close(self):
        """Flush all the events to disk and close the file."""
        self._writer.close()

    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        if not self._disable_warnings and msg not in self._warned_once:
            warnings.warn(
                colorize(msg, 'yellow'), NonexistentAxesWarning, stacklevel=3)
        self._warned_once.add(msg)
        return msg


class NonexistentAxesWarning(LoggerWarning):
    """Raise when the specified x axes do not exist in the tabular."""