import math
import os
import random
from collections import deque
import pathlib
import numpy as np
import scipy.linalg as sp_la
import scipy.signal
from matplotlib import figure
from moviepy import editor as mpy
from matplotlib.patches import Ellipse
from sklearn import decomposition
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.independent import Independent
from skimage.util.shape import view_as_windows
from torch import distributions as pyd
from dowel import Histogram
import copy
import sys

_g_session = None
_g_context = {}

class GlobalContext:
    def __init__(self, context):
        self.context = context

    def __enter__(self):
        global _g_context
        self.prev_g_context = _g_context
        _g_context = self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _g_context
        _g_context = self.prev_g_context

def get_metric_prefix():
    global _g_context
    prefix = ''
    if 'phase' in _g_context:
        prefix += _g_context['phase'].capitalize()
    if 'policy' in _g_context:
        prefix += {'sampling': 'Sp', 'skill': 'Op'}.get(
            _g_context['policy'].lower(), _g_context['policy'].lower()).capitalize()

    if len(prefix) == 0:
        return '', ''

    return prefix + '/'


def get_context():
    global _g_context
    return copy.copy(_g_context)

import dowel
dowel_eval = dowel
del sys.modules['dowel']

import dowel
dowel_plot = dowel
del sys.modules['dowel']

def get_dowel(phase=None):
    if (phase or get_context().get('phase')).lower() == 'plot':
        return dowel_plot
    if (phase or get_context().get('phase')).lower() == 'eval':
        return dowel_eval
    return dowel
def get_logger(phase=None):
    return get_dowel(phase).logger
def get_tabular(phase=None):
    return get_dowel(phase).tabular

class FigManager:
    def __init__(self, snapshot_dir, step_itr, label, extensions=None, subplot_spec=None):
        self.snapshot_dir = snapshot_dir
        self.step_itr = step_itr
        self.label = label
        self.fig = figure.Figure()
        if subplot_spec is not None:
            self.ax = self.fig.subplots(*subplot_spec).flatten()
        else:
            self.ax = self.fig.add_subplot()

        if extensions is None:
            self.extensions = ['png']
        else:
            self.extensions = extensions

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plot_paths = [(pathlib.Path(self.snapshot_dir)
                       / 'plots'
                       / f'{self.label}_{self.step_itr}.{extension}') for extension in self.extensions]
        plot_paths[0].parent.mkdir(parents=True, exist_ok=True)
        for plot_path in plot_paths:
            self.fig.savefig(plot_path, dpi=300)
        get_tabular('plot').record(self.label, self.fig)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def discount_cumsum(x, discount):
    """Discounted cumulative sum.

    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.

    Returns:
        np.ndarrary: Discounted cumulative sum.

    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]

def log_performance_ex(itr, batch, discount, additional_records=None, additional_prefix=''):
    """Evaluate the performance of an algorithm on a batch of trajectories.

    Args:
        itr (int): Iteration number.
        batch (TrajectoryBatch): The trajectories to evaluate with.
        discount (float): Discount value, from algorithm's property.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
    if additional_records is None:
        additional_records = {}
    returns = []
    undiscounted_returns = []
    completion = []
    success = []
    for trajectory in batch.split():
        returns.append(discount_cumsum(trajectory.rewards, discount))
        undiscounted_returns.append(sum(trajectory.rewards))
        completion.append(float(trajectory.terminals.any()))
        if 'success' in trajectory.env_infos:
            success.append(float(trajectory.env_infos['success'].any()))

    average_discounted_return = np.mean([rtn[0] for rtn in returns])

    prefix_tabular = get_metric_prefix()
    with get_tabular().prefix(prefix_tabular):
        def _record(key, val, pre=''):
            get_tabular().record(
                    (pre + '/' if len(pre) > 0 else '') + key,
                    val)

        def _record_histogram(key, val):
            get_tabular('plot').record(key, Histogram(val))

        _record('Iteration', itr)
        get_tabular().record('Iteration', itr)
        _record('NumTrajs', len(returns))

        max_undiscounted_returns = np.max(undiscounted_returns)
        min_undiscounted_returns = np.min(undiscounted_returns)
        _record('AverageDiscountedReturn', average_discounted_return)
        _record('AverageReturn', np.mean(undiscounted_returns))
        _record('StdReturn', np.std(undiscounted_returns))
        _record('MaxReturn', max_undiscounted_returns)
        _record('MinReturn', min_undiscounted_returns)
        _record('DiffMaxMinReturn', max_undiscounted_returns - min_undiscounted_returns)
        _record('CompletionRate', np.mean(completion))
        if success:
            _record('SuccessRate', np.mean(success))

        for key, val in additional_records.items():
            is_scalar = True
            try:
                if len(val) > 1:
                    is_scalar = False
            except TypeError:
                pass
            if is_scalar:
                _record(key, val, pre=additional_prefix)
            else:
                _record_histogram(key, val)

    return dict(
        undiscounted_returns=undiscounted_returns,
        discounted_returns=[rtn[0] for rtn in returns],
    )

def get_torch_concat_obs(obs, skill, dim=1):
    concat_obs = torch.cat([obs] + [skill], dim=dim)
    return concat_obs

def get_np_concat_obs(obs, skill):
    concat_obs = np.concatenate([obs] + [skill])
    return concat_obs

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def draw_2d_gaussians(means, stddevs, colors, ax, fill=False, alpha=0.8, use_adaptive_axis=False, draw_unit_gaussian=True, plot_axis=None):
    means = np.clip(means, -1000, 1000)
    stddevs = np.clip(stddevs, -1000, 1000)
    square_axis_limit = 2.0
    if draw_unit_gaussian:
        ellipse = Ellipse(xy=(0, 0), width=2, height=2,
                          edgecolor='r', lw=1, facecolor='none', alpha=0.5)
        ax.add_patch(ellipse)
    for mean, stddev, color in zip(means, stddevs, colors):
        if len(mean) == 1:
            mean = np.concatenate([mean, [0.]])
            stddev = np.concatenate([stddev, [0.1]])
        ellipse = Ellipse(xy=mean, width=stddev[0] * 2, height=stddev[1] * 2,
                          edgecolor=color, lw=1, facecolor='none' if not fill else color, alpha=alpha)
        ax.add_patch(ellipse)
        square_axis_limit = max(
                square_axis_limit,
                np.abs(mean[0] + stddev[0]),
                np.abs(mean[0] - stddev[0]),
                np.abs(mean[1] + stddev[1]),
                np.abs(mean[1] - stddev[1]),
        )
    square_axis_limit = square_axis_limit * 1.2
    ax.axis('scaled')
    if plot_axis is None:
        if use_adaptive_axis:
            ax.set_xlim(-square_axis_limit, square_axis_limit)
            ax.set_ylim(-square_axis_limit, square_axis_limit)
        else:
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
    else:
        ax.axis(plot_axis)

def get_skill_colors(skills, color_range=4):
    num_skills = skills.shape[0]
    dim_skill = skills.shape[1]

    if dim_skill <= 2:
        # Use a predefined skill color scheme
        if dim_skill == 1:
            skills_2d = []
            d = 2.
            for i in range(len(skills)):
                skill = skills[i][0]
                if skill < 0:
                    abs_value = -skill
                    skills_2d.append((d - abs_value * d, d))
                else:
                    abs_value = skill
                    skills_2d.append((d, d - abs_value * d))
            skills = np.array(skills_2d)
        skill_colors = get_2d_colors(skills, (-color_range, -color_range), (color_range, color_range))
    else:
        if dim_skill > 3 and num_skills >= 3:
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_skills = np.vstack((skills, np.random.randn(dim_skill, dim_skill)))
            pca.fit(pca_skills)
            skill_colors = np.array(pca.transform(skills))
        elif dim_skill > 3 and num_skills < 3:
            skill_colors = skills[:, :3]
        elif dim_skill == 3:
            skill_colors = skills

        max_colors = np.array([color_range] * 3)
        min_colors = np.array([-color_range] * 3)
        if all((max_colors - min_colors) > 0):
            skill_colors = (skill_colors - min_colors) / (max_colors - min_colors)
        skill_colors = np.clip(skill_colors, 0, 1)

        skill_colors = np.c_[skill_colors, np.full(len(skill_colors), 0.8)]

    return skill_colors

def get_2d_colors(points, min_point, max_point):
    points = np.array(points)
    min_point = np.array(min_point)
    max_point = np.array(max_point)

    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((
        colors,
        (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
    ))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 0.8)]

    return colors

def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        import warnings
        warnings.warn(
            'Enabeling deterministic mode in PyTorch can have a performance '
            'impact when using GPU.')
        import torch  # pylint: disable=import-outside-toplevel
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def prepare_video(v, n_cols=None):
    orig_ndim = v.ndim
    if orig_ndim == 4:
        v = v[None, ]

    _, t, c, h, w = v.shape

    if v.dtype == np.uint8:
        v = np.float32(v) / 255.

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if n_cols is None:
        if v.shape[0] <= 3:
            n_cols = v.shape[0]
        elif v.shape[0] <= 9:
            n_cols = 3
        else:
            n_cols = 6
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate(
            (v, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = v.reshape((n_rows, n_cols, t, c, h, w))
    v = v.transpose((2, 0, 4, 1, 5, 3))
    v = v.reshape((t, n_rows * h, n_cols * w, c))

    return v


def save_video(snapshot_dir, step_itr, label, tensor, fps=15, n_cols=None):
    def _to_uint8(t):
        # If user passes in uint8, then we don't need to rescale by 255
        if t.dtype != np.uint8:
            t = (t * 255.0).astype(np.uint8)
        return t
    if tensor.dtype in [object]:
        tensor = [_to_uint8(prepare_video(t, n_cols)) for t in tensor]
    else:
        tensor = prepare_video(tensor, n_cols)
        tensor = _to_uint8(tensor)

    # Encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    plot_path = (pathlib.Path(snapshot_dir)
                 / 'plots'
                 # / f'{label}_{step_itr}.gif')
                 / f'{label}_{step_itr}.mp4')
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(str(plot_path), audio=False, verbose=False, logger=None)


def record_video(snapshot_dir, step_itr, label, trajectories, n_cols=None, skip_frames=1, shape=(64, 64)):
    renders = []
    for trajectory in trajectories:
        render = trajectory['observations']
        if render.ndim >= 5:
            render = render.reshape(-1, *render.shape[-3:])
        elif render.ndim == 1:
            render = np.concatenate(render, axis=0)
        renders.append(render)
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        renders[i] = np.concatenate([render, np.zeros((max_length - render.shape[0], *render.shape[1:]), dtype=render.dtype)], axis=0)
        renders[i] = renders[i][::skip_frames]
    renders = np.array(renders)
    renders = renders.reshape((renders.shape[0], renders.shape[1], *shape, -1)).transpose((0, 1, 4, 2, 3)) # (N, T, C, H, W)
    save_video(snapshot_dir, step_itr, label, renders, n_cols=n_cols)



class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


"""A Gaussian distribution with tanh transformation."""

class TanhNormal(torch.distributions.Distribution):
    r"""A distribution induced by applying a tanh transformation to a Gaussian random variable.

    Algorithms like SAC and Pearl use this transformed distribution.
    It can be thought of as a distribution of X where
        :math:`Y ~ \mathcal{N}(\mu, \sigma)`
        :math:`X = tanh(Y)`

    Args:
        loc (torch.Tensor): The mean of this distribution.
        scale (torch.Tensor): The stdev of this distribution.

    """ # noqa: 501

    def __init__(self, loc, scale):
        self._normal = Independent(Normal(loc, scale), 1)
        super().__init__(batch_shape=self._normal.batch_shape,
                         event_shape=self._normal.event_shape,
                         validate_args=False)

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
        """The log likelihood of a sample on the this Tanh Distribution.

        Args:
            value (torch.Tensor): The sample whose loglikelihood is being
                computed.
            pre_tanh_value (torch.Tensor): The value prior to having the tanh
                function applied to it but after it has been sampled from the
                normal distribution.
            epsilon (float): Regularization constant. Making this value larger
                makes the computation more stable but less precise.

        Note:
              when pre_tanh_value is None, an estimate is made of what the
              value is. This leads to a worse estimation of the log_prob.
              If the value being used is collected from functions like
              `sample` and `rsample`, one can instead use functions like
              `sample_return_pre_tanh_value` or
              `rsample_return_pre_tanh_value`


        Returns:
            torch.Tensor: The log likelihood of value on the distribution.

        """
        # pylint: disable=arguments-differ
        if pre_tanh_value is None:
            # Fix in order to TanhNormal.log_prob(1.0) != inf
            pre_tanh_value = torch.log((1 + epsilon + value) / (1 + epsilon - value)) / 2
        norm_lp = self._normal.log_prob(pre_tanh_value)
        ret = (norm_lp - torch.sum(
            torch.log(self._clip_but_pass_gradient((1. - value**2)) + epsilon),
            axis=-1))
        return ret

    def sample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients `do not` pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Sample from this TanhNormal distribution.

        """
        z = self._normal.rsample(sample_shape)
        return torch.tanh(z)

    def rsample_with_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this TanhNormal distribution.

        Returns the sampled value before the tanh transform is applied and the
        sampled value with the tanh transform applied to it.

        Args:
            sample_shape (list): shape of the return.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Samples from this distribution.
            torch.Tensor: Samples from the underlying
                :obj:`torch.distributions.Normal` distribution, prior to being
                transformed with `tanh`.

        """
        z = self._normal.rsample(sample_shape)
        return z, torch.tanh(z)

    def cdf(self, value):
        """Returns the CDF at the value.

        Returns the cumulative density/mass function evaluated at
        `value` on the underlying normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.cdf(value)

    def icdf(self, value):
        """Returns the icdf function evaluated at `value`.

        Returns the icdf function evaluated at `value` on the underlying
        normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.icdf(value)

    @classmethod
    def _from_distribution(cls, new_normal):
        """Construct a new TanhNormal distribution from a normal distribution.

        Args:
            new_normal (Independent(Normal)): underlying normal dist for
                the new TanhNormal distribution.

        Returns:
            TanhNormal: A new distribution whose underlying normal dist
                is new_normal.

        """
        # pylint: disable=protected-access
        new = cls(torch.zeros(1), torch.zeros(1))
        new._normal = new_normal
        return new

    def expand(self, batch_shape, _instance=None):
        """Returns a new TanhNormal distribution.

        (or populates an existing instance provided by a derived class) with
        batch dimensions expanded to `batch_shape`. This method calls
        :class:`~torch.Tensor.expand` on the distribution's parameters. As
        such, this does not allocate new memory for the expanded distribution
        instance. Additionally, this does not repeat any args checking or
        parameter broadcasting in `__init__.py`, when an instance is first
        created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance(instance): new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            Instance: New distribution instance with batch dimensions expanded
            to `batch_size`.

        """
        new_normal = self._normal.expand(batch_shape, _instance)
        new = self._from_distribution(new_normal)
        return new

    def enumerate_support(self, expand=True):
        """Returns tensor containing all values supported by a discrete dist.

        The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Note:
            Calls the enumerate_support function of the underlying normal
            distribution.

        Returns:
            torch.Tensor: Tensor iterating over dimension 0.

        """
        return self._normal.enumerate_support(expand)

    @property
    def mean(self):
        """torch.Tensor: mean of the distribution."""
        return torch.tanh(self._normal.mean)

    @property
    def variance(self):
        """torch.Tensor: variance of the underlying normal distribution."""
        return self._normal.variance

    def entropy(self):
        """Returns entropy of the underlying normal distribution.

        Returns:
            torch.Tensor: entropy of the underlying normal distribution.

        """
        return self._normal.entropy()

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0., upper=1.):
        """Clipping function that allows for gradients to flow through.

        Args:
            x (torch.Tensor): value to be clipped
            lower (float): lower bound of clipping
            upper (float): upper bound of clipping

        Returns:
            torch.Tensor: x clipped between lower and upper.

        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = ((upper - x) * clip_up + (lower - x) * clip_low)
        return x + clip

    def __repr__(self):
        """Returns the parameterization of the distribution.

        Returns:
            str: The parameterization of the distribution and underlying
                distribution.

        """
        return self.__class__.__name__


"""Utiliy functions for tensors."""
import numpy as np
import scipy.signal


def discount_cumsum(x, discount):
    """Discounted cumulative sum.

    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501
    Here, we have y[t] - discount*y[t+1] = x[t]
    or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

    Args:
        x (np.ndarrary): Input.
        discount (float): Discount factor.

    Returns:
        np.ndarrary: Discounted cumulative sum.

    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1],
                                axis=0)[::-1]


def explained_variance_1d(ypred, y, valids=None):
    """Explained variation for 1D inputs.

    It is the proportion of the variance in one variable that is explained or
    predicted from another variable.

    Args:
        ypred (np.ndarray): Sample data from the first variable.
            Shape: :math:`(N, time_limit)`.
        y (np.ndarray): Sample data from the second variable.
            Shape: :math:`(N, time_limit)`.
        valids (np.ndarray): Optional argument. Array indicating valid indices.
            If None, it assumes the entire input array are valid.
            Shape: :math:`(N, time_limit)`.

    Returns:
        float: The explained variance.

    """
    if valids is not None:
        ypred = ypred[valids.astype(bool)]
        y = y[valids.astype(bool)]
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 0:
            return 0
        return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


def flatten_tensors(tensors):
    """Flatten a list of tensors.

    Args:
        tensors (list[numpy.ndarray]): List of tensors to be flattened.

    Returns:
        numpy.ndarray: Flattened tensors.

    """
    if tensors:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    """Unflatten a flattened tensors into a list of tensors.

    Args:
        flattened (numpy.ndarray): Flattened tensors.
        tensor_shapes (tuple): Tensor shapes.

    Returns:
        list[numpy.ndarray]: Unflattened list of tensors.

    """
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [
        np.reshape(pair[0], pair[1])
        for pair in zip(np.split(flattened, indices), tensor_shapes)
    ]


def pad_tensor(x, max_len, mode='zero'):
    """Pad tensors.

    Args:
        x (numpy.ndarray): Tensors to be padded.
        max_len (int): Maximum length.
        mode (str): If 'last', pad with the last element, otherwise pad with 0.

    Returns:
        numpy.ndarray: Padded tensor.

    """
    padding = np.zeros_like(x[0])
    if mode == 'last':
        padding = x[-1]
    return np.concatenate(
        [x, np.tile(padding, (max_len - len(x), ) + (1, ) * np.ndim(x[0]))])


def pad_tensor_n(xs, max_len):
    """Pad array of tensors.

    Args:
        xs (numpy.ndarray): Tensors to be padded.
        max_len (int): Maximum length.

    Returns:
        numpy.ndarray: Padded tensor.

    """
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][:len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict, max_len, mode='zero'):
    """Pad dictionary of tensors.

    Args:
        tensor_dict (dict[numpy.ndarray]): Tensors to be padded.
        max_len (int): Maximum length.
        mode (str): If 'last', pad with the last element, otherwise pad with 0.

    Returns:
        dict[numpy.ndarray]: Padded tensor.

    """
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len, mode=mode)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len, mode=mode)
    return ret


def stack_tensor_dict_list(tensor_dict_list):
    """Stack a list of dictionaries of {tensors or dictionary of tensors}.

    Args:
        tensor_dict_list (dict[list]): a list of dictionaries of {tensors or
            dictionary of tensors}.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
        if isinstance(example, dict):
            v = stack_tensor_dict_list(dict_list)
        else:
            v = np.array(dict_list)
        ret[k] = v
    return ret


def stack_and_pad_tensor_dict_list(tensor_dict_list, max_len):
    """Stack and pad array of list of tensors.

    Input paths are a list of N dicts, each with values of shape
    :math:`(D, S^*)`. This function stack and pad the values with the input
    key with max_len, so output will be shape :math:`(N, D, S^*)`.

    Args:
        tensor_dict_list (list[dict]): List of dict to be stacked and padded.
            Value of each dict will be shape of :math:`(D, S^*)`.
        max_len (int): Maximum length for padding.

    Returns:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}. Shape: :math:`(N, D, S^*)`
            where N is the len of input paths.

    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
        if isinstance(example, dict):
            v = stack_and_pad_tensor_dict_list(dict_list, max_len)
        else:
            v = pad_tensor_n(np.array(dict_list), max_len)
        ret[k] = v
    return ret


def concat_tensor_dict_list(tensor_dict_list):
    """Concatenate dictionary of list of tensor.

    Args:
        tensor_dict_list (dict[list]): a list of dictionaries of {tensors or
            dictionary of tensors}.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
        if isinstance(example, dict):
            v = concat_tensor_dict_list(dict_list)
        else:
            v = np.concatenate(dict_list, axis=0)
        ret[k] = v
    return ret


def split_tensor_dict_list(tensor_dict):
    """Split dictionary of list of tensor.

    Args:
        tensor_dict (dict[numpy.ndarray]): a dictionary of {tensors or
            dictionary of tensors}.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def truncate_tensor_dict(tensor_dict, truncated_len):
    """Truncate dictionary of list of tensor.

    Args:
        tensor_dict (dict[numpy.ndarray]): a dictionary of {tensors or
            dictionary of tensors}.
        truncated_len (int): Length to truncate.

    Return:
        dict: a dictionary of {stacked tensors or dictionary of
            stacked tensors}

    """
    ret = dict()
    for k, v in tensor_dict.items():
        if isinstance(v, dict):
            ret[k] = truncate_tensor_dict(v, truncated_len)
        else:
            ret[k] = v[:truncated_len]
    return ret


def normalize_pixel_batch(observations):
    """Normalize the observations (images).

    Normalize pixel values to be between [0, 1].

    Args:
        observations (numpy.ndarray): Observations from environment.
            obses should be unflattened and should contain pixel
            values.

    Returns:
        numpy.ndarray: Normalized observations.

    """
    return [obs.astype(np.float32) / 255.0 for obs in observations]


def slice_nested_dict(dict_or_array, start, stop):
    """Slice a dictionary containing arrays (or dictionaries).

    This function is primarily intended for un-batching env_infos and
    action_infos.

    Args:
        dict_or_array (dict[str, dict or np.ndarray] or np.ndarray): A nested
            dictionary should only contain dictionaries and numpy arrays
            (recursively).
        start (int): First index to be included in the slice.
        stop (int): First index to be excluded from the slice. In other words,
            these are typical python slice indices.

    Returns:
        dict or np.ndarray: The input, but sliced.

    """
    if isinstance(dict_or_array, dict):
        return {
            k: slice_nested_dict(v, start, stop)
            for (k, v) in dict_or_array.items()
        }
    else:
        # It *should* be a numpy array (unless someone ignored the type
        # signature).
        return dict_or_array[start:stop]


def rrse(actual, predicted):
    """Root Relative Squared Error.

    Args:
        actual (np.ndarray): The actual value.
        predicted (np.ndarray): The predicted value.

    Returns:
        float: The root relative square error between the actual and the
            predicted value.

    """
    return np.sqrt(
        np.sum(np.square(actual - predicted)) /
        np.sum(np.square(actual - np.mean(actual))))


def sliding_window(t, window, smear=False):
    """Create a sliding window over a tensor.

    Args:
        t (np.ndarray): A tensor to create sliding window from,
            with shape :math:`(N, D)`, where N is the length of a trajectory,
            D is the dimension of each step in trajectory.
        window (int): Window size, mush be less than N.
        smear (bool): If true, copy the last window so that N windows are
            generated.

    Returns:
        np.ndarray: All windows generate over t, with shape :math:`(M, W, D)`,
            where W is the window size. If smear if False, M is :math:`N-W+1`,
            otherwise M is N.

    Raises:
        NotImplementedError: If step_size is not 1.
        ValueError: If window size is larger than the input tensor.

    """
    if window > t.shape[0]:
        raise ValueError('`window` must be <= `t.shape[0]`')
    if window == t.shape[0]:
        return np.stack([t] * window)

    # The stride trick works only on the last dimension of an ndarray, so we
    # operate on the transpose, which reverses the dimensions of t.
    t_T = t.T

    shape = t_T.shape[:-1] + (t_T.shape[-1] - window, window)
    strides = t_T.strides + (t_T.strides[-1], )
    t_T_win = np.lib.stride_tricks.as_strided(t_T,
                                              shape=shape,
                                              strides=strides)

    # t_T_win has shape (d_k, d_k-1, ..., (n - window_size), window_size)
    # To arrive at the final shape, we first transpose the result to arrive at
    # (window_size, (n - window_size), d_1, ..., d_k), then swap the firs two
    # axes
    t_win = np.swapaxes(t_T_win.T, 0, 1)

    # Optionally smear the last element to preserve the first dimension
    if smear:
        t_win = pad_tensor(t_win, t.shape[0], mode='last')

    return t_win


from math import inf
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
from torch.distributions import Beta, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, _InverseTransform

class ParameterModule(nn.Module):
    def __init__(
            self,
            init_value
    ):
        super().__init__()

        self.param = torch.nn.Parameter(init_value)

class NoWeakrefTrait(object):
    def _inv_no_weakref(self):
        """
        Returns the inverse :class:`Transform` of this transform.
        This should satisfy ``t.inv.inv is t``.
        """
        inv = None
        if self._inv is not None:
            #inv = self._inv()
            inv = self._inv
        if inv is None:
            inv = _InverseTransform(self)
            #inv = _InverseTransformNoWeakref(self)
            #self._inv = weakref.ref(inv)
            self._inv = inv
        return inv

class AffineTransformEx(AffineTransform, NoWeakrefTrait):
    @property
    def inv(self):
        return NoWeakrefTrait._inv_no_weakref(self)

    def maybe_clone_to_device(self, device):
        if device == self.loc.device:
            return self
        return AffineTransformEx(loc=self.loc.to(device, copy=True),
                                 scale=self.scale.to(device, copy=True))
class TransformedDistributionEx(TransformedDistribution):
    def entropy(self):
        """
        Returns entropy of distribution, batched over batch_shape.

        Returns:
            Tensor of shape batch_shape.
        """
        ent = self.base_dist.entropy()
        for t in self.transforms:
            if isinstance(t, AffineTransform):
                affine_ent = torch.log(torch.abs(t.scale))
                if t.event_dim > 0:
                    sum_dims = list(range(-t.event_dim, 0))
                    affine_ent = affine_ent.sum(dim=sum_dims)
                ent = ent + affine_ent
            else:
                raise NotImplementedError
        return ent

def unsqueeze_expand_flat_dim0(x, num):
    return x.unsqueeze(dim=0).expand(num, *((-1,) * x.ndim)).reshape(
            num * x.size(0), *x.size()[1:])

def _get_transform_summary(transform):
    if isinstance(transform, AffineTransform):
        return f'{type(transform).__name__}({transform.loc}, {transform.scale})'
    raise NotImplementedError

def wrap_dist_with_transforms(base_dist_cls, transforms):
    def _create(*args, **kwargs):
        return TransformedDistributionEx(base_dist_cls(*args, **kwargs),
                                         transforms)
    _create.__name__ = (f'{base_dist_cls.__name__}['
                        + ', '.join(_get_transform_summary(t) for t in transforms) + ']')
    return _create

def unwrap_dist(dist):
    while hasattr(dist, 'base_dist'):
        dist = dist.base_dist
    return dist

def get_outermost_dist_attr(dist, attr):
    while (not hasattr(dist, attr)) and hasattr(dist, 'base_dist'):
        dist = dist.base_dist
    return getattr(dist, attr, None)

def get_affine_transform_for_beta_dist(target_min, target_max):
    # https://stackoverflow.com/a/12569453/2182622
    if isinstance(target_min, (np.ndarray, np.generic)):
        assert np.all(target_min <= target_max)
    else:
        assert target_min <= target_max
    #return AffineTransform(loc=torch.Tensor(target_min),
    #                       scale=torch.Tensor(target_max - target_min))
    return AffineTransformEx(loc=torch.tensor(target_min),
                             scale=torch.tensor(target_max - target_min))

def compute_total_norm(parameters, norm_type=2):
    # Code adopted from clip_grad_norm_().
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class TrainContext:
    def __init__(self, modules):
        self.modules = modules

    def __enter__(self):
        for m in self.modules:
            m.train()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for m in self.modules:
            m.eval()

def xavier_normal_ex(tensor, gain=1., multiplier=0.1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std * multiplier)

def kaiming_uniform_ex_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', gain=None):
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    gain = gain or torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

