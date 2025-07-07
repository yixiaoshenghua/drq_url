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
        prefix += {'sampling': 'Sp', 'option': 'Op'}.get(
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

def get_torch_concat_obs(obs, option, dim=1):
    concat_obs = torch.cat([obs] + [option], dim=dim)
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

def get_option_colors(options, color_range=4):
    num_options = options.shape[0]
    dim_option = options.shape[1]

    if dim_option <= 2:
        # Use a predefined option color scheme
        if dim_option == 1:
            options_2d = []
            d = 2.
            for i in range(len(options)):
                option = options[i][0]
                if option < 0:
                    abs_value = -option
                    options_2d.append((d - abs_value * d, d))
                else:
                    abs_value = option
                    options_2d.append((d, d - abs_value * d))
            options = np.array(options_2d)
        option_colors = get_2d_colors(options, (-color_range, -color_range), (color_range, color_range))
    else:
        if dim_option > 3 and num_options >= 3:
            pca = decomposition.PCA(n_components=3)
            # Add random noises to break symmetry.
            pca_options = np.vstack((options, np.random.randn(dim_option, dim_option)))
            pca.fit(pca_options)
            option_colors = np.array(pca.transform(options))
        elif dim_option > 3 and num_options < 3:
            option_colors = options[:, :3]
        elif dim_option == 3:
            option_colors = options

        max_colors = np.array([color_range] * 3)
        min_colors = np.array([-color_range] * 3)
        if all((max_colors - min_colors) > 0):
            option_colors = (option_colors - min_colors) / (max_colors - min_colors)
        option_colors = np.clip(option_colors, 0, 1)

        option_colors = np.c_[option_colors, np.full(len(option_colors), 0.8)]

    return option_colors

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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

    v = np.reshape(v, newshape=(n_rows, n_cols, t, c, h, w))
    v = np.transpose(v, axes=(2, 0, 4, 1, 5, 3))
    v = np.reshape(v, newshape=(t, n_rows * h, n_cols * w, c))

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


def record_video(snapshot_dir, step_itr, label, trajectories, n_cols=None, skip_frames=1):
    renders = []
    for trajectory in trajectories:
        render = trajectory['env_infos']['render']
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
    save_video(snapshot_dir, step_itr, label, renders, n_cols=n_cols)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype)
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


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
            Shape: :math:`(N, max_path_length)`.
        y (np.ndarray): Sample data from the second variable.
            Shape: :math:`(N, max_path_length)`.
        valids (np.ndarray): Optional argument. Array indicating valid indices.
            If None, it assumes the entire input array are valid.
            Shape: :math:`(N, max_path_length)`.

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
