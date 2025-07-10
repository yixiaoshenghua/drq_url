import copy
import math
import os
import abc
import pickle as pkl
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message='ing')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import argparse
import functools
from collections import namedtuple

# import dmc2gym
# import hydra # Hydra configuration is replaced by argparse
import tqdm
import json


#########################################################################################################
#                                                                                                       #
#                                            Helper functions                                           #
#                                                                                                       #
#########################################################################################################
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_

def xavier_normal_ex(tensor, gain=1., multiplier=0.1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _no_grad_normal_(tensor, 0., std * multiplier)


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

class EnvSpec:
    """A class to hold environment specifications."""
    def __init__(self, obs_space, act_space):
        self.obs_space = obs_space
        self.act_space = act_space

    @property
    def observation_space(self):
        return self.obs_space["image"]

    @property
    def action_space(self):
        return self.act_space["action"]

#########################################################################################################
#                                                                                                       #
#                                                 Models                                                #
#                                                                                                       #
#########################################################################################################

from torch.distributions import Normal, Categorical, MixtureSameFamily
from torch.distributions.independent import Independent
from typing import Any, Optional, TypeVar
from torch.nn import Module

class SpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float
    spectral_coef: float = 1.

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12, spectral_coef: float = 1.) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps
        self.spectral_coef = spectral_coef

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma * self.spectral_coef
        return weight

    def remove(self, module: Module) -> None:
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float, spectral_coef: float) -> 'SpectralNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = SpectralNorm(name, n_power_iterations, dim, eps, spectral_coef)
        weight = module._parameters[name]

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn
    
# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None:
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if version is None and all(weight_key + s in state_dict for s in ('_orig', '_u', '_v')) and \
                    weight_key not in state_dict:
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ('_orig', '', '_u'):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + '_orig']
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + '_v'] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class SpectralNormStateDictHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.fn._version


T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None,
                  spectral_coef=1.) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps, spectral_coef)
    return module

class _NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                'Non linear function {} is not supported'.format(non_linear))

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        return self.module(input_value)

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def __repr__(self):
        return repr(self.module)


class MultiHeadedMLPModule(nn.Module):
    """MultiHeadedMLPModule Model.

    A PyTorch module composed only of a multi-layer perceptron (MLP) with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        n_heads (int): Number of different output layers
        input_dim (int): Dimension of the network input.
        output_dims (int or list or tuple): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module or list or tuple):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearities (callable or torch.nn.Module or list or tuple):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_inits (callable or list or tuple): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_inits (callable or list or tuple): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 n_heads,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=torch.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False,
                 bias=True,
                 spectral_normalization=False,
                 spectral_coef=1.,
                 ):
        super().__init__()

        self._layers = nn.ModuleList()

        output_dims = self._check_parameter_for_output_layer(
            'output_dims', output_dims, n_heads)
        output_w_inits = self._check_parameter_for_output_layer(
            'output_w_inits', output_w_inits, n_heads)
        output_b_inits = self._check_parameter_for_output_layer(
            'output_b_inits', output_b_inits, n_heads)
        output_nonlinearities = self._check_parameter_for_output_layer(
            'output_nonlinearities', output_nonlinearities, n_heads)

        self._layers = nn.ModuleList()

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            if spectral_normalization:
                linear_layer = spectral_norm(nn.Linear(prev_size, size, bias=bias), spectral_coef=spectral_coef)
            else:
                linear_layer = nn.Linear(prev_size, size, bias=bias)
            hidden_w_init(linear_layer.weight)
            if bias:
                hidden_b_init(linear_layer.bias)
            hidden_layers.add_module('linear', linear_layer)

            if layer_normalization:
                hidden_layers.add_module('layer_normalization', nn.LayerNorm(size))

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity', _NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            output_layer = nn.Sequential()
            if spectral_normalization:
                linear_layer = spectral_norm(nn.Linear(prev_size, output_dims[i], bias=bias), spectral_coef=spectral_coef)
            else:
                linear_layer = nn.Linear(prev_size, output_dims[i], bias=bias)
            output_w_inits[i](linear_layer.weight)
            if bias:
                output_b_inits[i](linear_layer.bias)
            output_layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                output_layer.add_module(
                    'non_linearity', _NonLinearity(output_nonlinearities[i]))

            self._output_layers.append(output_layer)

    @classmethod
    def _check_parameter_for_output_layer(cls, var_name, var, n_heads):
        """Check input parameters for output layer are valid.

        Args:
            var_name (str): variable name
            var (any): variable to be checked
            n_heads (int): number of head

        Returns:
            list: list of variables (length of n_heads)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_heads

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_heads
            if len(var) == n_heads:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'n_heads ({}), but {} provided.')
            raise ValueError(msg.format(var_name, n_heads, var))
        return [copy.deepcopy(var) for _ in range(n_heads)]

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]

    def get_last_linear_layer(self):
        for m in reversed(self._layers):
            if isinstance(m, nn.Sequential):
                for l in reversed(m):
                    if isinstance(l, nn.Linear):
                        return l
            if isinstance(m, nn.Linear):
                return m
        return None

class MLPModule(MultiHeadedMLPModule):
    """MLP Model.

    A Pytorch module composed only of a multi-layer perceptron (MLP), which
    maps real-valued inputs to real-valued outputs.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 **kwargs):
        super().__init__(1, input_dim, output_dim, hidden_sizes,
                         hidden_nonlinearity, hidden_w_init, hidden_b_init,
                         output_nonlinearity, output_w_init, output_b_init,
                         layer_normalization, **kwargs)

        self._output_dim = output_dim

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value

        """
        return super().forward(input_value)[0]

    @property
    def output_dim(self):
        """Return output dimension of network.

        Returns:
            int: Output dimension of network.

        """
        return self._output_dim

class GaussianMLPBaseModule(nn.Module):
    """Base of GaussianMLPModel.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two skills:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation.
            - softplus: the std will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_std = learn_std
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        if self._std_parameterization not in ('exp', 'softplus', 'softplus_real'):
            raise NotImplementedError

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

        self._min_std_param = self._max_std_param = None
        if min_std is not None:
            self._min_std_param = torch.Tensor([min_std]).log()
            self.register_buffer('min_std_param', self._min_std_param)
        if max_std is not None:
            self._max_std_param = torch.Tensor([max_std]).log()
            self.register_buffer('max_std_param', self._max_std_param)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        ret = super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        self._min_std_param = buffers.get('min_std_param', None)
        self._max_std_param = buffers.get('max_std_param', None)
        return ret

    # Parent module's .to(), .cpu(), and .cuda() call children's ._apply().
    def _apply(self, *args, **kwargs):
        ret = super()._apply(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']
        self._min_std_param = buffers.get('min_std_param', None)
        self._max_std_param = buffers.get('max_std_param', None)
        return ret

    @abc.abstractmethod
    def _get_mean_and_log_std(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._std_parameterization not in ['softplus_real']:
            if self._min_std_param or self._max_std_param:
                log_std_uncentered = log_std_uncentered.clamp(
                    min=(None if self._min_std_param is None else
                         self._min_std_param.item()),
                    max=(None if self._max_std_param is None else
                         self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        elif self._std_parameterization == 'softplus':
            std = log_std_uncentered.exp().exp().add(1.).log()
        elif self._std_parameterization == 'softplus_real':
            std = log_std_uncentered.exp().add(1.).log()
        else:
            assert False
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist

    @abc.abstractmethod
    def get_last_linear_layers(self):
        pass

class GaussianMLPIndependentStdModule(GaussianMLPBaseModule):
    """GaussianMLPModule which has two different mean and std network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two skills:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=torch.tanh,
                 std_hidden_w_init=nn.init.xavier_uniform_,
                 std_hidden_b_init=nn.init.zeros_,
                 std_output_nonlinearity=None,
                 std_output_w_init=nn.init.xavier_uniform_,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal,
                 **kwargs):
        super(GaussianMLPIndependentStdModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_hidden_sizes=std_hidden_sizes,
                             std_hidden_nonlinearity=std_hidden_nonlinearity,
                             std_hidden_w_init=std_hidden_w_init,
                             std_hidden_b_init=std_hidden_b_init,
                             std_output_nonlinearity=std_output_nonlinearity,
                             std_output_w_init=std_output_w_init,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)

        self._mean_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization,
            **kwargs)

        self._log_std_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._std_hidden_sizes,
            hidden_nonlinearity=self._std_hidden_nonlinearity,
            hidden_w_init=self._std_hidden_w_init,
            hidden_b_init=self._std_hidden_b_init,
            output_nonlinearity=self._std_output_nonlinearity,
            output_w_init=self._std_output_w_init,
            output_b_init=self._init_std_b,
            layer_normalization=self._layer_normalization,
            **kwargs)

    def _init_std_b(self, b):
        """Default bias initialization function.

        Args:
            b (torch.Tensor): The bias tensor.

        Returns:
            torch.Tensor: The bias tensor itself.

        """
        if self._std_parameterization not in ['softplus_real']:
            return nn.init.constant_(b, self._init_std.item())
        else:
            return nn.init.constant_(b, self._init_std.exp().exp().add(-1.0).log().item())

    def _get_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return self._mean_module(*inputs), self._log_std_module(*inputs)

    def get_last_linear_layers(self):
        return {
            'mean': self._mean_module.get_last_linear_layer(),
            'std': self._log_std_module.get_last_linear_layer(),
        }


class ForwardWithTransformTrait(object):
    def forward_with_transform(self, *inputs, transform):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        mean = transform(mean)
        std = transform(std)

        dist_transformed = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist_transformed, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist_transformed.batch_shape samples.
            dist_transformed = Independent(dist_transformed, 1)

        return dist, dist_transformed

class ForwardWithChunksTrait(object):
    def forward_with_chunks(self, *inputs, merge):
        mean = []
        log_std_uncentered = []
        for chunk_inputs in zip(*inputs):
            chunk_mean, chunk_log_std_uncentered = self._get_mean_and_log_std(*chunk_inputs)
            mean.append(chunk_mean)
            log_std_uncentered.append(chunk_log_std_uncentered)
        mean = merge(mean, batch_dim=0)
        log_std_uncentered = merge(log_std_uncentered, batch_dim=0)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pretanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist

class ForwardModeTrait(object):
    def forward_mode(self, *inputs):
        mean, log_std_uncentered = self._get_mean_and_log_std(*inputs)

        if self._min_std_param or self._max_std_param:
            log_std_uncentered = log_std_uncentered.clamp(
                min=(None if self._min_std_param is None else
                     self._min_std_param.item()),
                max=(None if self._max_std_param is None else
                     self._max_std_param.item()))

        if self._std_parameterization == 'exp':
            std = log_std_uncentered.exp()
        else:
            std = log_std_uncentered.exp().exp().add(1.).log()

        dist = self._norm_dist_class(mean, std)
        # This control flow is needed because if a TanhNormal distribution is
        # wrapped by torch.distributions.Independent, then custom functions
        # such as rsample_with_pre_tanh_value of the TanhNormal distribution
        # are not accessable.
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist.mean

class GaussianMLPTwoHeadedModule(GaussianMLPBaseModule):
    """GaussianMLPModule which has only one mean network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two skills:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        normal_distribution_cls (torch.distribution): normal distribution class
            to be constructed and returned by a call to forward. By default, is
            `torch.distributions.Normal`.

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_parameterization='exp',
                 layer_normalization=False,
                 normal_distribution_cls=Normal):
        super(GaussianMLPTwoHeadedModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_std=learn_std,
                             init_std=init_std,
                             min_std=min_std,
                             max_std=max_std,
                             std_parameterization=std_parameterization,
                             layer_normalization=layer_normalization,
                             normal_distribution_cls=normal_distribution_cls)

        self._shared_mean_log_std_network = MultiHeadedMLPModule(
            n_heads=2,
            input_dim=self._input_dim,
            output_dims=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=[
                nn.init.zeros_,
                (lambda x: nn.init.constant_(x, self._init_std.item())
                 if self._std_parameterization not in ['softplus_real']
                 else lambda x: nn.init.constant_(x, self._init_std.exp().exp().add(-1.0).log().item())),
            ],
            layer_normalization=self._layer_normalization)

    def _get_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        return self._shared_mean_log_std_network(*inputs)

    def get_last_linear_layers(self):
        return {
            'mean': self._shared_mean_log_std_network.get_last_linear_layer(),
        }

class GaussianMLPIndependentStdModuleEx(GaussianMLPIndependentStdModule, ForwardWithTransformTrait, ForwardWithChunksTrait, ForwardModeTrait):
    pass

class GaussianMLPTwoHeadedModuleEx(GaussianMLPTwoHeadedModule, ForwardWithTransformTrait, ForwardWithChunksTrait, ForwardModeTrait):
    pass

class NormLayer(nn.Module):
    def __init__(self, name, dim=None):
        super().__init__()
        if name == 'none':
            self._layer = None
        elif name == 'layer':
            assert dim != None
            self._layer = nn.LayerNorm(dim)
        else:
            raise NotImplementedError(name)

    def forward(self, features):
        if self._layer is None:
            return features
        return self._layer(features)


class CNN(nn.Module):
    def __init__(self, num_inputs, act=nn.ELU, norm='none', cnn_depth=48, cnn_kernels=(4, 4, 4, 4), mlp_layers=(400, 400, 400, 400), spectral_normalization=False):
        super().__init__()

        self._num_inputs = num_inputs
        self._act = act()
        self._norm = norm
        self._cnn_depth = cnn_depth
        self._cnn_kernels = cnn_kernels
        self._mlp_layers = mlp_layers

        self._conv_model = []
        for i, kernel in enumerate(self._cnn_kernels):
            if i == 0:
                prev_depth = num_inputs
            else:
                prev_depth = 2 ** (i - 1) * self._cnn_depth
            depth = 2 ** i * self._cnn_depth
            if spectral_normalization:
                self._conv_model.append(spectral_norm(nn.Conv2d(prev_depth, depth, kernel, stride=2)))
            else:
                self._conv_model.append(nn.Conv2d(prev_depth, depth, kernel, stride=2))
            self._conv_model.append(NormLayer(norm, depth))
            self._conv_model.append(self._act)
        self._conv_model = nn.Sequential(*self._conv_model)

    def forward(self, data):
        output = self._conv_model(data)
        output = output.reshape(output.shape[0], -1)
        return output


class Encoder(nn.Module):
    def __init__(
            self,
            pixel_shape,
            spectral_normalization=False,
    ):
        super().__init__()

        self.pixel_shape = pixel_shape
        self.pixel_dim = np.prod(pixel_shape)

        self.pixel_depth = self.pixel_shape[-1]

        self.encoder = CNN(self.pixel_depth, spectral_normalization=spectral_normalization)

    def forward(self, input):
        '''
        input: (N, h*w*c*framestack+dim_skill)
        '''
        assert len(input.shape) == 2

        pixel = input[..., :self.pixel_dim].reshape(-1, *self.pixel_shape).permute(0, 3, 1, 2)
        state = input[..., self.pixel_dim:]

        pixel = pixel / 255.

        rep = self.encoder(pixel)
        rep = rep.reshape(rep.shape[0], -1)
        output = torch.cat([rep, state], dim=-1)

        return output


class WithEncoder(nn.Module):
    def __init__(
            self,
            encoder,
            module,
    ):
        super().__init__()

        self.encoder = encoder
        self.module = module

    def get_rep(self, input):
        return self.encoder(input)

    def forward(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module(rep, *inputs[1:])

    def forward_mode(self, *inputs):
        rep = self.get_rep(inputs[0])
        return self.module.forward_mode(rep, *inputs[1:])

def make_encoder(pixel_shape, **kwargs):
    return Encoder(pixel_shape=pixel_shape, **kwargs)

def with_encoder(module, pixel_shape, encoder=None):
    if encoder is None:
        encoder = make_encoder(pixel_shape)

    return WithEncoder(encoder=encoder, module=module)

#########################################################################################################
#                                                                                                       #
#                                            Skill policy                                               #
#                                                                                                       #
#########################################################################################################


class MetraAgent(torch.nn.Module):
    def __init__(self,
                 name,
                 env_spec,
                 env,
                 framestack,
                 dim_skill,
                 model_master_dim,
                 model_master_num_layers,
                 model_master_nonlinearity,
                 use_encoder=True,
                 discrete=False,
                 unit_length=False,
                 clip_action=False,
                 omit_obs_idxs=None,
                 skill_info=None,
                 force_use_mode_actions=False,
                 spectral_normalization=False,
                 inner=True,
                 ):
        super().__init__()

        self._name = name
        self.env_spec = env_spec
        self._clip_action = clip_action
        self._omit_obs_idxs = omit_obs_idxs
        self.dim_skill = dim_skill
        self.discrete = discrete
        self.unit_length = unit_length
        self.inner = inner
        self.framestack = framestack

        self._skill_info = skill_info
        self._force_use_mode_actions = force_use_mode_actions

        pixel_shape = (*env.obs_space['image'].shape[:-1], 
                   env.obs_space['image'].shape[-1] * framestack)
        nonlinearity = {'relu': torch.relu, 'tanh': torch.tanh, None: None}[model_master_nonlinearity]
        module_obs_dim = make_encoder(pixel_shape)(
                                    torch.as_tensor(
                                        np.random.randint(0, 255, size=np.prod(pixel_shape), dtype=np.uint8)
                                        ).float().unsqueeze(0)
                                    ).shape[-1]
        master_dims = [model_master_dim] * model_master_num_layers
        # skill policy
        module = GaussianMLPTwoHeadedModuleEx(
                        input_dim=module_obs_dim + dim_skill,
                        output_dim=env.act_space['action'].shape[0],
                        hidden_sizes=master_dims,
                        hidden_nonlinearity=nonlinearity,
                        layer_normalization=False,
                        max_std=np.exp(2.),
                        normal_distribution_cls=TanhNormal,
                        output_w_init=functools.partial(xavier_normal_ex, gain=1.),
                        init_std=1.,
                    )
        self.policy = with_encoder(module, pixel_shape=pixel_shape) if use_encoder else module

        # traj encoder
        traj_encoder_obs_dim = module_obs_dim
        traj_encoder = GaussianMLPIndependentStdModuleEx(
            input_dim=traj_encoder_obs_dim,
            output_dim=dim_skill,
            std_hidden_sizes=master_dims,
            std_hidden_nonlinearity=nonlinearity or torch.relu,
            std_hidden_w_init=torch.nn.init.xavier_uniform_,
            std_output_w_init=torch.nn.init.xavier_uniform_,
            init_std=1.0,
            min_std=1e-6,
            max_std=None,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            hidden_w_init=torch.nn.init.xavier_uniform_,
            output_w_init=torch.nn.init.xavier_uniform_,
            std_parameterization='exp',
            bias=True,
            spectral_normalization=spectral_normalization,
        )
        if use_encoder:
            if spectral_normalization:
                te_encoder = make_encoder(pixel_shape, spectral_normalization=True)
            else:
                te_encoder = None
            traj_encoder = with_encoder(traj_encoder, pixel_shape, encoder=te_encoder)
        self.traj_encoder = traj_encoder

    @property
    def name(self):
        """Name of policy.

        Returns:
            str: Name of policy

        """
        return self._name

    @property
    def observation_space(self):
        """The observation space for the environment.

        Returns:
            akro.Space: Observation space.

        """
        return self.env_spec.observation_space

    @property
    def action_space(self):
        """The action space for the environment.

        Returns:
            akro.Space: Action space.

        """
        return self.env_spec.action_space
    
    def set_to_eval_mode(self):
        self.policy.eval()
        self.traj_encoder.eval()
    
    def load(self, load_path, device='cuda'):
        params_dict = torch.load(load_path, map_location=device)
        self.policy.load_state_dict(params_dict['skill_policy_core_module'])
        self.traj_encoder.load_state_dict(params_dict['traj_encoder'])

    def get_param_values(self):
        """Get the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Returns:
            dict: The parameters (in the form of the state dictionary).

        """
        return self.state_dict()

    def set_param_values(self, state_dict):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            state_dict (dict): State dictionary.

        """
        self.load_state_dict(state_dict)

    def reset(self, dones=None):
        """Reset the environment.

        Args:
            dones (numpy.ndarray): Reset values

        """
        return

    def process_observations(self, observations):
        if self._omit_obs_idxs is not None:
            observations = observations.clone()
            observations[:, self._omit_obs_idxs] = 0
        return observations

    def forward(self, observations):
        '''
        observations: (N, h*w*c*framestack+dim_skill)
        '''
        observations = self.process_observations(observations)
        dist = self.policy(observations)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()
        if hasattr(dist, '_normal'):
            info.update(dict(
                normal_mean=dist._normal.mean,
                normal_std=dist._normal.variance.sqrt(),
            ))

        return dist, info

    def forward_mode(self, observations):
        observations = self.process_observations(observations)
        samples = self.policy.forward_mode(observations)
        return samples, dict()

    def forward_with_transform(self, observations, *, transform):
        observations = self.process_observations(observations)
        dist, dist_transformed = self.policy.forward_with_transform(observations, transform=transform)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            ret_mean_transformed = dist_transformed.mean.cpu()
            ret_log_std_transformed = (dist_transformed.variance.sqrt()).log().cpu()
            info = (dict(mean=ret_mean, log_std=ret_log_std),
                    dict(mean=ret_mean_transformed, log_std=ret_log_std_transformed))
        except NotImplementedError:
            info = (dict(),
                    dict())
        return (dist, dist_transformed), info

    def forward_with_chunks(self, observations, *, merge):
        observations = [self.process_observations(o) for o in observations]
        dist = self.policy.forward_with_chunks(observations,
                                                merge=merge)
        try:
            ret_mean = dist.mean
            ret_log_std = (dist.variance.sqrt()).log()
            info = dict(mean=ret_mean, log_std=ret_log_std)
        except NotImplementedError:
            info = dict()

        return dist, info

    def get_mode_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            samples, info = self.forward_mode(observations)
            return samples.cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }

    def get_sample_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
            dist, info = self.forward(observations)
            if isinstance(dist, TanhNormal):
                pre_tanh_values, actions = dist.rsample_with_pre_tanh_value()
                log_probs = dist.log_prob(actions, pre_tanh_values)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['pre_tanh_value'] = pre_tanh_values.detach().cpu().numpy()
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            else:
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                actions = actions.detach().cpu().numpy()
                infos = {
                    k: v.detach().cpu().numpy()
                    for (k, v) in info.items()
                }
                infos['log_prob'] = log_probs.detach().cpu().numpy()
            return actions, infos

    def get_actions(self, observations):
        assert isinstance(observations, np.ndarray) or isinstance(observations, torch.Tensor)
        if self._force_use_mode_actions:
            actions, info = self.get_mode_actions(observations)
        else:
            actions, info = self.get_sample_actions(observations)
        if self._clip_action:
            epsilon = 1e-6
            actions = np.clip(
                actions,
                self.env_spec.action_space.low + epsilon,
                self.env_spec.action_space.high - epsilon,
            )
        return actions, info

    def get_action(self, observation):
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.as_tensor(observation).float().to(next(self.parameters()).device)
            observation = observation.unsqueeze(0)
            action, agent_infos = self.get_actions(observation)
            return action[0], {k: v[0] for k, v in agent_infos.items()}

    def get_skills(self, batch_size=1):
        if self.discrete: # False
            skills = np.eye(self.dim_skill)[np.random.randint(0, self.dim_skill, batch_size)]
        else:
            skills = np.random.randn(batch_size, self.dim_skill)
            if self.unit_length:
                skills /= np.linalg.norm(skills, axis=-1, keepdims=True)
        return skills

    def choose_action(self, obs, skill, sample=False):
        """
        Selects an action based on observation and current skill (only for world_model post-training).
        params
            skill: (b*T, dim_skill)
            obs: (b*T, 9, 64, 64)
        """
        self._force_use_mode_actions = not sample
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs).float().to(next(self.parameters()).device)
            if len(obs.shape) > 2:
                if obs.shape[-1] != self.framestack * 3:
                    obs = obs.permute(0, 2, 3, 1)
                obs = obs.reshape((obs.shape[0], -1)) # (bT, h*w*c)
            if not isinstance(skill, torch.Tensor):
                skill = torch.as_tensor(skill).float().to(next(self.parameters()).device)
            observations = torch.cat([obs, skill], dim=-1)
            action, _ = self.get_actions(observations)
        return action 

    def assign_rewards(self, skill, obs, next_obs):
        """
        Computes the METRA intrinsic reward: r_i = .
        Args:
            obs (torch.Tensor): Batch of observations (states), (bT, 9, 64, 64)
            skill (torch.Tensor): Batch of skills (actions/latents z), (bT, dim_skill)
        Returns:
            torch.Tensor: Batch of intrinsic rewards.
        """
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs).float().to(next(self.parameters()).device)
                next_obs = torch.as_tensor(next_obs).float().to(next(self.parameters()).device)
            if len(obs.shape) > 2:
                if obs.shape[-1] != self.framestack * 3:
                    obs = obs.permute(0, 2, 3, 1)
                    next_obs = next_obs.permute(0, 2, 3, 1)
                obs = obs.reshape((obs.shape[0], -1)) # (bT, h*w*c)
                next_obs = next_obs.reshape((next_obs.shape[0], -1))
            if not isinstance(skill, torch.Tensor):
                skill = torch.as_tensor(skill).float().to(next(self.parameters()).device)
            if self.inner:
                cur_z = self.traj_encoder(obs).mean
                next_z = self.traj_encoder(next_obs).mean
                target_z = next_z - cur_z

                if self.discrete:
                    masks = (skill - skill.mean(dim=1, keepdim=True)) * self.dim_skill / (self.dim_skill - 1 if self.dim_skill != 1 else 1)
                    rewards = (target_z * masks).sum(dim=1)
                else:
                    rewards = (target_z * skill).sum(dim=1)
            else:
                target_dists = self.traj_encoder(next_obs)

                if self.discrete:
                    logits = target_dists.mean
                    rewards = -torch.nn.functional.cross_entropy(logits, skill.argmax(dim=1), reduction='none')
                else:
                    rewards = target_dists.log_prob(skill)
            if len(rewards.shape) > 1:
                rewards = rewards.squeeze(-1) # Ensure shape [b*T,]

        return rewards


#########################################################################################################
#                                                                                                       #
#                                              Test Case                                                #
#                                                                                                       #
#########################################################################################################


if __name__ == "__main__":
    from video import RewardVideoRecorder
    import utils
    from moviepy import editor as mpy
    from envs import make_env
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = json.load(open("runs/metra/metaworld_reach/20250709-014111_seed0/args.json", 'r'))
    tmp = namedtuple('args', args.keys())
    args = tmp(*list(args.values()))
    env = make_env(mode="train", config=args)
    
    skill_policy = MetraAgent(
            name='metra_policy',
            env_spec=EnvSpec(env.obs_space, env.act_space),
            env=env,
            framestack=1,#args.framestack,
            dim_skill=args.dim_skill,
            model_master_dim=args.model_master_dim,
            model_master_num_layers=args.model_master_num_layers,
            model_master_nonlinearity=args.model_master_nonlinearity,
            use_encoder=True,
            discrete=args.discrete,
            skill_info={'dim_skill': args.dim_skill},
            spectral_normalization=args.spectral_normalization,
        ).to(device)
    
    logdir = '/data1/qiuzh/wansh/RL/experiments/drq_url/runs/metra/metaworld_reach/20250709-014111_seed0/models'

    saved_policy = torch.load(f"{logdir}/epoch-2999/skill_policy.pt")
    saved_encoder = torch.load(f"{logdir}/epoch-2999/traj_encoder.pt")
    params_dict = dict(skill_policy_core_module=saved_policy['policy']._module.state_dict(),
                       traj_encoder=saved_encoder['traj_encoder'].state_dict())
    torch.save(params_dict, f"{logdir}/final_metra_model.pt")
    print(f"Successfully save checkpoints to {logdir}!")

    skill_policy.load(f"{logdir}/final_metra_model.pt")
    print(f"Successfully load checkpoints from {logdir}!")


    # generate skills
    num_trajs = 10
    skills = skill_policy.get_skills(args.dim_skill)
    videos = np.zeros((args.time_limit, args.dim_skill*64, num_trajs*64, 3), dtype=np.uint8)
    for skill_i, skill in enumerate(skills):
        # init env
        env_returns_for_curr_skill = []
        int_returns_for_curr_skill = []
        # 10 traj for each skill
        for traj_i in tqdm.tqdm(range(num_trajs)):
            i = 0
            obs = env.reset()
            env_rewards_for_curr_skill = []
            int_rewards_for_curr_skill = []
            while not obs['is_terminal']:
                prev_obs = obs
                videos[i, 64*skill_i:64*(skill_i+1), 64*traj_i:64*(traj_i+1), :] = prev_obs['image'].reshape((64, 64, 9))[:, :, -3:]
                action = skill_policy.choose_action(prev_obs['image'].reshape((1, -1)), skill.reshape((1, *skill.shape)))
                obs = env.step({'action': action.ravel()})
                env_rewards_for_curr_skill.append(obs['reward'])
                int_reward = skill_policy.assign_rewards(skill.reshape((1, *skill.shape)), prev_obs['image'].reshape((1, -1)), obs['image'].reshape((1, -1)))
                int_rewards_for_curr_skill.append(int_reward.item())
                i += 1
            env_returns_for_curr_skill.append(np.sum(env_rewards_for_curr_skill))
            int_returns_for_curr_skill.append(np.sum(int_rewards_for_curr_skill))
        
        # calculate the stats for each traj (mean / std)
        print("Skill_id: {} | Intrinsic return mean: {:.1f}, std: {:.1f} | Env return mean: {:.1f}, std: {:.1f}".format(skill_i, np.mean(int_returns_for_curr_skill), np.std(int_returns_for_curr_skill), np.mean(env_returns_for_curr_skill), np.std(env_returns_for_curr_skill)))
    # record videos (T, skills*height, 10*width, 3) with per-step reward
    np.savez("videos.npz", videos=videos)
    clip = mpy.ImageSequenceClip(list(videos.astype(np.uint8)), fps=15)
    clip.write_videofile(f"test.mp4", audio=False, verbose=False, logger=None)





















