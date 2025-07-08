import numpy as np
import functools
import gym
from gym.spaces import Discrete, Dict
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from envs.wrappers import Box
import collections
from collections import defaultdict

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        self.capacity = capacity
        self.device = device

        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[0], obs_shape[0])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        # DIAYN: Add storage for skills
        self.skills = np.empty((capacity, 1), dtype=np.int64) # Skill IDs are integers

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, skill): # Added skill argument for DIAYN
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        np.copyto(self.skills[self.idx], skill) # Store the skill

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        # DIAYN: Sample skills along with other data
        skills = torch.as_tensor(self.skills[idxs], device=self.device).long() # Skills converted to long tensor

        if obses.shape[-1] == 3 or obses.shape[-1] == 9:
            obses = obses.permute(0, 3, 1, 2)
        if next_obses.shape[-1] == 3 or next_obses.shape[-1] == 9:
            next_obses = next_obses.permute(0, 3, 1, 2)
        if obses_aug.shape[-1] == 3 or obses_aug.shape[-1] == 9:
            obses_aug = obses_aug.permute(0, 3, 1, 2)
        if next_obses_aug.shape[-1] == 3 or next_obses_aug.shape[-1] == 9:
            next_obses_aug = next_obses_aug.permute(0, 3, 1, 2)
        
        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug, skills

"""A replay buffer that efficiently stores and can sample whole paths."""
class PathBufferEx:
    """A replay buffer that stores and can sample whole paths.

    This buffer only stores valid steps, and doesn't require paths to
    have a maximum length.

    Args:
        capacity_in_transitions (int): Total memory allocated for the buffer.

    """

    def __init__(self, capacity_in_transitions, pixel_shape):
        self._capacity = capacity_in_transitions
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        # Each path in the buffer has a tuple of two ranges in
        # self._path_segments. If the path is stored in a single contiguous
        # region of the buffer, the second range will be range(0, 0).
        # The "left" side of the deque contains the oldest path.
        self._path_segments = collections.deque()
        self._buffer = {}

        if pixel_shape is not None:
            self._pixel_dim = np.prod(pixel_shape)
        else:
            self._pixel_dim = None
        self._pixel_keys = ['obs', 'next_obs']

    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).

        Raises:
            ValueError: If a key is missing from path or path has wrong shape.

        """
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while (self._path_segments and self._segments_overlap(
                first_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        while (self._path_segments and self._segments_overlap(
                second_seg, self._path_segments[0][0])):
            self._path_segments.popleft()
        self._path_segments.append((first_seg, second_seg))
        for key, array in path.items():
            if self._pixel_dim is not None and key in self._pixel_keys:
                pixel_key = f'{key}_pixel'
                state_key = f'{key}_state'
                if pixel_key not in self._buffer:
                    self._buffer[pixel_key] = np.random.randint(0, 255, (self._capacity, self._pixel_dim), dtype=np.uint8)  # For memory preallocation
                    self._buffer[state_key] = np.zeros((self._capacity, array.shape[1] - self._pixel_dim), dtype=array.dtype)
                self._buffer[pixel_key][first_seg.start:first_seg.stop] = array[:len(first_seg), :self._pixel_dim]
                self._buffer[state_key][first_seg.start:first_seg.stop] = array[:len(first_seg), self._pixel_dim:]
                self._buffer[pixel_key][second_seg.start:second_seg.stop] = array[len(first_seg):, :self._pixel_dim]
                self._buffer[state_key][second_seg.start:second_seg.stop] = array[len(first_seg):, self._pixel_dim:]
            else:
                buf_arr = self._get_or_allocate_key(key, array)
                buf_arr[first_seg.start:first_seg.stop] = array[:len(first_seg)]
                buf_arr[second_seg.start:second_seg.stop] = array[len(first_seg):]
        if second_seg.stop != 0:
            self._first_idx_of_next_path = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop
        self._transitions_stored = min(self._capacity,
                                       self._transitions_stored + path_len)

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        idx = np.random.choice(self._transitions_stored, batch_size)
        if self._pixel_dim is not None:
            ret_dict = {}
            keys = set(self._buffer.keys())
            for key in self._pixel_keys:
                pixel_key = f'{key}_pixel'
                state_key = f'{key}_state'
                keys.remove(pixel_key)
                keys.remove(state_key)
                if self._buffer[state_key].shape[1] != 0:
                    ret_dict[key] = np.concatenate([self._buffer[pixel_key][idx], self._buffer[state_key][idx]], axis=1)
                else:
                    ret_dict[key] = self._buffer[pixel_key][idx]
            for key in keys:
                ret_dict[key] = self._buffer[key][idx]
            return ret_dict
        else:
            return {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}

    def _next_path_segments(self, n_indices):
        """Compute where the next path should be stored.

        Args:
            n_indices (int): Path length.

        Returns:
            tuple: Lists of indices where path should be stored.

        Raises:
            ValueError: If path length is greater than the size of buffer.

        """
        if n_indices > self._capacity:
            raise ValueError('Path is too long to store in buffer.')
        start = self._first_idx_of_next_path
        end = start + n_indices
        if end > self._capacity:
            second_end = end - self._capacity
            return (range(start, self._capacity), range(0, second_end))
        else:
            return (range(start, end), range(0, 0))

    def _get_or_allocate_key(self, key, array):
        """Get or allocate key in the buffer.

        Args:
            key (str): Key in buffer.
            array (numpy.ndarray): Array corresponding to key.

        Returns:
            numpy.ndarray: A NumPy array corresponding to key in the buffer.

        """
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = np.zeros((self._capacity, array.shape[1]), array.dtype)
            self._buffer[key] = buf_arr
        return buf_arr

    def clear(self):
        """Clear buffer."""
        self._transitions_stored = 0
        self._first_idx_of_next_path = 0
        self._path_segments.clear()
        self._buffer.clear()

    @staticmethod
    def _get_path_length(path):
        """Get path length.

        Args:
            path (dict): Path.

        Returns:
            length: Path length.

        Raises:
            ValueError: If path is empty or has inconsistent lengths.

        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    @staticmethod
    def _segments_overlap(seg_a, seg_b):
        """Compute if two segments overlap.

        Args:
            seg_a (range): List of indices of the first segment.
            seg_b (range): List of indices of the second segment.

        Returns:
            bool: True iff the input ranges overlap at at least one index.

        """
        # Empty segments never overlap.
        if not seg_a or not seg_b:
            return False
        first = seg_a
        second = seg_b
        if seg_b.start < seg_a.start:
            first, second = seg_b, seg_a
        assert first.start <= second.start
        return first.stop > second.start

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return int(self._transitions_stored)


class TrajectoryBatch(
        collections.namedtuple('TrajectoryBatch', [
            'env_spec',
            'observations',
            'last_observations',
            'actions',
            'rewards',
            'terminals',
            'env_infos',
            'agent_infos',
            'lengths',
        ])):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a batch of whole trajectories.

    Data type for on-policy algorithms.

    A :class:`TrajectoryBatch` represents a batch of whole trajectories
    produced when one or more agents interacts with one or more environments.

    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Trajectory index dimension                      |
    +-----------------------+-------------------------------------------------+
    | :math:`[T]`           | Variable-length time dimension of each          |
    |                       | trajectory                                      |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+
    | :math:`N \bullet [T]` | A dimension computed by flattening a            |
    |                       | variable-length time dimension :math:`[T]` into |
    |                       | a single batch dimension with length            |
    |                       | :math:`sum_{i \in N} [T]_i`                     |
    +-----------------------+-------------------------------------------------+

    Attributes:
        env_spec (garage.envs.EnvSpec): Specification for the environment from
            which this data was sampled.
        observations (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T], O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`env_spec.observation_space`.
        last_observations (numpy.ndarray): A numpy array of shape
            :math:`(N, O^*)` containing the last observation of each
            trajectory.  This is necessary since there are one more
            observations than actions every trajectory.
        actions (numpy.ndarray): A  numpy array of shape
            :math:`(N \bullet [T], A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`env_spec.action_space`.
        rewards (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T])` containing the rewards for all time steps
            in this batch.
        terminals (numpy.ndarray): A boolean numpy array of shape
            :math:`(N \bullet [T])` containing the termination signals for all
            time steps in this batch.
        env_infos (dict): A dict of numpy arrays arbitrary environment state
            information. Each value of this dict should be a numpy array of
            shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
        agent_infos (numpy.ndarray): A dict of numpy arrays arbitrary agent
            state information. Each value of this dict should be a numpy array
            of shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
            For example, this may contain the hidden states from an RNN policy.
        lengths (numpy.ndarray): An integer numpy array of shape :math:`(N,)`
            containing the length of each trajectory in this batch. This may be
            used to reconstruct the individual trajectories.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
    __slots__ = ()

    def __new__(cls, env_spec, observations, last_observations, actions,
                rewards, terminals, env_infos, agent_infos,
                lengths):  # noqa: D102
        # pylint: disable=too-many-branches

        first_observation = observations[0]
        first_action = actions[0]
        inferred_batch_size = lengths.sum()

        # lengths
        if len(lengths.shape) != 1:
            raise ValueError(
                'Lengths tensor must be a tensor of shape (N,), but got a '
                'tensor of shape {} instead'.format(lengths.shape))

        if not (lengths.dtype.kind == 'u' or lengths.dtype.kind == 'i'):
            raise ValueError(
                'Lengths tensor must have an integer dtype, but got dtype {} '
                'instead.'.format(lengths.dtype))

        # observations
        if not env_spec.observation_space.contains(first_observation):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(env_spec.observation_space,
                          (Box, Discrete, Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        first_observation.shape):
                    raise ValueError('observations should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         first_observation.shape))
            else:
                raise ValueError(
                    'observations must conform to observation_space {}, but '
                    'got data with shape {} instead.'.format(
                        env_spec.observation_space, first_observation))

        if observations.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of observations to be length {}, '
                'but got length {} instead.'.format(inferred_batch_size,
                                                    observations.shape[0]))

        # observations
        if not env_spec.observation_space.contains(last_observations[0]):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(env_spec.observation_space,
                          (Box, Discrete, Dict)):
                if env_spec.observation_space.flat_dim != np.prod(
                        last_observations[0].shape):
                    raise ValueError('last_observations should have the same '
                                     'dimensionality as the observation_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.observation_space.flat_dim,
                                         last_observations[0].shape))
            else:
                raise ValueError(
                    'last_observations must conform to observation_space {}, '
                    'but got data with shape {} instead.'.format(
                        env_spec.observation_space, last_observations[0]))

        if last_observations.shape[0] != len(lengths):
            raise ValueError(
                'Expected batch dimension of last_observations to be length '
                '{}, but got length {} instead.'.format(
                    len(lengths), last_observations.shape[0]))

        # actions
        if not env_spec.action_space.contains(first_action):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(env_spec.action_space,
                          (Box, Discrete, Dict)):
                if env_spec.action_space.flat_dim != np.prod(
                        first_action.shape):
                    raise ValueError('actions should have the same '
                                     'dimensionality as the action_space '
                                     '({}), but got data with shape {} '
                                     'instead'.format(
                                         env_spec.action_space.flat_dim,
                                         first_action.shape))
            else:
                raise ValueError(
                    'actions must conform to action_space {}, but got data '
                    'with shape {} instead.'.format(env_spec.action_space,
                                                    first_action))

        if actions.shape[0] != inferred_batch_size:
            raise ValueError(
                'Expected batch dimension of actions to be length {}, but got '
                'length {} instead.'.format(inferred_batch_size,
                                            actions.shape[0]))

        # rewards
        if rewards.shape != (inferred_batch_size, ):
            raise ValueError(
                'Rewards tensor must have shape {}, but got shape {} '
                'instead.'.format(inferred_batch_size, rewards.shape))

        # terminals
        if terminals.shape != (inferred_batch_size, ):
            raise ValueError(
                'terminals tensor must have shape {}, but got shape {} '
                'instead.'.format(inferred_batch_size, terminals.shape))

        if terminals.dtype != bool:
            raise ValueError(
                'terminals tensor must be dtype np.bool, but got tensor '
                'of dtype {} instead.'.format(terminals.dtype))

        # env_infos
        for key, val in env_infos.items():
            if not isinstance(val, (dict, np.ndarray)):
                raise ValueError(
                    'Each entry in env_infos must be a numpy array or '
                    'dictionary, but got key {} with value type {} instead.'.
                    format(key, type(val)))

            if (isinstance(val, np.ndarray)
                    and val.shape[0] != inferred_batch_size):
                if not (val.shape[0] == len(lengths) and sum([len(v) for v in val]) == inferred_batch_size):
                    raise ValueError(
                        'Each entry in env_infos must have a batch dimension of '
                        'length {}, but got key {} with batch size {} instead.'.
                        format(inferred_batch_size, key, val.shape[0]))

        # agent_infos
        for key, val in agent_infos.items():
            if not isinstance(val, (dict, np.ndarray)):
                raise ValueError(
                    'Each entry in agent_infos must be a numpy array or '
                    'dictionary, but got key {} with value type {} instead.'
                    'instead'.format(key, type(val)))

            if (isinstance(val, np.ndarray)
                    and val.shape[0] != inferred_batch_size):
                raise ValueError(
                    'Each entry in agent_infos must have a batch dimension of '
                    'length {}, but got key {} with batch size {} instead.'.
                    format(inferred_batch_size, key, val.shape[0]))

        return super().__new__(TrajectoryBatch, env_spec, observations,
                               last_observations, actions, rewards, terminals,
                               env_infos, agent_infos, lengths)

    @classmethod
    def concatenate(cls, *batches):
        """Create a TrajectoryBatch by concatenating TrajectoryBatches.

        Args:
            batches (list[TrajectoryBatch]): Batches to concatenate.

        Returns:
            TrajectoryBatch: The concatenation of the batches.

        """
        if __debug__:
            for b in batches:
                assert (set(b.env_infos.keys()) == set(
                    batches[0].env_infos.keys()))
                assert (set(b.agent_infos.keys()) == set(
                    batches[0].agent_infos.keys()))
        def _concatenate_env_info(x):
            if not isinstance(x[0], np.ndarray):
                return np.concatenate(x)
            all_ndims = set([i.ndim for i in x])
            if len(all_ndims) != 1 or len(set([i.shape[1:] for i in x])) != 1:
                #assert 1 in all_ndims
                #assert np.object in [i.dtype for i in x]
                res = np.empty(sum(len(i) for i in x), object)
                idx = 0
                for i in x:
                    for j in i:
                        res[idx] = j
                        idx += 1
                return res
            return np.concatenate(x)

        env_infos = {
            #k: np.concatenate([b.env_infos[k] for b in batches])
            k: _concatenate_env_info([b.env_infos[k] for b in batches])
            for k in batches[0].env_infos.keys()
        }
        agent_infos = {
            k: np.concatenate([b.agent_infos[k] for b in batches])
            for k in batches[0].agent_infos.keys()
        }
        return cls(
            batches[0].env_spec,
            np.concatenate([batch.observations for batch in batches]),
            np.concatenate([batch.last_observations for batch in batches]),
            np.concatenate([batch.actions for batch in batches]),
            np.concatenate([batch.rewards for batch in batches]),
            np.concatenate([batch.terminals for batch in batches]), env_infos,
            agent_infos, np.concatenate([batch.lengths for batch in batches]))

    def split(self):
        """Split a TrajectoryBatch into a list of TrajectoryBatches.

        The opposite of concatenate.

        Returns:
            list[TrajectoryBatch]: A list of TrajectoryBatches, with one
                trajectory per batch.

        """
        trajectories = []
        start = 0
        for i, length in enumerate(self.lengths):
            stop = start + length
            traj = TrajectoryBatch(env_spec=self.env_spec,
                                   observations=self.observations[start:stop],
                                   last_observations=np.asarray(
                                       [self.last_observations[i]]),
                                   actions=self.actions[start:stop],
                                   rewards=self.rewards[start:stop],
                                   terminals=self.terminals[start:stop],
                                   env_infos=utils.slice_nested_dict(
                                       self.env_infos, start, stop),
                                   agent_infos=utils.slice_nested_dict(
                                       self.agent_infos, start, stop),
                                   lengths=np.asarray([length]))
            trajectories.append(traj)
            start = stop
        return trajectories

    def to_trajectory_list(self):
        """Convert the batch into a list of dictionaries.

        Returns:
            list[dict[str, np.ndarray or dict[str, np.ndarray]]]: Keys:
                * observations (np.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*) (the unflattened state
                    space of the current environment).  observations[i] was
                    used by the agent to choose actions[i].
                * next_observations (np.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*). next_observations[i] was
                    observed by the agent after taking actions[i].
                * actions (np.ndarray): Non-flattened array of actions. Should
                    have shape (T, S^*) (the unflattened action space of the
                    current environment).
                * rewards (np.ndarray): Array of rewards of shape (T,) (1D
                    array of length timesteps).
                * dones (np.ndarray): Array of dones of shape (T,) (1D array
                    of length timesteps).
                * agent_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.

        """
        start = 0
        trajectories = []
        for i, length in enumerate(self.lengths):
            stop = start + length
            trajectories.append({
                'observations':
                self.observations[start:stop],
                'next_observations':
                np.concatenate((self.observations[1 + start:stop],
                                [self.last_observations[i]])),
                'actions':
                self.actions[start:stop],
                'rewards':
                self.rewards[start:stop],
                'env_infos':
                {k: v[start:stop]
                 for (k, v) in self.env_infos.items()},
                'agent_infos':
                {k: v[start:stop]
                 for (k, v) in self.agent_infos.items()},
                'dones':
                self.terminals[start:stop]
            })
            start = stop
        return trajectories

    @classmethod
    def from_trajectory_list(cls, env_spec, paths):
        """Create a TrajectoryBatch from a list of trajectories.

        Args:
            env_spec (garage.envs.EnvSpec): Specification for the environment
                from which this data was sampled.
            paths (list[dict[str, np.ndarray or dict[str, np.ndarray]]]): Keys:
                * observations (np.ndarray): Non-flattened array of
                    observations. Typically has shape (T, S^*) (the unflattened
                    state space of the current environment). observations[i]
                    was used by the agent to choose actions[i]. observations
                    may instead have shape (T + 1, S^*).
                * next_observations (np.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*). next_observations[i] was
                    observed by the agent after taking actions[i]. Optional.
                    Note that to ensure all information from the environment
                    was preserved, observations[i] should have shape (T + 1,
                    S^*), or this key should be set. However, this method is
                    lenient and will "duplicate" the last observation if the
                    original last observation has been lost.
                * actions (np.ndarray): Non-flattened array of actions. Should
                    have shape (T, S^*) (the unflattened action space of the
                    current environment).
                * rewards (np.ndarray): Array of rewards of shape (T,) (1D
                    array of length timesteps).
                * dones (np.ndarray): Array of rewards of shape (T,) (1D array
                    of length timesteps).
                * agent_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, np.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.

        """
        lengths = np.asarray([len(p['rewards']) for p in paths])
        if all(
                len(path['observations']) == length + 1
                for (path, length) in zip(paths, lengths)):
            last_observations = np.asarray(
                [p['observations'][-1] for p in paths])
            observations = np.concatenate(
                [p['observations'][:-1] for p in paths])
        else:
            # The number of observations and timesteps must match.
            observations = np.concatenate([p['observations'] for p in paths])
            if paths[0].get('next_observations') is not None:
                last_observations = np.asarray(
                    [p['next_observations'][-1] for p in paths])
            else:
                last_observations = np.asarray(
                    [p['observations'][-1] for p in paths])

        stacked_paths = utils.concat_tensor_dict_list(paths)
        return cls(env_spec=env_spec,
                   observations=observations,
                   last_observations=last_observations,
                   actions=stacked_paths['actions'],
                   rewards=stacked_paths['rewards'],
                   terminals=stacked_paths['dones'],
                   env_infos=stacked_paths['env_infos'],
                   agent_infos=stacked_paths['agent_infos'],
                   lengths=lengths)


class SkillRolloutWorker:
    def __init__(
            self,
            seed,
            time_limit,
            cur_extra_keys,
    ):
        self._observations = []
        self._last_observations = []
        self._actions = []
        self._rewards = []
        self._terminals = []
        self._lengths = []
        self._agent_infos = defaultdict(list)
        self._env_infos = defaultdict(list)
        self._prev_obs = None
        self._path_length = 0
        self._time_limit_override = None
        self._cur_extra_keys = cur_extra_keys
        self._render = False
        self._deterministic_policy = None
        self._seed = seed
        self._time_limit = time_limit
        self.worker_init()

    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None:
            utils.set_seed_everywhere(self._seed)

    def get_attrs(self, keys):
        attr_dict = {}
        for key in keys:
            attr_dict[key] = functools.reduce(getattr, [self] + key.split('.'))
        return attr_dict

    def start_rollout(self, env, policy, deterministic_policy=False):
        """Begin a new rollout."""

        # while hasattr(env, 'env'):
        #     env = getattr(env, 'env')

        self._path_length = 0
        timestep = env.reset()
        self._prev_obs = timestep["image"]
        self._prev_extra = None

        policy.reset()
        policy._force_use_mode_actions = deterministic_policy

    def step_rollout(self, env, policy, extra=None):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination of due to reaching `time_limit`.

        """
        cur_time_limit = self._time_limit if self._time_limit_override is None else self._time_limit_override
        if self._path_length < cur_time_limit:
            if 'skill' in self._cur_extra_keys:
                cur_extra_key = 'skill'
            else:
                cur_extra_key = None

            if cur_extra_key is None:
                agent_input = self._prev_obs
            else:
                cur_extra = extra[cur_extra_key]

                agent_input = utils.get_np_concat_obs(
                    self._prev_obs, cur_extra,
                )
                self._prev_extra = cur_extra

            a, agent_info = policy.get_action(agent_input)

            timestep = env.step({"action": a})

            self._observations.append(self._prev_obs)
            self._rewards.append(timestep['reward'])
            self._actions.append(a)

            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k in self._cur_extra_keys:
                self._agent_infos[k].append(extra[k])

            for k, v in timestep['info'].items():
                self._env_infos[k].append(v)
            self._path_length += 1
            self._terminals.append(timestep['is_terminal'])
            if not timestep['is_terminal']:
                self._prev_obs = timestep["image"]
                return False
        self._terminals[-1] = True
        self._lengths.append(self._path_length)
        self._last_observations.append(self._prev_obs)
        return True

    def collect_rollout(self, env):
        """Collect the current rollout, clearing the internal buffer.

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().

        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        terminals = self._terminals
        self._terminals = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        lengths = self._lengths
        self._lengths = []
        return TrajectoryBatch(env.spec, np.asarray(observations),
                               np.asarray(last_observations),
                               np.asarray(actions), np.asarray(rewards),
                               np.asarray(terminals), dict(env_infos),
                               dict(agent_infos), np.asarray(lengths,
                                                             dtype='i'))

    def rollout(self, env, policy, extra=None, deterministic_policy=False):
        """Sample a single rollout of the agent in the environment.
        Params:
            extra: {'skill': skill}
        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout(env, policy, deterministic_policy=deterministic_policy)
        while not self.step_rollout(env, policy, extra):
            pass
        return self.collect_rollout(env)

