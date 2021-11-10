import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from dataclasses import dataclass
from torch.optim import Adam
import math

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GATHER_DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def calc_discounted_return(rewards, discount, final_value):
    """
    Calculate discounted returns based on rewards and discount factor.
    """
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1 , -1):
        discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
    return discounted_returns

def compute_advantages(rewards, values, discount, gae_lambda):
    """
    Compute General Advantage.
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1 , -1):
        advs[i] = advs[i + 1] * multiplier  + deltas[i]
    return advs[:-1]

def gather_trajectories(input_data):
    """
    Gather policy trajectories from gym environment.
    """

    # Unpack inputs.
    env = input_data["env"]
    ac = input_data["actor_critic"]
    parallel_rollouts = input_data["parallel_rollouts"]
    steps = input_data["steps"]

    # Initialise variables.
    obsv = env.reset()
    trajectory_data = {"states": [],
                       "actions": [],
                       "action_probabilities": [],
                       "rewards": [],
                       # "true_rewards": [],
                       "values": [],
                       "terminals": [],
                       "actor_hidden_states": [],
                       "actor_cell_states": [],
                       "critic_hidden_states": [],
                       "critic_cell_states": []}
    terminal = torch.ones(parallel_rollouts)

    with torch.no_grad():
        # Reset actor and critic state.
        ac.pi.get_init_state(parallel_rollouts, GATHER_DEVICE)
        ac.v.get_init_state(parallel_rollouts, GATHER_DEVICE)
        # Take 1 additional step in order to collect the state and value for the final state.
        for i in range(steps):

            trajectory_data["actor_hidden_states"].append(ac.pi.hidden_cell[0].squeeze(0).cpu())
            trajectory_data["actor_cell_states"].append(ac.pi.hidden_cell[1].squeeze(0).cpu())
            trajectory_data["critic_hidden_states"].append(ac.v.hidden_cell[0].squeeze(0).cpu())
            trajectory_data["critic_cell_states"].append(ac.v.hidden_cell[1].squeeze(0).cpu())

            # Choose next action
            state = torch.tensor(obsv, dtype=torch.float32)
            trajectory_data["states"].append(state)
            action, value, logp_a = ac.step(obs=state.unsqueeze(0).to(GATHER_DEVICE), terminal=terminal.to(GATHER_DEVICE))

            trajectory_data["values"].append(value.squeeze(1).cpu())
            trajectory_data["actions"].append(action.cpu())
            trajectory_data["action_probabilities"].append(logp_a)

            # Step environment
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)
            terminal = torch.tensor(done).float()
            # transformed_reward = hp.scale_reward * torch.max(_MIN_REWARD_VALUES, torch.tensor(reward).float())
            trajectory_data["rewards"].append(torch.tensor(reward))
            # trajectory_data["rewards"].append(transformed_reward)
            # trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal)

        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        _, value, _ = ac.step(obs=state.unsqueeze(0).to(GATHER_DEVICE), terminal=terminal.to(GATHER_DEVICE))
        # Future value for terminal episodes is 0.
        trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))

    # Combine step lists into tensors.
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    return trajectory_tensors

def split_trajectories_episodes(trajectory_tensors, input_data):
    """
    Split trajectories by episode. The return values are episode-wise trajectories. This function split the 2048 trajectories
    into almost 7-10 trajectories, each one is a episode. For the values, the episodic trajectory is appended one in the end.
    """
    parallel_rollouts = input_data["parallel_rollouts"]
    epochs = input_data["epochs"]
    states_episodes, actions_episodes, action_probabilities_episodes = [], [], []
    rewards_episodes, terminal_rewards_episodes, terminals_episodes, values_episodes = [], [], [], []
    policy_hidden_episodes, policy_cell_episodes, critic_hidden_episodes, critic_cell_episodes = [], [], [], []
    len_episodes = []
    trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
    for i in range(parallel_rollouts):
        terminals_tmp = trajectory_tensors["terminals"].clone()
        terminals_tmp[0, i] = 1
        terminals_tmp[-1, i] = 1
        split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

        split_lens = split_points[1:] - split_points[:-1]
        split_lens[0] += 1

        len_episode = [split_len.item() for split_len in split_lens]
        len_episodes += len_episode
        for key, value in trajectory_tensors.items():
            # Value includes additional step.
            if key == "values":
                value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward.
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                trajectory_episodes[key] += value_split
            else:
                trajectory_episodes[key] += torch.split(value[:, i], len_episode)
    return trajectory_episodes, len_episodes


def pad_and_compute_returns(trajectory_episodes, len_episodes, input_data):

    """
    Pad the trajectories up to hp.rollout_steps so they can be combined in a
    single tensor.
    Add advantages and discounted_returns to trajectories.
    trajector_episodes: dict{key: list[2d/1dtensor]}
    """
    parallel_rollouts = input_data["parallel_rollouts"]
    gamma = input_data["discount"]
    lam = input_data["gae_lambda"]
    epochs = input_data["epochs"]
    episode_count = len(len_episodes)
    advantages_episodes, discounted_returns_episodes = [], []
    padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
    padded_trajectories["advantages"] = []
    padded_trajectories["discounted_returns"] = []

    for i in range(episode_count):
        single_padding = torch.zeros(epochs - len_episodes[i])
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(epochs - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
            else:
                padding = torch.zeros(epochs - len_episodes[i], dtype=value[i].dtype)
            padded_trajectories[key].append(torch.cat((value[i], padding)))
        padded_trajectories["advantages"].append(torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                                                               values=trajectory_episodes["values"][i],
                                                                               discount=gamma,
                                                                               gae_lambda=lam), single_padding)))
        padded_trajectories["discounted_returns"].append(torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                                                                           discount=gamma,
                                                                                           final_value=trajectory_episodes["values"][i][-1]), single_padding)))
    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
    return_val["seq_len"] = torch.tensor(len_episodes)

    return return_val

@dataclass
class TrajectorBatch():
    """
    Dataclass for storing data batch.
    """
    states: torch.tensor
    actions: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    batch_size: torch.tensor
    actor_hidden_states: torch.tensor
    actor_cell_states: torch.tensor
    critic_hidden_states: torch.tensor
    critic_cell_states: torch.tensor

class TrajectoryDataset():
    """
    Fast dataset for producing training batches from trajectories.
    """
    def __init__(self, trajectories, batch_size, device, batch_len, rollout_steps):

        # Combine multiple trajectories into
        self.trajectories = {key: value.to(device) for key, value in trajectories.items()}
        self.batch_len = batch_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - batch_len + 1, 0, rollout_steps) # episode length - 8 + 1
        self.cumsum_seq_len =  np.cumsum(np.concatenate( (np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size

    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size)
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False )
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx) # values in self.valid_idx not in start_idx
            eps_idx = np.digitize(start_idx, bins = self.cumsum_seq_len, right=False) - 1 # Return the indices of the bins to which each value in input array belongs.
            a = self.cumsum_seq_len[eps_idx]  # help to understand the logic
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]  # The index of start_idx in this episode
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1
            return TrajectorBatch(**{key: value[eps_idx, series_idx]for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)


class LSTMGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_sizes, num_layers=num_layers)
        self.layer_hidden = nn.Linear(hidden_sizes, hidden_sizes)
        self.mu_output = nn.Linear(hidden_sizes, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.hidden_cell = None
        self.hidden_size = hidden_sizes
        self.num_layers = num_layers

    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def forward(self, obs, act=None, terminal=None):
        batch_size = obs.shape[1]
        device = obs.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        _, self.hidden_cell = self.lstm(obs, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        mu = self.mu_output(hidden_out)
        std = torch.exp(self.log_std)
        policy_dist = Normal(mu, std)
        logp_a = None
        if act is not None:
            logp_a = policy_dist.log_prob(act).sum(axis=-1)
        return policy_dist, logp_a


class LSTMCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_sizes, num_layers=num_layers)
        self.layer_hidden = nn.Linear(hidden_sizes, hidden_sizes)
        self.v_output = nn.Linear(hidden_sizes, 1)
        self.hidden_cell = None
        self.hidden_size = hidden_sizes
        self.num_layers = num_layers

    def get_init_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

    def forward(self, obs, terminal=None):
        batch_size = obs.shape[1]
        device = obs.device
        if self.hidden_cell is None or batch_size != self.hidden_cell[0].shape[1]:
            self.get_init_state(batch_size, device)
        if terminal is not None:
            self.hidden_cell = [value * (1. - terminal).reshape(1, batch_size, 1) for value in self.hidden_cell]
        _, self.hidden_cell = self.lstm(obs, self.hidden_cell)
        hidden_out = F.elu(self.layer_hidden(self.hidden_cell[0][-1]))
        v = self.v_output(hidden_out)
        return v

class LSTMActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.pi = LSTMGaussianActor(obs_dim, act_dim, hidden_size)
        self.v = LSTMCritic(obs_dim, hidden_size)

    def step(self, obs, act=None, terminal=None):
        with torch.no_grad():
            pi, _ = self.pi(obs, act=None, terminal=terminal)
            a = pi.sample()
            logp_a = pi.log_prob(a).sum(axis=-1)
            v = self.v(obs,terminal)
        return a, v, logp_a

    def act(self, obs, terminal):
        return self.step(obs, terminal)[0]

# model = LSTMGaussianActor(3,2,8,1)
# model = LSTMCritic(3,8)
# print(model(torch.as_tensor([[1,2,3],[4,5,6]], dtype=torch.float32).unsqueeze(0)))

