import gym
import torch
import numpy as np
from core_lstm import *
import time
import wandb
import datetime
import sys
from copy import deepcopy


def train_model(actor_critic, actor_optimizer, critic_optimizer, epochs):

    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    env = gym.vector.make(args.env, args.parallel_rollouts, asynchronous=False)
    # # print(env)
    iteration = 0
    while iteration < epochs:

        ac = actor_critic
        start_gather_time = time.time()
        # transforms = [ hl.transforms.Prune('Constant') ] # Removes Constant nodes from graph.
        #
        # graph = hl.build_graph(actor, torch.zeros([1,32,24]), transforms=transforms)
        # graph.theme = hl.graph.THEMES['blue'].copy()
        # graph.save('rnn_hiddenlayer', format='png')
        # print('actor:', actor)
        # print('critic:', critic)
        # Gather trajectories.
        input_data = {"env": env, "actor_critic": ac, "discount": args.gamma,
                      "gae_lambda": args.lam, "parallel_rollouts": args.parallel_rollouts,
                      "epochs": args.epochs, "steps": args.steps}
        trajectory_tensors = gather_trajectories(input_data)
        # b = {key: [] for key in trajectory_tensors.keys()}
        # {b[key].append(x.numpy()) for key, value in trajectory_tensors.items() for x in value}
        trajectory_episodes, len_episodes = split_trajectories_episodes(trajectory_tensors, input_data)
        trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes, input_data)
        # trajectories_np = {key: value.numpy() for key, value in trajectories.items()}

        # Calculate mean reward.
        complete_episode_count = trajectories["terminals"].sum().item() # total episode number is 308, but the complete count is 276, this is because
        # there are 32 training in parallel, for each training, the episode is not done at the end of trajectory (2048)
        terminal_episodes_rewards = (trajectories["terminals"].sum(axis=1) * trajectories["rewards"].sum(axis=1)).sum()
        # if the episode finishes, trajectories["terminals"].sum(axis=1) = 1, then the rewards of this trajectory counts, otherwise
        # trajectories["terminals"].sum(axis=1) = 0, multiplication is 0, means this trajectory doesn't count
        mean_reward =  terminal_episodes_rewards / (complete_episode_count)
        wandb.log({"reward/epoch reward": mean_reward})
        # Check stop conditions.
        # if mean_reward > stop_conditions.best_reward:
        #     stop_conditions.best_reward = mean_reward
        #     stop_conditions.fail_to_improve_count = 0
        # else:
        #     stop_conditions.fail_to_improve_count += 1
        # if stop_conditions.fail_to_improve_count > hp.patience:
        #     print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
        #     break

        trajectory_dataset = TrajectoryDataset(trajectories, batch_size=args.batch_size,
                                               device=TRAIN_DEVICE, batch_len=args.recurrent_seq_len, rollout_steps=args.epochs)
        end_gather_time = time.time()
        print(f"The gather time is {end_gather_time-start_gather_time}")
        start_train_time = time.time()

        # actor = actor.to(TRAIN_DEVICE)
        # critic = critic.to(TRAIN_DEVICE)
        ac = ac.to(TRAIN_DEVICE)

        # Train actor and critic.
        for epoch_idx in range(20):
            start_batch_time = time.time()
            for i, batch in enumerate(trajectory_dataset):

                # Get batch
                ac.pi.hidden_cell = (batch.actor_hidden_states[:1], batch.actor_cell_states[:1])

                # Update actor
                actor_optimizer.zero_grad()
                action_dist, action_probabilities = ac.pi(obs=batch.states, act=batch.actions[-1, :].to("cpu"))
                # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                # action_probabilities = action_dist.log_prob(batch.actions[-1, :].to("cpu")).to(TRAIN_DEVICE)
                # Compute probability ratio from probabilities in logspace.
                probabilities_ratio = torch.exp(action_probabilities - batch.action_probabilities[-1, :])
                surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :]
                surrogate_loss_1 =  torch.clamp(probabilities_ratio, 1. - args.clip_ratio, 1. + args.clip_ratio) * batch.advantages[-1, :]
                surrogate_loss_2 = action_dist.entropy().to(TRAIN_DEVICE)
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(args.entropy_factor * surrogate_loss_2)
                actor_loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(ac.pi.parameters(), args.max_grad_norm)
                actor_optimizer.step()

                # Update critic
                critic_optimizer.zero_grad()
                ac.v.hidden_cell = (batch.critic_hidden_states[:1], batch.critic_cell_states[:1])
                values = ac.v(batch.states)
                critic_loss = F.mse_loss(batch.discounted_returns[-1, :], values.squeeze(1))
                torch.nn.utils.clip_grad.clip_grad_norm_(ac.v.parameters(), args.max_grad_norm)
                critic_loss.backward()
                critic_optimizer.step()
            end_batch_time = time.time()
            # print(f"It takes {i} times iteration")
            # print(f"It takes {end_batch_time-start_batch_time} s to train one batch")

        end_train_time = time.time()
        print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
              f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
              f"Train time: {end_train_time - start_train_time:.2f}s")

        if iteration % 2 == 0:
            env_test = gym.make(args.env)
            obsv_test = env_test.reset()
            obsv_test = torch.tensor(obsv_test, dtype=torch.float32)
            done_test = False
            terminal_test = torch.ones(1)
            ac_test = LSTMActorCritic(env_test.observation_space, env_test.action_space, args.hid)
            ac_test.load_state_dict(deepcopy(ac.state_dict()))
            ac_test.pi.get_init_state(1, GATHER_DEVICE)
            ac_test.v.get_init_state(1, GATHER_DEVICE)
            while not done_test:
                action_test = ac_test.act(obs=obsv_test.view(1,1,-1).to(GATHER_DEVICE), terminal=terminal_test.to(GATHER_DEVICE))
                action_np_test = action_test.cpu().numpy()
                obsv_test, reward_test, done_test, _ = env_test.step(action_np_test)
                obsv_test = torch.tensor(obsv_test, dtype=torch.float32)
                terminal_test = torch.tensor(done_test).float()
            if len(env_test.forge.rod.pieces_temp) != 0:
                result = np.r_[np.array(env_test.forge.rod.pieces_temp),
                               np.interp(obsv_test[-3], [-1, 1], [-25., 5.]),
                               np.interp(obsv_test[-2], [-1, 1], [0, 1199]),
                               obsv_test[-1] * 0.04]
            else:
                result = np.r_[np.array(env_test.forge.rod.current_temp),
                               np.interp(obsv_test[-3], [-1, 1], [-25., 5.]),
                               np.interp(obsv_test[-2], [-1, 1], [0, 1199]),
                               obsv_test[-1] * 0.04]
            print("The temperature, position, time and speed are: \n", [np.round(x, 3) for x in result])

        iteration += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='forgewhpomdp:forgewhpomdp-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--recurrent_seq_len', type=int, default=8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--clip_ratio', type=float, default=0.1)
    parser.add_argument('--entropy_factor', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--kl_con', type=float, default=0.5)
    parser.add_argument('--parallel_rollouts', type=int, default=32)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--wandb_log', type=bool, default=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed = args.seed
    seed += 10000
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(args.env)
    ac = LSTMActorCritic(env.observation_space, env.action_space, args.hid)

    pi_optimizer = Adam(ac.pi.parameters(), lr=args.pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=args.vf_lr)
    TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GATHER_DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    test_name = 'whpomdp_unlimitedwhpower'+date_str
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            test_name = test_name + a
    if args.wandb_log:
        wandb.login()
        wandb.init(sync_tensorboard=True, config=args, name=test_name, project="smart_forge")

    train_model(actor_critic=ac, actor_optimizer=pi_optimizer, critic_optimizer=vf_optimizer, epochs=args.epochs)