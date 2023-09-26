"""Implement Proximal Policy Optimization (PPO) framework. 
The code originated from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py."""

import os
import random
import time
import pickle
import copy
import numpy as np
import math

import wandb
import gymnasium as gym
import gin
import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import asdict
from torch.distributions.normal import Normal
from typing import Tuple, Optional
from .experiment_configs import PPOConfigs, PolicyTypes
from .mpc_policy import MPCWithLCSPolicy
from .learning_helper_functions import Adam, layer_init
from .lcs_model import LcsViolationBasedLoss

from .cassie_mpc_gym import CassieRolloutDataCollector
from .mpc_policy import MPCPolicy
from pathlib import Path
from datetime import datetime
import subprocess
import codecs

def GetCommandOutput(cmd, use_shell=False):
    process = subprocess.Popen(cmd, shell=use_shell, stdout=subprocess.PIPE)
    while process.poll() is None:  # while subprocess is alive
        time.sleep(0.1)

    output_bytes = process.communicate()[0]
    output_string = codecs.getdecoder("unicode_escape")(output_bytes)[0]
    return output_string

class NNCritic(nn.Module):
    def __init__(self, state_size: int, net_arch: Tuple[int]):
        super().__init__()
        net_arch = (state_size,) + net_arch + (1,)
        self._layers = []
        for i in range(len(net_arch) - 1):
            if i != len(net_arch) - 2:
                self._layers.append(layer_init(nn.Linear(net_arch[i], net_arch[i + 1])))
                self._layers.append(nn.Tanh())
            else:
                self._layers.append(layer_init(nn.Linear(net_arch[i], net_arch[i + 1]), std=1.0))
        self.critic = nn.Sequential(*self._layers)

    def get_value(self, x):
        return self.critic(x)


@gin.configurable
class PPO:
    def __init__(self, exp_configs: PPOConfigs):
        self.exp_configs = exp_configs
        self.set_random_seed(self.exp_configs.random_seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() and self.exp_configs.use_cuda else "cpu")

        # We need to increase the size of the data type, since we will do something like exp(130) (in b_logprobs[mb_inds].exp())
        torch.set_default_dtype(torch.float64)

        if self.exp_configs.sync_with_wandb:
            wandb.init(
                project=self.exp_configs.wandb_project_name,
                entity=self.exp_configs.wandb_entity,
                sync_tensorboard=True,
                config={
                    **asdict(self.exp_configs),
                },
                name=self.exp_configs.exp_name + f"_{self.exp_configs.policy_type}",
                save_code=True,
            )

        # instantiate policy
        if self.exp_configs.policy_type == "MPC":
            self.policy = MPCPolicy(device=self._device)
        else:
            raise ValueError("Invalid type of policy! It should only be 'PolicyTypes.NEURAL_NET' or 'PolicyTypes.MPC'")

        # Instantiate data collector (which contains both policy and "env")
        self.data_collector = CassieRolloutDataCollector(self.policy, self.exp_configs.exp_name, self.exp_configs.sync_with_wandb, cmaes_or_ppo="ppo")
        # Checking that policy is passed by reference
        # original_value = self.data_collector.policy.cur_params[0]
        # self.data_collector.policy.cur_params[0] = -1.98451
        # assert self.policy.cur_params[0] == -1.98451
        # self.data_collector.policy.cur_params[0] = original_value

        self.policy.cur_var = self.exp_configs.init_var * np.ones(self.policy.cur_var.size)
        self.data_collector.policy_output_noise_bound = self.exp_configs.policy_output_noise_bound

        self.obs_size = self.data_collector.obs_size
        self.action_size = self.data_collector.action_size

        # instantiate critic (value function)
        self.critic = NNCritic(self.obs_size, self.exp_configs.net_arch_value_func).to(self._device)

        (
            self._buffer_obs,
            self._buffer_next_obs,
            self._buffer_actions,
            self._buffer_logprobs,
            self._buffer_rewards,
            self._buffer_next_terminated,
            self._buffer_next_truncated,
            self._buffer_values,
            self._buffer_next_values
        ) = self._initialize_pytorch_buffers(
            max_buffer_size=self.exp_configs.sample_size,
            observation_size=self.obs_size,
            action_size=self.action_size,
            n_envs=self.exp_configs.num_envs,
            device=self._device
        )

        self._buffer_dict_rollouts = []

        assert self.exp_configs.num_envs == 1  # We don't use n_envs in our implementation to avoid bugs, because "bootstrap value" section also parallelize it
        if self.data_collector.randomize_tasks:
            assert self.exp_configs.sample_size >= self.data_collector.n_sample_needed_to_have_enough_number_of_tasks()
        else:
            assert self.exp_configs.sample_size >= self.data_collector.n_sample_needed_to_cover_all_tasks()  # We want to collect all tasks in every iteration's rollouts

        print("estimated training time per iteration: %.1f (hours)" % (self.exp_configs.sample_size / (1/self.data_collector.min_mpc_thread_loop_duration) / 60 / 60 * (1 + self.exp_configs.update_epochs)))


    def _initialize_pytorch_buffers(
            self, max_buffer_size: int, observation_size: int, action_size: int, n_envs: int, device: torch.device
    ) -> Tuple[torch.Tensor]:
        """Setup buffers to store rollout data from the current policy."""
        # Notes: the `n_envs` is just for parallelization
        # `buffer_obs` is a 3 dimensional array
        buffer_obs = torch.zeros(max_buffer_size, n_envs, observation_size).to(device)
        buffer_next_obs = torch.zeros(max_buffer_size, n_envs, observation_size).to(device)
        buffer_actions = torch.zeros(max_buffer_size, n_envs, action_size).to(device)
        buffer_logprobs = torch.zeros((max_buffer_size, n_envs)).to(device)
        buffer_rewards = torch.zeros((max_buffer_size, n_envs)).to(device)
        buffer_terminated = torch.zeros((max_buffer_size, n_envs)).to(device)
        buffer_truncated = torch.zeros((max_buffer_size, n_envs)).to(device)
        buffer_values = torch.zeros((max_buffer_size, n_envs)).to(device)
        buffer_next_values = torch.zeros((max_buffer_size, n_envs)).to(device)

        return (
            buffer_obs,
            buffer_next_obs,
            buffer_actions,
            buffer_logprobs,
            buffer_rewards,
            buffer_terminated,
            buffer_truncated,
            buffer_values,
            buffer_next_values
        )

    def set_random_seed(self, seed: int):
        """Set random seed to numpy and torch for reproducibility

        Args:
            seed (int): the random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(self, continue_train: bool = False, path_to_policy_chkpt: Optional[str] = None, path_to_critic_chkpt: Optional[str] = None):
        optimizer_policy = Adam(
            learning_rate=self.exp_configs.policy_learning_rate, epsilon=1e-5
        )  # TODO: how to use built-in Adam
        optimizer_var = Adam(
            learning_rate=self.exp_configs.policy_learning_rate, epsilon=1e-5
        )  # TODO: how to use built-in Adam
        optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.exp_configs.critic_learning_rate, eps=1e-5)

        global_step = 0  # total simulation steps from all envs
        init_iter = 1
        num_iterations = int(self.exp_configs.total_samples // self.exp_configs.sample_size)

        # Pick up from a previous training
        if continue_train:
            assert path_to_policy_chkpt is not None
            assert path_to_critic_chkpt is not None
        if path_to_policy_chkpt is not None:
            # Initialize policy
            with open(path_to_policy_chkpt, "rb") as f:
                loaded_params = pickle.load(f)
            assert loaded_params["params"].shape == self.policy.cur_params.shape
            self.policy.cur_params = loaded_params["params"]
            self.policy.cur_var = loaded_params["var"]
            # Set adam optimizer's state
            # TODO: test this
            optimizer_policy.set_mt(loaded_params["optimizer_policy_mt"])
            optimizer_policy.set_vt(loaded_params["optimizer_policy_vt"])
            optimizer_var.set_mt(loaded_params["optimizer_var_mt"])
            optimizer_var.set_vt(loaded_params["optimizer_var_vt"])
            # Initilize global_step
            init_iter = loaded_params["latest_iter_idx"]
            global_step = loaded_params["global_step"]
        if path_to_critic_chkpt is not None:
            # Initialize critic and its adam optimizer
            checkpoint = torch.load(path_to_critic_chkpt) if torch.cuda.is_available() else torch.load(path_to_critic_chkpt, map_location=torch.device('cpu'))
            if type(checkpoint) is dict:
                # TODO: have not tested this part yet
                self.critic.load_state_dict(checkpoint["model_state_dict"])
                optimizer_critic.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                self.critic.load_state_dict(checkpoint)

        # Start iterating
        for iteration in range(init_iter, num_iterations + 1):
            print("iteration = ", iteration, "; ", str(datetime.now()))
            print(GetCommandOutput("ip a | grep dynamic | grep brd", True))
            start_time = time.time()

            # Initialize for logging
            lrnow = -1.0
            v_loss = -1.0
            pg_loss = -1.0
            old_approx_kl = -1.0
            approx_kl = -1.0
            entropy_loss = -1.0
            clipfracs = []

            # annealing the rate if instructed to do so.
            if self.exp_configs.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lrnow = frac * self.exp_configs.policy_learning_rate
            else:
                lrnow = self.exp_configs.policy_learning_rate
            optimizer_policy.learning_rate = lrnow
            optimizer_var.learning_rate = lrnow
            optimizer_critic.param_groups[0]["lr"] = lrnow

            # Increase number of steps in the beginning here so that we log rollouts without and with noise together
            global_step += self.exp_configs.sample_size

            # For logging -- rollout once without noise policy noise to record the performance
            # Note that we have to run this before the actual rollout data collection step, since we need the folders (running reset_for_new_iteration() would remove the rollout data for the policy update step)
            self.data_collector.reset_for_new_iteration()
            self.record_performance(iteration, global_step, self.data_collector.run_multiple_rollouts(iteration, -1, False, True), without_noise=True)

            # rollout to collect training data
            self.data_collector.reset_for_new_iteration()
            list_of_dict = []
            while len(list_of_dict) < self.exp_configs.sample_size:
                list_of_dict = list_of_dict + self.data_collector.run_multiple_rollouts(iteration,
                                                                                        self.exp_configs.sample_size - len(
                                                                                            list_of_dict))
            # Store rollout data into buffers
            self._buffer_dict_rollouts.clear()
            for step in range(0, self.exp_configs.sample_size):
                dict_rollout = list_of_dict[step]

                self._buffer_obs[step] = torch.Tensor(dict_rollout['s']).to(self._device)
                self._buffer_actions[step] = torch.Tensor(dict_rollout['a']).to(self._device)
                self._buffer_next_obs[step] = torch.Tensor(dict_rollout['sp']).to(self._device)
                self._buffer_next_terminated[step] = torch.Tensor(dict_rollout['terminated']).to(self._device)
                self._buffer_next_truncated[step] = torch.Tensor(dict_rollout['truncated']).to(self._device)
                self._buffer_rewards[step] = torch.Tensor(dict_rollout['r']).to(self._device)
                self._buffer_values[step] = self.critic.get_value(torch.Tensor(dict_rollout['s']).to(self._device))
                self._buffer_next_values[step] = self.critic.get_value(torch.Tensor(dict_rollout['sp']).to(self._device))
                self._buffer_logprobs[step] = torch.Tensor(dict_rollout['log_prob']).to(self._device)

                self._buffer_dict_rollouts.append(dict_rollout)

            # For logging stats of the rollouts
            self.record_performance(iteration, global_step, list_of_dict, without_noise=False)

            # bootstrap value if not done
            with torch.no_grad():
                advantages = torch.zeros_like(self._buffer_rewards).to(self._device)
                lastgaelam = 0
                # Notes:
                #  - The algorithm here assume that the buffer is always full. Each episode doesn't necessarily have the same lengh
                #  - `_buffer_next_terminated` is of dimension n_env, but I assume n_env=1, so it doesn't matter.
                for t in reversed(range(self.exp_configs.sample_size)):
                    nextnonterminal = 1.0 - self._buffer_next_terminated[t]
                    nextvalues = self._buffer_next_values[t]
                    delta = (
                            self._buffer_rewards[t]
                            + self.exp_configs.gamma * nextvalues * nextnonterminal
                            - self._buffer_values[t]
                    )
                    advantages[t] = lastgaelam = (
                            delta + self.exp_configs.gamma * self.exp_configs.gae_lambda * nextnonterminal * lastgaelam
                    )
                # Note: `return` is the bootstrapped value of total reward. (better estimation of the value function (V(s)))
                returns = advantages + self._buffer_values

            # TODO: might need to take care of the order of self._buffer_dict_rollouts if n_env > 1
            # flatten the batch
            # Notes: the reshape here flatten the "max_buffer_size x n_envs" dimensions
            b_obs = self._buffer_obs.reshape((-1,) + (self.obs_size,))
            # b_next_obs = self._buffer_next_obs.reshape((-1,) + (self.obs_size,))
            b_logprobs = self._buffer_logprobs.reshape(-1)
            b_actions = self._buffer_actions.reshape((-1,) + (self.action_size,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self._buffer_values.reshape(-1)

            assert len(self._buffer_dict_rollouts) == len(b_values)

            # Optimizing the policy and value network
            b_inds = np.arange(self.exp_configs.sample_size)

            for epoch in range(self.exp_configs.update_epochs):
                print("epoch = ", epoch, "; ", str(datetime.now()))
                np.random.shuffle(b_inds)

                for start in range(
                        0,
                        self.exp_configs.sample_size,
                        self.exp_configs.minibatch_size,
                ):
                    end = start + self.exp_configs.minibatch_size
                    mb_inds = b_inds[start:end]

                    # Optimize policy loss for neural net policy
                    # We don't update the policy if it's the first iteration, because we want to initialize the value function first
                    # if iteration > 1:
                    if global_step > self.exp_configs.total_samples_for_value_initialization:
                        # Recompute the policy output and compute the ratio
                        (
                            traj_sols,
                            newaction,
                            newlogprob,
                            entropy,
                            action_mean,
                        ) = self.policy.get_action(b_obs[mb_inds], b_actions[mb_inds],
                                                   [self._buffer_dict_rollouts[i] for i in mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        # Only for logging purposes
                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [((ratio - 1.0).abs() > self.exp_configs.clip_coef).float().mean().item()]
                            entropy_loss = entropy.mean()

                        # TODO: check the math here, also double check variance vs std here
                        mb_advantages = b_advantages[mb_inds]
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio,
                            1 - self.exp_configs.clip_coef,
                            1 + self.exp_configs.clip_coef,
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean().item()

                        # convert all pytorch to numpy
                        newaction = newaction.cpu().numpy().reshape((mb_inds.shape[0], self.action_size, 1))
                        action_mean = action_mean.cpu().numpy().reshape((mb_inds.shape[0], self.action_size, 1))
                        mb_advantages = mb_advantages.cpu().numpy().reshape((mb_inds.shape[0], 1, 1))
                        ratio = ratio.cpu().numpy().reshape((mb_inds.shape[0], 1, 1))

                        # compute d_ratio / d_u
                        d_ratio_d_p = 1.0 / b_logprobs[mb_inds].exp()
                        d_ratio_d_p = d_ratio_d_p.cpu().numpy().reshape((mb_inds.shape[0], 1, 1))

                        # compute d_u / d_theta
                        # TODO: double check the dimension of d_u_d_theta
                        d_u_d_theta = self.policy.mpc_diff(traj_sols)

                        # Notes: prod() returns the product of all elements in the vector
                        d_p_d_theta = (((2 * np.pi) ** (-self.action_size / 2.0) * self.policy.cur_var.prod() ** (
                            -0.5) * np.exp(-0.5 * ((newaction - action_mean).reshape(
                            (mb_inds.shape[0], 1, self.action_size)) @ np.tile(np.diag(1.0 / self.policy.cur_var),
                                                                               (mb_inds.shape[0], 1, 1), ) @ (
                                                           newaction - action_mean)))) * (
                                               np.tile(np.diag(1.0 / self.policy.cur_var),
                                                       (mb_inds.shape[0], 1, 1), ) @ (
                                                       newaction - action_mean)).reshape(
                            (mb_inds.shape[0], 1, self.action_size)) @ d_u_d_theta)

                        # compute d_p / d_var
                        d_p_d_var = []

                        for i in range(self.action_size):
                            d_p_d_var_i = (
                                    self.policy.cur_var.prod() ** (-0.5) * self.policy.cur_var[i] ** (-2) * 2 ** (
                                    -self.action_size / 2.0 - 1) * np.pi ** (-self.action_size / 2) * np.exp(
                                -0.5 * ((newaction - action_mean).reshape(
                                    (mb_inds.shape[0], 1, self.action_size)) @ np.tile(
                                    np.diag(1.0 / self.policy.cur_var), (mb_inds.shape[0], 1, 1), ) @ (
                                                newaction - action_mean))) * (
                                            (newaction - action_mean)[:, [i], :] ** 2 - self.policy.cur_var[i]))
                            d_p_d_var.append(d_p_d_var_i)

                        d_p_d_var = np.concatenate(d_p_d_var, axis=2)

                        # combine gradients using chain rule
                        non_neg_mb_advantage_indices = np.where(mb_advantages >= 0)[0]
                        neg_mb_advantage_indices = np.where(mb_advantages < 0)[0]

                        d_loss_d_theta_non_neg_adv = (
                                (ratio[non_neg_mb_advantage_indices] <= 1 + self.exp_configs.clip_coef)
                                * d_ratio_d_p[non_neg_mb_advantage_indices]
                                * d_p_d_theta[non_neg_mb_advantage_indices]
                                * mb_advantages[non_neg_mb_advantage_indices]
                        )

                        d_loss_d_theta_neg_adv = (
                                (ratio[neg_mb_advantage_indices] >= 1 - self.exp_configs.clip_coef)
                                * d_ratio_d_p[neg_mb_advantage_indices]
                                * d_p_d_theta[neg_mb_advantage_indices]
                                * mb_advantages[neg_mb_advantage_indices]
                        )

                        d_loss_d_theta = np.concatenate(
                            [
                                d_loss_d_theta_neg_adv,
                                d_loss_d_theta_non_neg_adv,
                            ],
                            axis=0,
                        ).mean(axis=0)

                        d_loss_d_var_non_neg_adv = (
                                (ratio[non_neg_mb_advantage_indices] <= 1 + self.exp_configs.clip_coef)
                                * d_ratio_d_p[non_neg_mb_advantage_indices]
                                * d_p_d_var[non_neg_mb_advantage_indices]
                                * mb_advantages[non_neg_mb_advantage_indices]
                        )

                        d_loss_d_var_neg_adv = (
                                (ratio[neg_mb_advantage_indices] >= 1 - self.exp_configs.clip_coef)
                                * d_ratio_d_p[neg_mb_advantage_indices]
                                * d_p_d_var[neg_mb_advantage_indices]
                                * mb_advantages[neg_mb_advantage_indices]
                        )

                        d_loss_d_var = np.concatenate(
                            [d_loss_d_var_neg_adv, d_loss_d_var_non_neg_adv],
                            axis=0,
                        ).mean(axis=0)

                        total_gradient = -self.exp_configs.ppo_loss_coef * d_loss_d_theta

                        # update the parameters
                        self.policy.cur_params = optimizer_policy.step(
                            self.policy.cur_params, total_gradient
                        ).ravel()
                        self.policy.cur_var = optimizer_var.step(self.policy.cur_var, -d_loss_d_var).ravel()
                        # lower bound variance since it has to be PSD
                        # also upper bound variance, since I'm limiting the mpc output noise anyway
                        min_var = 1e-10 * np.ones_like(self.policy.cur_var)
                        max_var = self.exp_configs.max_var * np.ones_like(self.policy.cur_var)
                        self.policy.cur_var = np.clip(self.policy.cur_var, a_min=min_var, a_max=max_var)

                    # perform regression on value function (critic)
                    newvalue = self.critic.get_value(b_obs[mb_inds])
                    newvalue = newvalue.view(-1)
                    v_loss = self.exp_configs.vf_coef * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    optimizer_critic.zero_grad()
                    v_loss.backward()
                    optimizer_critic.step()

            # Notes: RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
            y_pred, y_true = b_values.detach().cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y   # estimate how good the value function is

            if self.exp_configs.sync_with_wandb:
                wandb.log(
                    {
                        "learning_rate": lrnow,
                        "value_loss": v_loss,
                        "policy_loss": pg_loss,
                        "entropy": entropy_loss,
                        "old_approx_kl": old_approx_kl,
                        "approx_kl": approx_kl,
                        "clipfrac": np.mean(clipfracs),
                        "explained_variance": explained_var,
                        "SPS": int(self.exp_configs.sample_size / (time.time() - start_time)),
                        "global_step": global_step,
                        "iteration": iteration,
                    }
                )

            # save checkpoints for policy and critic models
            # chkpt_folder = os.path.normpath(os.path.join(os.path.dirname(__file__), f"../data/checkpoints/{self.exp_configs.exp_name}",))
            chkpt_folder = self.data_collector.dir_rl + "checkpoints/"
            if not os.path.exists(chkpt_folder):
                Path(chkpt_folder).mkdir(parents=True, exist_ok=True)

            critic_path = os.path.join(chkpt_folder, f"critic_{global_step}.pt")
            policy_path = os.path.join(chkpt_folder, f"policy_{self.exp_configs.policy_type}_{global_step}.pt",)

            torch.save({"model_state_dict": self.critic.state_dict(),
                        "optimizer_state_dict": optimizer_critic.state_dict()}, critic_path)

            with open(policy_path, "wb") as f:
                pickle.dump(
                    {
                        "params": self.policy.cur_params,
                        "var": self.policy.cur_var,
                        "latest_iter_idx": iteration,
                        "global_step": global_step,
                        "optimizer_policy_mt": optimizer_policy.get_mt(),
                        "optimizer_policy_vt": optimizer_policy.get_vt(),
                        "optimizer_var_mt": optimizer_var.get_mt(),
                        "optimizer_var_vt": optimizer_var.get_vt(),
                    },
                    f,
                )

    def record_performance(self, iteration, global_step, list_of_dict, without_noise):
        n_steps = len(list_of_dict) if without_noise else self.exp_configs.sample_size

        # Compute rewards
        average_reward_per_episode_for_logging = []
        average_reward_main_per_episode_for_logging = []
        average_reward_task_per_episode_for_logging = []
        average_cost_main_per_episode_for_logging = []
        episode_length_for_logging = []
        sampled_reward = 0.0
        sampled_reward_main = 0.0
        sampled_reward_task = 0.0
        sampled_cost_main = 0.0
        step_idx_within_episode = 0
        for step in range(0, n_steps):
            dict_rollout = list_of_dict[step]

            sampled_reward += dict_rollout['r']
            sampled_reward_main += dict_rollout['r_breakdown']["r_main_obj"]
            sampled_reward_task += dict_rollout['r_breakdown']["r_tracking"]
            sampled_cost_main += -math.log(dict_rollout['r_breakdown']["r_main_obj"])
            step_idx_within_episode += 1
            if dict_rollout['terminated'] or dict_rollout['truncated']:
                average_reward_per_episode_for_logging.append(sampled_reward / step_idx_within_episode)
                average_reward_main_per_episode_for_logging.append(sampled_reward_main / step_idx_within_episode)
                average_reward_task_per_episode_for_logging.append(sampled_reward_task / step_idx_within_episode)
                average_cost_main_per_episode_for_logging.append(sampled_cost_main / step_idx_within_episode)
                episode_length_for_logging.append(step_idx_within_episode)
                sampled_reward = 0.0
                sampled_reward_main = 0.0
                sampled_reward_task = 0.0
                sampled_cost_main = 0.0
                step_idx_within_episode = 0

        # For logging
        if self.exp_configs.sync_with_wandb:
            if without_noise:
                wandb.log(
                    {
                        "rewards_average_wo_noise": np.average(np.array(average_reward_per_episode_for_logging)),
                        "main_rewards_average_wo_noise": np.average(np.array(average_reward_main_per_episode_for_logging)),
                        "task_rewards_average_wo_noise": np.average(np.array(average_reward_task_per_episode_for_logging)),
                        "main_cost_average_wo_noise": np.average(np.array(average_cost_main_per_episode_for_logging)),
                        "global_step": global_step,
                        "iteration": iteration,
                    }
                )
            else:
                wandb.log(
                    {
                        "rewards_average": np.average(np.array(average_reward_per_episode_for_logging)),
                        "rewards_std": np.std(np.array(average_reward_per_episode_for_logging)),
                        "main_rewards_average": np.average(np.array(average_reward_main_per_episode_for_logging)),
                        "main_rewards_std": np.std(np.array(average_reward_main_per_episode_for_logging)),
                        "task_rewards_average": np.average(np.array(average_reward_task_per_episode_for_logging)),
                        "task_rewards_std": np.std(np.array(average_reward_task_per_episode_for_logging)),
                        "episodic_length_average": np.average(np.array(episode_length_for_logging)),
                        "episodic_length_std": np.std(np.array(episode_length_for_logging)),
                        "var_median": np.median(self.policy.cur_var),
                        "global_step": global_step,
                        "iteration": iteration,
                    }
                )
