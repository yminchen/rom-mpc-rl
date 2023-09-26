import json
import optuna
import torch
import numpy as np
import scipy
from scipy import linalg
import math
import wandb
import datetime
import time
import os

import traceback
import logging

from functools import partial
from common.cassie_mpc_gym import CassieRolloutDataCollector
from common.mpc_policy import MPCPolicy


# Turn off optuna log notes.
optuna.logging.set_verbosity(optuna.logging.WARN)

# Wandb Parameter
sync_with_wandb = True
wandb_entity = "username"
wandb_project_name = "cassie-rom-mpc-cmaes"
experiment_name = wandb_project_name + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Path to store the best set of parameters
DIR_DAIRLIB = "/home/"+ os.getlogin() + "/workspace/dairlib/"
DIR_CMAES = DIR_DAIRLIB + "../dairlib_data/goldilocks_models/rl_training/" + experiment_name + "/"
PATH_TO_STORE_BEST_PARAMS = DIR_CMAES + "best_params.json"

# CMA parameter
init_sigma = 1e-3
n_total_trials = 30000  # use 10k instead of 30k, just in case we might need more trials
restart_strategy = None #'ipop'
inc_popsize = 2  # optuna default to 0

# Value calculation parameters
average_reward_over_number_of_samples = False
average_reward_over_number_of_episodes = False


class CMAES:
    def __init__(self):
        # Initialize wandb
        if sync_with_wandb:
            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                config={
                    "n_total_trials": n_total_trials,
                    "init_sigma": init_sigma,
                    "restart_strategy": "none" if restart_strategy is None else restart_strategy,
                    "inc_popsize": inc_popsize,
                    "average_reward_over_number_of_samples": average_reward_over_number_of_samples,
                    "average_reward_over_number_of_episodes": average_reward_over_number_of_episodes,
                },
                name=experiment_name,
                save_code=True,
            )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = MPCPolicy(device=self._device)
        self.data_collector = CassieRolloutDataCollector(
            self.policy, project_name=experiment_name, sync_with_wandb=sync_with_wandb, cmaes_or_ppo="cmaes"
        )

        # Paramter buffer for logging covariance 
        self.recent_param_list = []
        self.pop_size = 4 + math.floor(3 * math.log(len(self.policy.cur_params)))  # CMA parameter

        # Some setup
        self.cma_eval_idx = -1  # doesn't start with 0, because I couln't get rid of the one-time random eval in the beginning
        self.cma_iter = 0
        self.successful_task_list_indices = []
        self.best_param_buffer = []
        self.previous_time = time.time()

        # Get the initial global task indices for logging
        if self.data_collector.curriculum_learning:
            _, sampled_task_idx_to_full_task_idx_map = self.data_collector.filter_for_active_task_list()  
            self.full_task_idx_of_initial_tasks = sampled_task_idx_to_full_task_idx_map

        # Printing and logging flags for debugging
        if self.data_collector.curriculum_learning:
            self.data_collector.task_sampler.print_task_active_flags(self.data_collector.dir_rl)

        # Assertions
        if self.data_collector.curriculum_learning:
            # We don't want the task space to grow too slowly and we never get to reach the end
            max_task_grid_length = max([self.data_collector.tasks.GetSizeByDimIdx(i) for i in range(self.data_collector.tasks.get_task_dim())])
            assert int(max_task_grid_length / 2) * self.data_collector.n_cma_iter_per_task_region_grow * self.pop_size < 10000


    def run_experiment(self, init_param_file=""):
        """Create and run Optuna study, in particular, find set of parameters that maximize the task performance.
        """
        if len(init_param_file) > 0:
            self.policy.cur_params = np.loadtxt(init_param_file)

        init_param = {}
        for i in range(len(self.policy.cur_params)):
            init_param[f"Param_{i}"] = self.policy.cur_params[i]

        self.sampler = optuna.samplers.CmaEsSampler(x0=init_param, sigma0=init_sigma, n_startup_trials=0, restart_strategy=restart_strategy, inc_popsize=inc_popsize)
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(partial(self._objective), n_trials=n_total_trials, callbacks=[self._save_best_params_callback])

    # Callback is called after every evaluation of objective, and it takes Study and FrozenTrial as arguments, and does some work.
    # Ref: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html
    @staticmethod
    def _save_best_params_callback(study, frozen_trial):
        print("=== Callback ===")

        """Save the parameters to file whenever Optuna discovers new parameters that yield new best performance."""
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        # import pdb;pdb.set_trace()
        if study.trials[0].value is None:  # I cannot get rid of the initial random sampler even after I specify n_startup_trials=0.
            return
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            print(
                "Trial {} finished with best value: {} and parameters: {}. ".format(
                    frozen_trial.number,
                    frozen_trial.value,
                    frozen_trial.params,
                )
            )
            # save the best parameters into txt file
            with open(PATH_TO_STORE_BEST_PARAMS, "w") as f:
                json.dump(
                    {
                        "trial_id": frozen_trial.number,
                        "performance": frozen_trial.value,
                        "best_params": CMAES._to_list(frozen_trial.params),
                    },
                    f,
                )

    def _objective(self, trial):
        # Normal CMA trials: `cma_eval_idx` ranges [1, 2, ..., self.pop_size]
        # Normal CMA trials: `cma_iter` ranges [1, 2, ...]
        # In the very first trial (NOT a normal CMA trial; random param): `cma_eval_idx` is 0, and `cma_iter` is 0
        if self.cma_eval_idx == self.pop_size:
            self.cma_eval_idx = 0
        self.cma_eval_idx += 1
        if self.cma_eval_idx == 1:
            self.cma_iter += 1
        print("\n\n\n")
        print("=== Obj eval #%d ===" % self.cma_eval_idx)

        # optuna samples parameters within the range
        n_params = self.policy.cur_params.shape[0]
        upper_bound = 2.0
        lower_bound = -2.0
        sampled_params = np.empty(shape=(n_params,))
        # import pdb;pdb.set_trace()

        for i in range(n_params):
            sampled_params[i] = trial.suggest_float(f"Param_{i}", lower_bound, upper_bound)

        # after optuna chooses values for all parameters, we need to update the policy's parameters
        self.policy.cur_params = sampled_params.copy()
        # import pdb;pdb.set_trace()

        # with new set of parameters, collect rollouts and evaluate the performance of the MPC
        self.data_collector.reset_for_new_iteration()
        rollouts = self.data_collector.run_multiple_rollouts(
            iter_idx=trial.number,
            n_sample_needed=-1,
            construct_planner_only=False,
            rollout_without_policy_noise=True,
            use_previous_shuffled_task_idx_list=(self.cma_eval_idx>1),
        )
        value = self.evaluate_performance(rollouts, trial.number)

        regular_cma_evaluation = self.cma_eval_idx > 0
        if regular_cma_evaluation:
            if self.data_collector.curriculum_learning:
                self.update_task_space_for_curriculum_learning(rollouts, trial.number)

            self.best_param_buffer.append({"trial_iter": trial.number, "value": value, "param": sampled_params.copy()})
            self.save_best_param_per_cma_iter()
            self.some_logging()

            # Log parameter covariance
            self.computer_and_log_covariance(trial.number)

        return value


    def save_best_param_per_cma_iter(self):
        if self.cma_eval_idx == self.pop_size:
            argmax = np.array([d["value"] for d in self.best_param_buffer]).argmax()
            np.savetxt("%s%d_theta_yddot.csv" % (DIR_CMAES, self.cma_iter), self.best_param_buffer[argmax]["param"], delimiter=",")
            if sync_with_wandb:
                wandb.log(
                    {
                        "best_value_per_cma_iter": self.best_param_buffer[argmax]["value"],
                        "eval iteration": self.best_param_buffer[argmax]["trial_iter"],
                        "cma iteration": self.cma_iter,
                    }
                )
            self.best_param_buffer.clear()

    def some_logging(self):
        if self.cma_eval_idx == self.pop_size:
            if sync_with_wandb:
                wandb.log(
                    {
                        "duration per cma iter (in mins)": (time.time()-self.previous_time)/60,
                        "cma iteration": self.cma_iter,
                    }
                )
                self.previous_time = time.time()

    # I intend to use `computer_and_log_covariance` to see if the optimization has converged
    def computer_and_log_covariance(self, iter_idx):
        if sync_with_wandb:
            self.recent_param_list.append(self.policy.cur_params)
            if len(self.recent_param_list) > self.pop_size * 10:
                self.recent_param_list.pop(0)

                cov = np.cov(np.array(self.recent_param_list).T)
                assert cov.shape[0] == len(self.policy.cur_params) # sanity check

                s = linalg.svd(cov, compute_uv=False)
                param_std_max = math.sqrt(s[0])

                # Logging
                wandb.log(
                    {
                        "param_std_max": param_std_max,
                        "eval iteration": iter_idx,
                    }
                )


    @staticmethod
    def _to_list(optuna_params):
        """Convert Optuna params of Dict type to list."""
        n_params = len(optuna_params.keys())
        params = []

        for i in range(n_params):
            params.append(optuna_params[f"Param_{i}"])
        return params


    def update_task_space_for_curriculum_learning(self, list_of_dict, iter_idx):
        # `sampled_task_idx_in_full_task_list` is just for debugging (print to visualize the sampled tasks)
        sampled_task_idx_in_full_task_list = np.zeros((len(self.data_collector.task_sampler.get_full_task_list())), dtype=bool)

        for step in range(0, len(list_of_dict)):
            dict_rollout = list_of_dict[step]
            if dict_rollout["truncated"]:
                self.successful_task_list_indices.append(dict_rollout["extra_info"]["full_task_idx"])
                sampled_task_idx_in_full_task_list[dict_rollout["extra_info"]["full_task_idx"]] = True
        
        # Printing for debugging
        print("Sampled tasks visualization:")
        self.data_collector.task_sampler.print_full_task_list_flags(sampled_task_idx_in_full_task_list, write_to_file=False)


        if (self.cma_iter % self.data_collector.n_cma_iter_per_task_region_grow == 0) and (self.cma_eval_idx == self.pop_size):  # we only change the task space per `n_cma_iter_per_task_region_grow` number of cma iterations
            for task_idx in list(set(self.successful_task_list_indices)):
                self.data_collector.task_sampler.update_successful_flags(task_idx)
            self.successful_task_list_indices.clear()

            # Printing and logging flags for debugging
            print("New active task set visualization:")
            self.data_collector.task_sampler.print_task_active_flags(self.data_collector.dir_rl)


    def evaluate_performance(self, list_of_dict, iter_idx):
        n_steps = len(list_of_dict)

        if n_steps == 0:
            return 0

        # Compute rewards
        accumulated_reward_per_episode_for_logging = []
        average_reward_per_episode_for_logging = []
        average_reward_main_per_episode_for_logging = []
        average_reward_main_accel_per_episode_for_logging = []
        average_reward_main_torque_per_episode_for_logging = []
        average_reward_task_per_episode_for_logging = []
        average_reward_task_vel_per_episode_for_logging = []
        average_reward_task_height_per_episode_for_logging = []
        # average_cost_main_per_episode_for_logging = []
        average_cost_main_accel_per_episode_for_logging = []
        average_cost_main_torque_per_episode_for_logging = []
        average_reward_main_torque_per_episode_initial_tasks_for_logging = []
        average_reward_task_height_per_episode_initial_tasks_for_logging = []
        episode_length_for_logging = []
        sampled_reward = 0.0
        sampled_reward_main = 0.0
        sampled_reward_main_accel = 0.0
        sampled_reward_main_torque = 0.0
        sampled_reward_task = 0.0
        sampled_reward_task_vel = 0.0
        sampled_reward_task_height = 0.0
        # sampled_cost_main = 0.0
        sampled_cost_main_accel = 0.0
        sampled_cost_main_torque = 0.0
        step_idx_within_episode = 0
        for step in range(0, n_steps):
            dict_rollout = list_of_dict[step]

            sampled_reward += dict_rollout["r"]
            sampled_reward_main += dict_rollout["r_breakdown"]["r_main_obj"]
            sampled_reward_main_accel += dict_rollout["r_breakdown"]["r_accel"]
            sampled_reward_main_torque += dict_rollout["r_breakdown"]["r_torque"]
            sampled_reward_task += dict_rollout["r_breakdown"]["r_tracking"]
            sampled_reward_task_vel += dict_rollout["r_breakdown"]["r_vel_tracking"]
            sampled_reward_task_height += dict_rollout["r_breakdown"]["r_height_tracking"]
            # sampled_cost_main += -math.log(dict_rollout["r_breakdown"]["r_main_obj"])
            sampled_cost_main_accel += -math.log(max(1e-323, dict_rollout["r_breakdown"]["r_accel"]))
            sampled_cost_main_torque += -math.log(max(1e-323, dict_rollout["r_breakdown"]["r_torque"]))
            step_idx_within_episode += 1
            if dict_rollout["terminated"] or dict_rollout["truncated"]:
                accumulated_reward_per_episode_for_logging.append(sampled_reward)
                average_reward_per_episode_for_logging.append(sampled_reward / step_idx_within_episode)
                average_reward_main_per_episode_for_logging.append(sampled_reward_main / step_idx_within_episode)
                average_reward_main_accel_per_episode_for_logging.append(sampled_reward_main_accel / step_idx_within_episode)
                average_reward_main_torque_per_episode_for_logging.append(sampled_reward_main_torque / step_idx_within_episode)
                average_reward_task_per_episode_for_logging.append(sampled_reward_task / step_idx_within_episode)
                average_reward_task_vel_per_episode_for_logging.append(sampled_reward_task_vel / step_idx_within_episode)
                average_reward_task_height_per_episode_for_logging.append(sampled_reward_task_height / step_idx_within_episode)
                # average_cost_main_per_episode_for_logging.append(sampled_cost_main / step_idx_within_episode)
                average_cost_main_accel_per_episode_for_logging.append(sampled_cost_main_accel / step_idx_within_episode)
                average_cost_main_torque_per_episode_for_logging.append(sampled_cost_main_torque / step_idx_within_episode)
                episode_length_for_logging.append(step_idx_within_episode)
                if self.data_collector.curriculum_learning:
                    if dict_rollout["extra_info"]["full_task_idx"] in self.full_task_idx_of_initial_tasks:
                        average_reward_main_torque_per_episode_initial_tasks_for_logging.append(sampled_reward_main_torque / step_idx_within_episode)
                        average_reward_task_height_per_episode_initial_tasks_for_logging.append(sampled_reward_task_height / step_idx_within_episode)

                sampled_reward = 0.0
                sampled_reward_main = 0.0
                sampled_reward_main_accel = 0.0
                sampled_reward_main_torque = 0.0
                sampled_reward_task = 0.0
                sampled_reward_task_vel = 0.0
                sampled_reward_task_height = 0.0
                # sampled_cost_main = 0.0
                sampled_cost_main_accel = 0.0
                sampled_cost_main_torque = 0.0
                step_idx_within_episode = 0

        # Value
        # Note that we maximize objective function because of optuna.create_study(direction="maximize").
        values_of_all_episodes = average_reward_per_episode_for_logging if average_reward_over_number_of_samples else accumulated_reward_per_episode_for_logging
        value = 0.0
        if average_reward_over_number_of_episodes:
            value = np.average(np.array(values_of_all_episodes))  
        else:
            value = np.sum(np.array(values_of_all_episodes)) / self.data_collector.n_tasks_per_value_evaluation()

        # Extra data
        height_error_list = []
        for dict_rollout in list_of_dict:
            height_error_list.append(dict_rollout["extra_logging"]["abs_height_error"])

        # Logging
        if sync_with_wandb:
            wandb.log(
                {
                    "value": value,
                    "rewards_average_wo_noise": np.average(np.array(average_reward_per_episode_for_logging)),
                    "main_rewards_average_wo_noise": np.average(np.array(average_reward_main_per_episode_for_logging)),
                    "accel_rewards_average_wo_noise": np.average(np.array(average_reward_main_accel_per_episode_for_logging)),
                    "torque_rewards_average_wo_noise": np.average(np.array(average_reward_main_torque_per_episode_for_logging)),
                    "task_rewards_average_wo_noise": np.average(np.array(average_reward_task_per_episode_for_logging)),
                    "vel_task_rewards_average_wo_noise": np.average(np.array(average_reward_task_vel_per_episode_for_logging)),
                    "height_task_rewards_average_wo_noise": np.average(np.array(average_reward_task_height_per_episode_for_logging)),
                    # "main_cost_average_wo_noise": np.average(np.array(average_cost_main_per_episode_for_logging)),
                    "accel_cost_average_wo_noise": np.average(np.array(average_cost_main_accel_per_episode_for_logging)),
                    "torque_cost_average_wo_noise": np.average(np.array(average_cost_main_torque_per_episode_for_logging)),
                    "init_tasks_torque_rewards_average_wo_noise": np.average(np.array(average_reward_main_torque_per_episode_initial_tasks_for_logging)),
                    "init_tasks_height_task_rewards_average_wo_noise": np.average(np.array(average_reward_task_height_per_episode_initial_tasks_for_logging)),
                    "ave_height_error": np.average(np.array(height_error_list)),
                    "eval iteration": iter_idx,
                }
            )

        return value



if __name__ == "__main__":
    cmaes_optimizer = CMAES()
    cmaes_optimizer.run_experiment()
    # cmaes_optimizer.run_experiment("/home/username/workspace/dairlib_data/goldilocks_models/planning/robot_1/good_rl_param/cassie-rom-mpc-cmaes20230813_110116/1857_theta_yddot.csv")
