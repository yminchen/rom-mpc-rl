"""A thin wrapper around DrakeSystems"""
import numpy as np
import warnings
from scipy.stats import multivariate_normal

from typing import Callable, Union, Optional

import yaml
import sys
import subprocess
from subprocess import DEVNULL
import time
import os
from pathlib import Path
from datetime import datetime
import psutil
import copy  # for deepcopying a list
import csv
import codecs
import math

import wandb

from common.mpc_policy import MPCPolicy


class CassieRolloutDataCollector:
    def __init__(self, policy, project_name, sync_with_wandb, cmaes_or_ppo):
        assert not (":" in project_name)
        self.sync_with_wandb = sync_with_wandb

        self.dir_dairlib = '/home/' + os.getlogin() + '/workspace/dairlib/'
        self.dir_rl = self.dir_dairlib + "../dairlib_data/goldilocks_models/rl_training/" + project_name + "/"
        self.dir_planner_data = self.dir_rl + "temp_sample_data/"
        self.dir_history_data = self.dir_rl + "history_data/"

        # RL parameters
        self.t_no_logging_at_start = 5.0
        self.tol_determine_last_state = 0.1  # in s
        self.min_mpc_thread_loop_duration = 0.05
        self.policy_output_noise_bound = 0.001  # will be overwritten by the cassie_optimal_rom_ppo_config.gin

        # Other high level parameter
        self.use_acceleration_cost_instead_of_torque = False
        self.w_tracking = 0.5
        self.w_torque = 1.0
        self.w_accel = 0.0
        self.exp_scaling_tracking_height = 1000.0
        self.exp_scaling_tracking_vel = 10.0
        self.exp_scaling_accel = 1.0 / 10000
        self.exp_scaling_torque = 1.0 / 8000
        self.include_previous_vel_in_rl_state = False
        # Parameters for tasks (randomize_tasks, curriculum learning)
        self.randomize_tasks = False
        self.curriculum_learning = False
        self.min_n_rollouts_per_iteration = 25  # Used for ppo; We don't want too few rollouts (too few sampled tasks)
        self.target_n_rollouts_per_iteration = 10  #25      # Used for cma-es + non-curriculum-learning
        self.min_task_sample_density_per_iteration = 1/10   # Used for cma-es + curriculum-learning; We don't want too low density, otherwise it wouldn't sample the boundary to expand
        self.max_n_rollouts_per_iteration = 176  #18        # Used for cma-es + curriculum-learning; Set a maximum, otherwise it takes too long
        self.n_cma_iter_per_task_region_grow = 30

        # CMA or PPO
        assert (cmaes_or_ppo == "ppo") or (cmaes_or_ppo == "cmaes")
        self.cmaes_or_ppo = cmaes_or_ppo

        # Collect data using lcm-logger
        self.collect_rewards_via_lcmlogs = False
        assert not self.collect_rewards_via_lcmlogs # not done implemeting

        # Visualizer
        self.run_visualizer = False

        #####################################################################################################################

        # Read the controller parameters
        a_yaml_file = open(self.dir_dairlib + "examples/goldilocks_models/rom_walking_gains.yaml")
        parsed_yaml_file = yaml.safe_load(a_yaml_file)
        self.model_dir = self.dir_dairlib + parsed_yaml_file.get('dir_model')

        self.FOM_model_dir_for_sim = "/home/"+ os.getlogin() + "/workspace/dairlib_data/goldilocks_models/planning/robot_1/20220511_explore_task_boundary_2D--20220417_rom27_big_torque/robot_1/"
        self.FOM_model_dir_for_planner = ""

        ### global parameters
        self.sim_end_time = 10.0
        self.spring_model = False
        self.close_sim_gap = False
        # Parameters that are modified often
        self.target_realtime_rate = 1  # 0.04
        self.init_sim_vel = True
        self.set_sim_init_state_from_trajopt = True   # use IK or provide a path to initial sim state
        use_single_cost_function_for_all_tasks = False

        # planner arguments
        self.realtime_rate_for_time_limit = self.target_realtime_rate
        self.use_ipopt = True
        self.knots_per_mode = 4  # can try smaller number like 3 or 5
        self.feas_tol = 1e-4
        self.opt_tol = 1e-4
        self.n_step = 2

        # Parameter adjustment
        if self.spring_model:
            assert not self.close_sim_gap
        if self.set_sim_init_state_from_trajopt:
            if len(self.FOM_model_dir_for_sim) == 0:
                raise ValueError("Need to specify the path for the FOM trjaopt")

        ### Parameter for multithreading algorithm here
        # Notes: Sometimes the whole multithread algorithm got stuck, and we can terminate a thread if it's spending too long.
        #        `max_time_for_a_thread` is a workaround for unknown bugs that sometimes planner or controller are waiting to receive message
        self.max_time_for_a_thread = 30.0

        # Parameter adjustment
        self.max_time_for_a_thread /= self.target_realtime_rate
        self.max_time_for_a_thread = max(self.max_time_for_a_thread, self.sim_end_time/self.target_realtime_rate + 15)  # give 15 seconds wall time of buffer

        ### Parameters for task
        # Initial task region (only for curriculum learning)
        self.initial_task_box_range = {"stride_length": [-0.25, 0.25], "ground_incline": [-0.05, 0.05], "turning_rate": [-0.25, 0.25]}

        # Task list
        if self.curriculum_learning:
            n_task_sl = 8  #13 #25 #10
            n_task_ph = 1  #25 #3
            n_task_tr = 9  #13 #25 #3
            n_task_gi = 13  #13 #25 #3

            # Note: The range here should include the hard task, because we are doing curriculum learning (gradually increase task space)
            self.tasks = Tasks()
            self.tasks.AddTaskDim(np.linspace(-0.1, 0.6, n_task_sl), "stride_length")
            self.tasks.AddTaskDim(np.linspace(-0.6, 0.6, n_task_gi), "ground_incline")
            self.tasks.AddTaskDim([-1.0], "duration")  # assign later; this shouldn't be a task for sim evaluation
            self.tasks.AddTaskDim(np.linspace(-1.8, 1.8, n_task_tr), "turning_rate")
            self.tasks.AddTaskDim([0.85], "pelvis_height")
            self.tasks.AddTaskDim([0.03], "swing_margin")  # This is not being used.

        else:
            n_task_sl = 5  #13 #25 #10
            n_task_ph = 1  #25 #3
            n_task_tr = 5  #13 #25 #3
            n_task_gi = 5  #13 #25 #3
            if self.randomize_tasks:
                n_task_sl = 30
                n_task_ph = 30
                n_task_tr = 30
                n_task_gi = 30
            self.tasks = Tasks()
            # self.tasks.AddTaskDim(np.linspace(-0.6, 0.6, n_task_sl), "stride_length")
            # self.tasks.AddTaskDim(np.linspace(-0.42, 0.42, n_task_sl), "stride_length")
            self.tasks.AddTaskDim(np.linspace(-0.3, 0.3, n_task_sl), "stride_length")
            # self.tasks.AddTaskDim(np.linspace(0, 0.2, n_task_sl), "stride_length")
            # self.tasks.AddTaskDim(np.linspace(-0.2, 0.2, n_task_sl), "stride_length")
            # self.tasks.AddTaskDim(np.linspace(0.3, 0.3, n_task_sl), "stride_length")
            # self.tasks.AddTaskDim(np.linspace(0, 0, n_task_sl), "stride_length")
            # stride_length = np.linspace(-0.2, -0.1, n_task)
            # stride_length = np.linspace(-0.3, 0, n_task, endpoint=False)
            # stride_length = np.linspace(0.4, 0.5, n_task)
            # stride_length = np.hstack([np.linspace(-0.6, -0.4, n_task, endpoint=False),
            #                            -np.linspace(-0.6, -0.4, n_task, endpoint=False)])
            self.tasks.AddTaskDim([0.0], "ground_incline")
            # self.tasks.AddTaskDim(np.linspace(-0.4, 0.4, n_task_gi), "ground_incline")
            # self.tasks.AddTaskDim(np.linspace(-0.3, 0.3, n_task_gi), "ground_incline")
            # self.tasks.AddTaskDim(np.linspace(-0.05, -0.05, n_task_gi), "ground_incline")
            self.tasks.AddTaskDim([-1.0], "duration")  # assign later; this shouldn't be a task for sim evaluation
            self.tasks.AddTaskDim([0.0], "turning_rate")
            # self.tasks.AddTaskDim(np.linspace(-0.6, 0.6, n_task_tr), "turning_rate")
            # self.tasks.AddTaskDim(np.linspace(-1.0, 1.0, n_task_tr), "turning_rate")
            # pelvis_heights used in both simulation and in CollectAllTrajoptSampleIndices
            # self.tasks.AddTaskDim(np.linspace(0.85, 1.05, n_task_ph), "pelvis_height")
            # self.tasks.AddTaskDim(np.linspace(0.5, 1.1, n_task_ph), "pelvis_height")
            # self.tasks.AddTaskDim(np.linspace(0.5, 1.0, n_task_ph), "pelvis_height")
            self.tasks.AddTaskDim([0.85], "pelvis_height")
            self.tasks.AddTaskDim([0.03], "swing_margin")  # This is not being used.

        ### Set up environment
        if use_single_cost_function_for_all_tasks:
            self.FOM_model_dir_for_planner = ""

        # Check directory names
        self.EnforceSlashEnding(self.model_dir)
        self.EnforceSlashEnding(self.FOM_model_dir_for_sim)
        self.EnforceSlashEnding(self.FOM_model_dir_for_planner)
        self.EnforceSlashEnding(self.dir_rl)
        self.EnforceSlashEnding(self.dir_planner_data)
        self.EnforceSlashEnding(self.dir_history_data)

        # Create folder if not exist
        if len(sys.argv) > 1 and sys.argv[1] == "fresh":
            input("WARNING: Going to delete lcmlog files! (type anything to continue)")
            os.system("rm -rf " + self.dir_rl)
        Path(self.dir_rl).mkdir(parents=True, exist_ok=True)
        Path(self.dir_planner_data).mkdir(parents=True, exist_ok=True)  # for MPC's init file
        Path(self.dir_history_data).mkdir(parents=True, exist_ok=True)

        ### Create task list
        self.nominal_task_names = np.loadtxt(self.model_dir + "task_names.csv", dtype=str, delimiter=',')
        # Make sure the order is correct
        if not ((self.nominal_task_names[0] == "stride_length") &
              (self.nominal_task_names[1] == "ground_incline") &
              (self.nominal_task_names[2] == "duration") &
              (self.nominal_task_names[3] == "turning_rate") &
              (self.nominal_task_names[4] == "pelvis_height")):
            raise ValueError("ERROR: unexpected task name or task order")
        # Get duration from model optimization file
        path_1_0_task = self.model_dir + "1_0_task.csv"
        if os.path.exists(path_1_0_task):
            duration = np.loadtxt(path_1_0_task)[2]
        else:
            raise ValueError("%s doesn't exist" % path_1_0_task)
        self.tasks.AddTaskDim([duration], "duration", True)
        # Construct task object
        if ("ground_incline" in self.tasks.GetVaryingTaskElementName()) and ("turning_rate" in self.tasks.GetVaryingTaskElementName()):
            # we don't allow non-zero turning rate + non-zero ground incline
            self.tasks1 = copy.deepcopy(self.tasks)
            self.tasks1.AddTaskDim([0], "ground_incline", True)
            self.tasks2 = copy.deepcopy(self.tasks)
            self.tasks2.AddTaskDim([0], "turning_rate", True)
            self.tasks.Construct()
            self.tasks1.Construct()
            self.tasks2.Construct()
            self.task_sampler = TasksSampler([self.tasks1, self.tasks2], self.curriculum_learning, self.initial_task_box_range)
            #self.task_list = np.vstack([self.tasks1.GetTaskList(), self.tasks2.GetTaskList()])
        else:
            self.tasks.Construct()
            self.task_sampler = TasksSampler([self.tasks], self.curriculum_learning, self.initial_task_box_range)
            #self.task_list = self.tasks.GetTaskList()

        # Compute density for curriculum learning with randomly drawing tasks from the grid
        self.sample_density = -1
        if self.curriculum_learning and self.randomize_tasks:
            self.sample_density = self.max_n_rollouts_per_iteration / len(self.task_sampler.get_full_task_list()) 
            if self.sample_density > 1:
                print("self.max_n_rollouts_per_iteration = ", self.max_n_rollouts_per_iteration)
                print("len(self.task_sampler.get_full_task_list()) = ", len(self.task_sampler.get_full_task_list()))
                raise ValueError("You might want to increase the task grid resolution or decrease self.max_n_rollouts_per_iteration")
            if self.sample_density < self.min_task_sample_density_per_iteration:
                raise ValueError("You might want to decrease the task grid resolution or increase self.max_n_rollouts_per_iteration")
        self.previous_shuffled_task_idx_list = []

        # Make sure the dimension is correct
        if len(self.nominal_task_names) != self.tasks.get_task_dim():
            raise ValueError("sim eval task dimension is different from trajopt dim. "
                         "We want them to be the same becasue we use the same code "
                         "to plot sim cost and trajopt cost")

        ### Set up multithreading
        # self.n_max_thread = min(int(psutil.cpu_count() / 3) - 1, len(task_list))  # TODO: check if the min is necessary
        # self.n_max_thread = min(int(psutil.cpu_count() / 2), len(task_list)) if self.target_realtime_rate == 0.1 else self.n_max_thread
        self.n_max_thread = 3
        # if self.target_realtime_rate == 1:
        #     self.n_max_thread = 1
        self.supress_simulation_standard_output = True

        ### Some other checks
        # duration in sim doesn't have to be the same as trajopt's, but I added a check here as a reminder.
        if not math.isclose(
            parsed_yaml_file.get('left_support_duration') + parsed_yaml_file.get(
              'double_support_duration'), duration):
        # raise ValueError("Reminder: you are setting a different duration in sim than in trajopt")
            print("Warning: duration in sim is different from in trajopt")
            input("type anything to confirm and continue")

        if not parsed_yaml_file.get('use_hybrid_rom_mpc'):
            raise ValueError("`use_hybrid_rom_mpc` must be True")
        if self.t_no_logging_at_start + 4 > self.sim_end_time:
            raise ValueError("We want at least 4 seconds of logging time")
        if parsed_yaml_file.get('use_radio'):
            print("reminder: `use_radio` should be set to False automatically, since we set constant speed in the planner")
        if parsed_yaml_file.get('use_virtual_radio'):
            print("reminder: `use_virtual_radio` should be set to False automatically, since we set constant speed in the planner")
        if parsed_yaml_file.get('rom_option') != 30 and parsed_yaml_file.get('rom_option') != 32:
            raise ValueError("Currently we want to default to 30 or 32 (save time in gradient calculation); if the user want to use 27, then comment this assertion out")
        if not self.spring_model:
            if parsed_yaml_file.get('double_support_duration') != 0:
                raise ValueError("Warning; if the user insists, then comment this assertion out")
        if self.spring_model:
            if parsed_yaml_file.get('double_support_duration') == 0:
                raise ValueError("Warning; if the user insists, then comment this assertion out")
        if self.collect_rewards_via_lcmlogs:
            assert self.cmaes_or_ppo == "cmaes"

        ##################### In run sim method ####################
        ### Channel names
        self.ch = self.ChannelNames()

        ### Logging
        self.LogSimCostStudySetting()

        ### Build files just in case forgetting
        self.BuildFiles('examples/goldilocks_models/...')
        self.BuildFiles('examples/Cassie:multibody_sim_w_ground_incline')
        if self.run_visualizer:
            self.BuildFiles('examples/Cassie:visualizer')

        ### Construct sample indices from the task list for simulation
        # `trajopt_sample_idx_for_sim` is also used to initialize simulation state
        # `trajopt_sample_idx_for_planner` is for planner's initial guess and cost regularization term
        # TODO: Should I use only *one* sample_idx for the planner???
        #       Maybe I should because this is the most common way of how I use it on hardware experiment.
        self.trajopt_sample_indices_for_sim = self.ConstructTrajoptSampleIndicesGivenModelAndTask([1], self.task_sampler.get_full_task_list(), self.FOM_model_dir_for_sim)  # Use iter1 could be better since iter0 (wo ROM) could move COP to the edge
        self.trajopt_sample_indices_for_planner = self.ConstructTrajoptSampleIndicesGivenModelAndTask([0],
                                                                                                      self.task_sampler.get_full_task_list(),
                                                                                                      self.FOM_model_dir_for_planner,
                                                                                                      False)
        print("trajopt_sample_indices_for_sim = \n" + str(self.trajopt_sample_indices_for_sim))
        print("trajopt_sample_indices_for_planner = \n" + str(self.trajopt_sample_indices_for_planner))

        ############################################################

        # Assign policy
        if type(policy) is not MPCPolicy:
            raise TypeError("policy has to be MPCPolicy")
        self.policy = policy

        # Initialize model parameters from the initial model
        self.policy.cur_params = np.loadtxt(self.model_dir + "1_theta_yddot.csv")
        if len(self.policy.cur_params.shape) != 1:
            raise ValueError("should be a 1D vector")

        # Get name maps and state/action size from C++ code; Also unit testing at the same time
        self.only_construct_to_get_RL_problem_size_so_do_not_simulate = True
        self.unit_testing = True
        self.path_unit_testing_success = self.dir_planner_data + "unit_testing_passed"
        if os.path.exists(self.path_unit_testing_success):
            self.RunCommand("rm " + self.path_unit_testing_success)
        self.run_multiple_rollouts(0, 1, construct_planner_only=True)
        dir_planner_data = self.dir_planner_data + str(0) + "/"
        path_s_names = dir_planner_data + 'RL_state_names.csv'
        path_a_names = dir_planner_data + 'RL_action_names.csv'
        path_additional_info_names = dir_planner_data + 'RL_addtl_info_names.csv'
        assert os.path.exists(path_s_names)
        assert os.path.exists(path_a_names)
        assert os.path.exists(path_additional_info_names)
        self.s_names_map = self.ConstructNameMap(path_s_names)
        # self.a_names_map = self.ConstructNameMap(path_a_names)
        self.a_names = self.ReadNameVector(path_a_names)
        self.addtl_info_names_map = self.ConstructNameMap(path_additional_info_names)
        self.only_construct_to_get_RL_problem_size_so_do_not_simulate = False
        self.unit_testing = False
        assert os.path.exists(self.path_unit_testing_success)
        self.RunCommand("rm " + self.path_unit_testing_success)

        self.obs_size = len(self.s_names_map)
        self.action_size = len(self.a_names)

        # Initialize variance (this is default and gets overwritten in ppo_trainer.py)
        self.policy.cur_var = 0.00001 * np.ones(self.action_size)  # 0.0001  # Reminder: this is variance (sigma^2)!

        # Others
        self.policy.dir_dairlib = self.dir_dairlib
        if self.use_ipopt:
            self.n_max_thread = 1
            self.policy.n_max_thread = 1  # TODO: multithread ipopt so we can use multiple thread

        if self.cmaes_or_ppo == "ppo" and self.curriculum_learning:
            raise ValueError("curriculum_learning was implemented for CMA. TODO: need to review the code to make sure that it also works for ppo")
            # For example: need to call update_task_space_for_curriculum_learning in ppo_trainer.py

        # Set up self.w_torque and self.w_accel
        w_total = self.w_torque + self.w_accel
        self.w_torque = self.w_torque / w_total
        self.w_accel = self.w_accel / w_total

        # Log more configuration to wandb
        if self.sync_with_wandb:
            wandb.config.update({"t_no_logging_at_start": self.t_no_logging_at_start,
                                 "sim_end_time": self.sim_end_time,
                                 "min_mpc_thread_loop_duration": self.min_mpc_thread_loop_duration,
                                 "w_tracking": self.w_tracking,
                                 "w_torque": self.w_torque,
                                 "w_accel": self.w_accel,
                                 "exp_scaling_tracking_height": self.exp_scaling_tracking_height,
                                 "exp_scaling_tracking_vel": self.exp_scaling_tracking_vel,
                                 "exp_scaling_accel": self.exp_scaling_accel,
                                 "exp_scaling_torque": self.exp_scaling_torque,
                                 "use_acceleration_cost_instead_of_torque": self.use_acceleration_cost_instead_of_torque,
                                 "include_previous_vel_in_rl_state": self.include_previous_vel_in_rl_state,
                                 "spring_model": self.spring_model,
                                 "close_sim_gap": self.close_sim_gap,
                                 "use_ipopt": self.use_ipopt,
                                 "randomize_tasks": self.randomize_tasks,
                                 "curriculum_learning": self.curriculum_learning,
                                 "sample_density": self.sample_density,
                                 "max_n_rollouts_per_iteration": self.max_n_rollouts_per_iteration,
                                 "target_n_rollouts_per_iteration": self.target_n_rollouts_per_iteration})

    def filter_for_active_task_list(self):
        task_list = self.task_sampler.get_full_task_list()
        sampled2full_task_list_idx_map = list(range(len(task_list)))
        if self.curriculum_learning:
            task_list = self.task_sampler.get_active_task_list()
            sampled2full_task_list_idx_map = np.array(sampled2full_task_list_idx_map)[self.task_sampler.get_active_flags() == True]

        return task_list, sampled2full_task_list_idx_map

    def pick_tasks_for_evaluation_logging(self, rollout_without_policy_noise):
        task_list = self.task_sampler.get_full_task_list()
        sampled2full_task_list_idx_map = list(range(len(task_list)))
        if self.cmaes_or_ppo == "ppo" and rollout_without_policy_noise and self.randomize_tasks:
            # We select a fixed number of tasks (always the same) for quick performance evaluation
            n_rollout_for_quick_eval = 10
            task_list = task_list[::max(1, int(len(task_list) / n_rollout_for_quick_eval))]
            sampled2full_task_list_idx_map = sampled2full_task_list_idx_map[::max(1, int(len(task_list) / n_rollout_for_quick_eval))]
        return task_list, sampled2full_task_list_idx_map

    def determine_n_rollout(self, task_list, n_sample_needed, rollout_without_policy_noise, construct_planner_only):
        if construct_planner_only:
            return len(task_list)

        if self.cmaes_or_ppo == "ppo":
            if rollout_without_policy_noise:
                n_rollout = len(task_list)
            else:
                n_estimiated_rollout_needed = int(n_sample_needed / ((self.sim_end_time - self.t_no_logging_at_start) / self.min_mpc_thread_loop_duration)) + 1
                n_rollout = min(len(task_list), n_estimiated_rollout_needed)
        elif self.cmaes_or_ppo == "cmaes":
            if self.randomize_tasks:
                if self.curriculum_learning:
                    n_rollout = int(math.ceil(max(3, self.sample_density * self.task_sampler.get_active_task_list_length())))
                else:
                    n_rollout = min(len(task_list), self.target_n_rollouts_per_iteration)
            else:
                n_rollout = len(task_list)
        else:
            raise ValueError("self.cmaes_or_ppo should be either cmaes or ppo")

        # Sanity check
        if self.cmaes_or_ppo == "cmaes" and not rollout_without_policy_noise:
            raise ValueError("We don't want any policy noise with CMA method")

        return n_rollout

    def reset_for_new_iteration(self):
        ### Delete and create new data folder
        os.system("rm -rf " + self.dir_planner_data)  # This is a blocking method
        Path(self.dir_planner_data).mkdir(parents=True, exist_ok=True)

    def run_multiple_rollouts(self, iter_idx, n_sample_needed, construct_planner_only=False, rollout_without_policy_noise=False, use_previous_shuffled_task_idx_list=False):
        """
        Spawning all processes (planner, controller, simulation, lcm-logger, and maybe estimator) using the initialized MPC solution. Then collect all data
        """

        # Construct/select task_list and number of rollouts
        if self.cmaes_or_ppo == "ppo" and rollout_without_policy_noise:
            task_list, sampled_task_idx_to_full_task_idx_map = self.pick_tasks_for_evaluation_logging(rollout_without_policy_noise) 
        else:
            task_list, sampled_task_idx_to_full_task_idx_map = self.filter_for_active_task_list()             
        n_rollout = self.determine_n_rollout(task_list, n_sample_needed, rollout_without_policy_noise, construct_planner_only)

        # Save the policy (model) parameters
        self.path_model_params = self.dir_history_data + "%d_theta_yddot.csv" % iter_idx
        # np.save(self.dir_history_data + "%d_theta_yddot.csv" % iter_idx, self.policy.cur_params)  # cannot save as numpy binary file because C++ code doesn't read it
        np.savetxt(self.path_model_params, self.policy.cur_params, delimiter=",")  # This saves into a column vector by default  # TODO: maybe add a unit test for this for future proof
        if construct_planner_only or rollout_without_policy_noise:
            self.path_var = ""
        else:
            self.path_var = self.dir_history_data + "%d_policy_variance.csv" % iter_idx
            np.savetxt(self.path_var, self.policy.cur_var, delimiter=",")

        ### Reconstruct set index (needed for cleaner code; we pop without added back when construct_planner_only=True)
        self.thread_idx_set = set()
        for i in range(self.n_max_thread):
            self.thread_idx_set.add(i)

        ### Create new data folders
        latest_folder_idx = 0
        while True:
            if os.path.exists(self.dir_planner_data + str(latest_folder_idx) + "/"):
                latest_folder_idx += 1
            else:
                break
        folder_indices = list(range(latest_folder_idx, latest_folder_idx + len(task_list)))
        for folder_idx in folder_indices:
            # The folder should not exist already, so setting `exist_ok` to False
            Path(self.dir_planner_data + str(folder_idx) + "/").mkdir(parents=True, exist_ok=False)

        ### Shuffle index
        if use_previous_shuffled_task_idx_list and len(self.previous_shuffled_task_idx_list)>0:
            assert self.cmaes_or_ppo == "cmaes"
            assert len(self.previous_shuffled_task_idx_list) == n_rollout
            shuffled_task_idx_list = self.previous_shuffled_task_idx_list
        else:
            shuffled_task_idx_list = np.array(range(len(task_list)))
            np.random.shuffle(shuffled_task_idx_list)
            shuffled_task_idx_list = shuffled_task_idx_list[:n_rollout]
        self.previous_shuffled_task_idx_list = shuffled_task_idx_list

        ### Start simulation
        print("%d # of sim eval" % len(shuffled_task_idx_list))
        working_threads = []
        self.reevaluation_infos = []
        for task_idx in shuffled_task_idx_list:
            if construct_planner_only:
                task_idx = 0

            task = task_list[task_idx]
            full_task_idx = sampled_task_idx_to_full_task_idx_map[task_idx]
            trajopt_sample_idx_for_sim = self.trajopt_sample_indices_for_sim[0][full_task_idx]
            trajopt_sample_idx_for_planner = self.trajopt_sample_indices_for_planner[0][full_task_idx]

            folder_idx = folder_indices[task_idx]
            dir_planner_data = self.dir_planner_data + str(folder_idx) + "/"

            # print("\n===========\n")
            # print("progress %.1f%%" % (float(task_idx) / n_rollout * 100))
            # print("run sim for model %d and task %d" % (iter_idx, task_idx))

            # Get the initial traj
            # print("1 thread_idx_set = " + str(self.thread_idx_set))
            # print("len(working_threads) = " + str(len(working_threads)))
            thread_idx = self.thread_idx_set.pop()
            working_threads.append(
                self.RunSimAndController(thread_idx, task, folder_idx,
                                  iter_idx, dir_planner_data, trajopt_sample_idx_for_sim, trajopt_sample_idx_for_planner, True))
            # print("2 thread_idx_set = " + str(self.thread_idx_set))
            # print("len(working_threads) = " + str(len(working_threads)))
            # print("BlockAndDeleteTheLatestThread starts")
            # print("=======outside: working_threads = ", working_threads)
            self.BlockAndDeleteTheLatestThread(working_threads)
            # print("BlockAndDeleteTheLatestThread ends")
            # print("=======outside: working_threads = ", working_threads)
            # import pdb;pdb.set_trace()

            if construct_planner_only:
                return None

            # Run the simulation
            # print("3 thread_idx_set = " + str(self.thread_idx_set))
            # print("len(working_threads) = " + str(len(working_threads)))
            working_threads.append(
                self.RunSimAndController(thread_idx, task, folder_idx,
                                  iter_idx, dir_planner_data, trajopt_sample_idx_for_sim, trajopt_sample_idx_for_planner, False))
            # print("4 thread_idx_set = " + str(self.thread_idx_set))
            # print("len(working_threads) = " + str(len(working_threads)))
            # print("CheckSimThreadAndBlockWhenNecessary starts")
            self.CheckSimThreadAndBlockWhenNecessary(working_threads)
            # print("CheckSimThreadAndBlockWhenNecessary ends")
            # print("5 thread_idx_set = " + str(self.thread_idx_set))
            # print("len(working_threads) = " + str(len(working_threads)))
            # import pdb;pdb.set_trace()
        self.CheckSimThreadAndBlockWhenNecessary(working_threads, finish_up=True)
        
        # Sanity check
        if len(self.reevaluation_infos) != len(shuffled_task_idx_list):
            raise ValueError("the size of reevaluation_infos is incorrect")

        ### Read data and compute reward
        if self.collect_rewards_via_lcmlogs:
            # We collect the reward for the whole episode
            list_of_dict_per_episode = []
            for i in range(len(shuffled_task_idx_list)):
                task_idx = shuffled_task_idx_list[i]
                full_task_idx = sampled_task_idx_to_full_task_idx_map[task_idx]
                dir_planner_data = self.dir_planner_data + str(folder_indices[task_idx]) + "/"

                # Pause on implementing this, since it takes more work. We need to write it just like the RL reward below, so that we can encourage longer sim time, and expand task region
                assert False  # Finish the implementation here
                assert False  # also need to check if we need to turn back on publishing lcmt osc_debug and cassie_out

                # Call `EvalCostInMultithread`

                # list_of_dict_per_episode.append({"R_main": return_main, "R_tracking": return_main, "R_breakdown": {"R_vel_tracking": R_vel_tracking, "R_height_tracking": R_height_tracking},
                #                      "terminated": np.array([False]), "truncated": np.array([False]),
                #                      "extra_logging": {"abs_height_error": abs_height_error},
                #                      "extra_info": {"full_task_idx": full_task_idx}})
            return list_of_dict_per_episode

        else:
            # Note: We don't compute the graident here, but in the policy (this will happen in the policy update step; not the roll out step)
            list_of_dict_per_timestep = []
            for i in range(len(shuffled_task_idx_list)):
                task_idx = shuffled_task_idx_list[i]
                full_task_idx = sampled_task_idx_to_full_task_idx_map[task_idx]
                dir_planner_data = self.dir_planner_data + str(folder_indices[task_idx]) + "/"

                time_idx = 0
                this_rollout_has_valid_samples = False
                while True:
                    path_s = dir_planner_data + '%d_s.csv' % time_idx
                    path_a = dir_planner_data + '%d_a.csv' % time_idx
                    path_a_noise = dir_planner_data + '%d_a_noise.csv' % time_idx
                    path_sp = dir_planner_data + '%d_s_prime.csv' % time_idx
                    path_addtl_info = dir_planner_data + '%d_RL_addtl_info.csv' % time_idx
                    if os.path.exists(path_addtl_info):  # Picking the last-stored csv in planner thread in case planner was cut between csv files
                        addtl_info = np.loadtxt(path_addtl_info, ndmin=1)
                        if addtl_info[self.addtl_info_names_map['s_prime_time']] > self.t_no_logging_at_start:
                            s = np.loadtxt(path_s, ndmin=1)
                            a = np.loadtxt(path_a, ndmin=1)
                            a_noise = np.loadtxt(path_a_noise, ndmin=1)
                            sp = np.loadtxt(path_sp, ndmin=1)

                            # Compute reward
                            r_vel_tracking = self.ErrorMinimizationReward(addtl_info[self.addtl_info_names_map['com_vel_x']], s[self.s_names_map['des_com_vel_x']], self.exp_scaling_tracking_vel)
                            r_height_tracking = self.ErrorMinimizationReward(addtl_info[self.addtl_info_names_map['com_height']], s[self.s_names_map['des_com_height']], self.exp_scaling_tracking_height)
                            r_tracking = 0.5 * r_vel_tracking + 0.5 * r_height_tracking
                            #if self.use_acceleration_cost_instead_of_torque:
                            if self.include_previous_vel_in_rl_state:
                                vdot = (s[self.s_names_map['hip_roll_leftdot']:self.s_names_map['toe_rightdot']+1] - s[self.s_names_map['hip_roll_leftdot_prev']:self.s_names_map['toe_rightdot_prev']+1]) / self.min_mpc_thread_loop_duration
                            else:
                                vdot = (s[self.s_names_map['hip_roll_leftdot']:self.s_names_map['toe_rightdot']+1] - sp[self.s_names_map['hip_roll_leftdot']:self.s_names_map['toe_rightdot']+1]) / self.min_mpc_thread_loop_duration
                            r_accel = self.ErrorMinimizationReward(vdot, 0, self.exp_scaling_accel)
                            #else:
                            r_torque = self.ErrorMinimizationReward(
                                s[self.s_names_map['hip_roll_left_motor']:self.s_names_map['toe_right_motor'] + 1], 0,
                                self.exp_scaling_torque)
                            r_main_obj = self.w_accel * r_accel + self.w_torque * r_torque
                            r_total = self.w_tracking * r_tracking + r_main_obj

                            # Compute log_prob here to interface with the ppo_trainer.py code
                            log_prob = multivariate_normal.logpdf(a + a_noise, a, np.diag(self.policy.cur_var))

                            # For logging
                            abs_height_error = abs(addtl_info[self.addtl_info_names_map['com_height']] - s[self.s_names_map['des_com_height']])

                            # TODO: maybe pack a_noise into the dictionary for logging (instead of looking at the entropy)                        

                            list_of_dict_per_timestep.append({"s": s, "a": a, "sp": sp, "r": np.array([r_total]), "r_breakdown": {"r_main_obj": r_main_obj, "r_tracking": r_tracking, "r_accel": r_accel, "r_torque": r_torque, "r_vel_tracking": r_vel_tracking, "r_height_tracking": r_height_tracking},
                                                 "model_iter": iter_idx, "time_idx": time_idx, "reevaluation_info": self.reevaluation_infos[i], "log_prob": np.array([log_prob]),
                                                 "terminated": np.array([False]), "truncated": np.array([False]),
                                                 "extra_logging": {"abs_height_error": abs_height_error},
                                                 "extra_info": {"full_task_idx": full_task_idx}})
                            this_rollout_has_valid_samples = True
                        time_idx += 1
                    else:
                        if this_rollout_has_valid_samples:
                            path_addtl_info = dir_planner_data + '%d_RL_addtl_info.csv' % (time_idx-1)
                            addtl_info = np.loadtxt(path_addtl_info, ndmin=1)

                            end_due_to_bad_state = addtl_info[self.addtl_info_names_map['s_prime_time']] < self.sim_end_time - self.tol_determine_last_state
                            list_of_dict_per_timestep[-1]["terminated"] = np.array([end_due_to_bad_state])
                            list_of_dict_per_timestep[-1]["truncated"] = np.array([not end_due_to_bad_state])
                        break

            return list_of_dict_per_timestep

    # Exp(-scale_exponent * ||feedback_value - target_value||^2)
    def ErrorMinimizationReward(self, feedback_value, target_value, scale_exponent=1.0):
        abs_error = np.linalg.norm(feedback_value - target_value)**2
        return math.exp(-abs_error*scale_exponent)

    @staticmethod
    def ConstructNameMap(path_names):
        name_map = {}
        names = np.loadtxt(path_names, dtype=str, delimiter=',', ndmin=1)
        for i in range(names.size):
            assert not (names[i] in name_map)
            name_map[names[i]] = i
        return name_map

    @staticmethod
    def ReadNameVector(path_names):
        name_map = {}
        names = np.loadtxt(path_names, dtype=str, delimiter=',', ndmin=1)
        return names

    def n_sample_needed_to_cover_all_tasks(self):
        n_sample_needed_to_cover_all_tasks = int((self.sim_end_time - self.t_no_logging_at_start) / self.min_mpc_thread_loop_duration) * len(self.task_sampler.get_full_task_list())
        print(n_sample_needed_to_cover_all_tasks)
        return n_sample_needed_to_cover_all_tasks
    def n_sample_needed_to_have_enough_number_of_tasks(self):
        n_sample_needed_to_have_enough_number_of_tasks = int((self.sim_end_time - self.t_no_logging_at_start) / self.min_mpc_thread_loop_duration) * self.min_n_rollouts_per_iteration
        print(n_sample_needed_to_have_enough_number_of_tasks)
        return n_sample_needed_to_have_enough_number_of_tasks

    def n_tasks_per_value_evaluation(self):
        if self.cmaes_or_ppo != "cmaes":
            raise ValueError("this function is for cmaes")

        if not self.randomize_tasks:
            return len(self.task_sampler.get_full_task_list())
        else:
            if self.curriculum_learning:
                return self.max_n_rollouts_per_iteration
            else:
                return self.target_n_rollouts_per_iteration


    # trajopt_sample_indices for planner (find the most similar tasks)
    def ConstructTrajoptSampleIndicesGivenModelAndTask(self, model_indices, task_list, trajopt_sample_dir, zero_stride_length=False):
        trajopt_sample_indices = np.zeros((len(model_indices), len(task_list)),
                                          dtype=np.dtype(int))
        for i in range(len(model_indices)):
            for j in range(len(task_list)):
                trajopt_sample_indices[i, j] = self.GetTrajoptSampleIndexGivenTask(model_indices[i],
                                                                          task_list[j],
                                                                          trajopt_sample_dir,
                                                                          zero_stride_length)
        return trajopt_sample_indices


    # Collection of all channel names
    # Currently this class is useless becuase we use different lcm url for different simulation thread
    class ChannelNames:
        def __init__(self, idx = -1):
            self.channel_x = "CASSIE_STATE_SIMULATION"
            self.channel_fsm_t = "FSM_T"
            self.channel_y = "MPC_OUTPUT"
            self.channel_u = "ROM_WALKING"
            if idx >= 0:
                self.channel_x += str(idx)
                self.channel_fsm_t += str(idx)
                self.channel_y += str(idx)
                self.channel_u += str(idx)


    def BuildFiles(self, bazel_file_argument):
        build_cmd = ['bazel', 'build', bazel_file_argument, ]
        # build_cmd = ['cd', self.dir_dairlib, '&&'] + build_cmd
        build_process = subprocess.Popen(build_cmd, cwd=self.dir_dairlib)
        while build_process.poll() is None:  # while subprocess is alive
            time.sleep(0.1)


    # We assume cmd is string or a list of string
    # WARNING: when we use shell=True, p.kill() won't kill the process. Need to use the function `KillProcess` below.
    # stdout = None if we want to print to terminal, otherwise, stdout = DEVNULL
    def RunCommand(self, cmd, blocking_thread=True, supress_standard_output=False):
        if (type(cmd) != list) and (type(cmd) != str):
            raise ValueError("the command has to be either list or str")

        if type(cmd) == list:
            cmd = ' '.join(cmd)

        stdout = DEVNULL if supress_standard_output else None
        process = subprocess.Popen(cmd, shell=True, cwd=self.dir_dairlib, stdout=stdout)
        if blocking_thread:
            while process.poll() is None:  # while subprocess is alive
                time.sleep(0.002)
        else:
            return process


    def KillProcess(self, proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()

    def EnforceSlashEnding(self, dir):
        if len(dir) > 0 and dir[-1] != "/":
            raise ValueError("Directory path name should end with slash")

    def LogSimCostStudySetting(self, ):
        f = open(self.dir_rl + "rl_sim_training_log.txt", "a")
        f.write("\n\n*************************************************************\n")
        f.write("Current time : %s\n" % str(datetime.now()))
        f.write("model_dir = %s\n" % self.model_dir)
        f.write("spring_model = %s\n" % self.spring_model)
        f.write("target_realtime_rate = %s\n" % self.target_realtime_rate)
        f.write("init_sim_vel = %s\n" % self.init_sim_vel)
        f.write("set_sim_init_state_from_trajopt = %s\n" % self.set_sim_init_state_from_trajopt)
        f.write("completely_use_trajs_from_model_opt_as_target = false (because of using hybrid_rom_mpc)\n")

        f.write("Task info:\n")
        f.write(self.tasks.tasks_info())
        f.write("\n")

        if self.curriculum_learning:
            f.write("Initial task region for curriculum_learning:\n")
            f.write(str(self.initial_task_box_range))
            f.write("\n")

        commit_tag = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.dir_dairlib)
        git_diff = subprocess.check_output(['git', 'diff'], cwd=self.dir_dairlib)
        f.write("git commit hash: " + commit_tag.decode('ascii').strip() + "\n")
        f.write("\ngit diff:\n\n")
        f.write(codecs.getdecoder("unicode_escape")(git_diff)[0])

        f.close()


    def LcmlogFilePath(self, dir_planner_data, rom_iter_idx, folder_idx, extra_layer=""):
        return dir_planner_data + extra_layer + 'lcmlog-idx_%d_%d' % (rom_iter_idx, folder_idx)

    def InitPoseSuccessPath(self, dir_planner_data, rom_iter_idx, folder_idx):
        return dir_planner_data + 'sim_init_failed_%d_%d' % (rom_iter_idx, folder_idx)

    def InitPoseSolverFailed(self, path, enforce_existence = False):
        if os.path.exists(path):
            fail = int(np.loadtxt(path)) == 1
            return fail
        else:
            if enforce_existence:
                return True
            else:
                return False


    # Get trajopt sample idx with the most similar task
    # TODO: This function is currently not perfect yet. It cannot pick sample accurately if the sampling density is much high in one dimension than the other + we randomize tasks
    def GetTrajoptSampleIndexGivenTask(self, rom_iter, task, trajopt_sample_dir, zero_stride_length=False):
        dir = self.model_dir if len(trajopt_sample_dir) == 0 else trajopt_sample_dir

        stride_length_idx = list(self.nominal_task_names).index("stride_length")

        n_sample_trajopt = int(np.loadtxt(dir + "n_sample.csv"))
        dist_list = []
        path = ""
        # iter_range = list(range(251, 265)) if zero_stride_length else range(n_sample_trajopt)
        # for j in iter_range:
        for j in range(n_sample_trajopt):
            path = dir + "%d_%d_task.csv" % (rom_iter, j)
            # print("try " + path)
            if os.path.exists(path):
                trajopt_task = np.loadtxt(path)
                if zero_stride_length:
                    trajopt_task[stride_length_idx] = 0.0
                dist_list.append(np.linalg.norm(trajopt_task - task))
        # print("dist_list = ")
        # print(dist_list)
        if len(dist_list) == 0:
            raise ValueError("ERROR: This path doesn't exist: " + path)
        # trajopt_sample_idx = iter_range[int(np.argmin(np.array(dist_list)))]
        trajopt_sample_idx = np.argmin(np.array(dist_list))

        return trajopt_sample_idx


    def EndSim(self, working_threads, idx, recycle_idx=True):
        # Once we reach the code here, it means one simulation has ended

        # Log sim status
        if self.InitPoseSolverFailed(working_threads[idx][1][0], True):
            print(working_threads[idx][1][1])
            f = open(self.dir_rl + "sim_status.txt", "a")
            f.write(working_threads[idx][1][1])
            f.close()

        # Kill the rest of processes (necessary)
        for i in range(0, len(working_threads[idx][0])):
            if working_threads[idx][0][i].poll() is None:  # if the process is still running
                # I added try-catch in case the process ended between p.poll() and os.killpg()
                try:
                    self.KillProcess(working_threads[idx][0][i].pid)
                except Exception:
                    print("attempted to kill a non-existing process")

        # Add back available thread idx.
        # We don't add back avaible thread idx when the sim is used to initialize the
        # MPC first solution, because we need the same thread index (used in folder
        # name) to store the files
        if recycle_idx:
            self.thread_idx_set.add(working_threads[idx][3])

        del working_threads[idx]


    def BlockAndDeleteTheLatestThread(self, working_threads):
        # print("=======inside: working_threads = ", working_threads)
        # It is always the last one in the list because we just appended it.
        while (working_threads[-1][0][0].poll() is None) and (time.time() - working_threads[-1][4] < self.max_time_for_a_thread):
            # print("working_threads[-1][0][0].poll() is None: ", working_threads[-1][0][0].poll() is None)
            time.sleep(0.1)
        # print("working_threads[-1][0][0].poll() is None: ", working_threads[-1][0][0].poll() is None)
        self.EndSim(working_threads, -1, False)


    def CheckSimThreadAndBlockWhenNecessary(self, working_threads, finish_up=False):
        # Wait for threads to finish once is more than self.n_max_thread
        while (not finish_up and (len(working_threads) >= self.n_max_thread)) or \
            (finish_up and (len(working_threads) > 0)):
            for j in range(len(working_threads)):
                # if it's still running and hasn't spent too much time
                if (working_threads[j][0][0].poll() is None) and (time.time() - working_threads[j][4] < self.max_time_for_a_thread):
                    # print("j = ", j)
                    # print("working_threads[j][0][0].poll() is None: ", working_threads[j][0][0].poll() is None)
                    time.sleep(0.1)
                else:
                    # I paused a second in sim script to give lcm-logger time to finish
                    # logging, so I don't need to pause here again. That is, no need to run
                    # `time.sleep(1)` here.
                    # print("j = ", j)
                    # print("working_threads[j][0][0].poll() is None: ", working_threads[j][0][0].poll() is None)
                    self.EndSim(working_threads, j)
                    break
        # print("(len(working_threads) >= self.n_max_thread): ", (len(working_threads) >= self.n_max_thread))

    def RunSimAndController(self, thread_idx, task, folder_idx, rom_iter_idx, dir_planner_data,
        trajopt_sample_idx_for_sim, trajopt_sample_idx_for_planner, get_init_file):

        """
        # Set `get_init_file` to True if you want to generate the initial traj for both
        # planner and controller
        # `trajopt_sample_idx` is used for planner's initial guess and cost regularization term
        """

        # simulation arguments
        init_x_vel_reduction_ratio = 0.66  # since the starting configuration is standing pose, setting the task velocity would throw the MPC off. Cassie couldn't catch itself.
        pause_second = 2.0 if get_init_file else 0
        path_init_pose_success = self.InitPoseSuccessPath(dir_planner_data, rom_iter_idx, folder_idx)

        # planner arguments
        planner_init_file = '' if get_init_file else '0_z.csv'

        # controller arguments
        init_traj_file_path = '' if get_init_file else dir_planner_data + '0_rom_trajectory'
        # Untested hack -- in the case when "0_rom_trajectory" doesn't exist for some reason
        if (not get_init_file) and (not os.path.exists(dir_planner_data + '0_rom_trajectory')):
            init_traj_file_path = ''
            pause_second = 2.0

        # other arguments
        port_idx = thread_idx + 1024  # Ports below 1024 are considered to be privileged in Linux. https://stackoverflow.com/questions/31899673/bind-returning-permission-denied-c
        planner_wait_identifier = dir_planner_data + "planner" + str(time.time())
        control_wait_identifier = dir_planner_data + "controller" + str(time.time())
        # print("planner_wait_identifier = ", planner_wait_identifier)
        # print("control_wait_identifier = ", control_wait_identifier)

        # Extract tasks
        task_sl = task[self.tasks.GetDimIdxByName("stride_length")]
        task_ph = task[self.tasks.GetDimIdxByName("pelvis_height")]
        task_gi = task[self.tasks.GetDimIdxByName("ground_incline")]
        task_tr = task[self.tasks.GetDimIdxByName("turning_rate")]
        task_du = task[self.tasks.GetDimIdxByName("duration")]

        # Some file paths
        dir_and_prefix_FOM_reg = "" if len(self.FOM_model_dir_for_planner) == 0 else "%s0_%d_" % (self.FOM_model_dir_for_planner, trajopt_sample_idx_for_planner)
        set_sim_init_state_from_trajopt = self.set_sim_init_state_from_trajopt
        # Currently we don't have trajopt samples for initializing simulation with non-zero ground incline
        if task_gi != 0:
            set_sim_init_state_from_trajopt = False
        path_simulation_init_state = "%s%d_%d_x_samples0.csv" % (self.FOM_model_dir_for_sim, 0, trajopt_sample_idx_for_sim) if set_sim_init_state_from_trajopt else ""

        # Store info for the re-evaluation of the MPC solves
        if not get_init_file:
            self.reevaluation_infos.append({"spring_model": self.spring_model,
                                            "close_sim_gap": self.close_sim_gap,
                                            "use_ipopt": self.use_ipopt,
                                            "knots_per_mode": self.knots_per_mode,
                                            "feas_tol": self.feas_tol,
                                            "opt_tol": self.opt_tol,
                                            "n_step": self.n_step,
                                            "task_sl": task_sl,
                                            "task_ph": task_ph,
                                            "task_gi": task_gi,
                                            "task_tr": task_tr,
                                            "task_du": task_du,
                                            "dir_and_prefix_FOM_reg": dir_and_prefix_FOM_reg,
                                            "dir_dairlib": self.dir_dairlib,
                                            "trajopt_sample_idx_for_planner": trajopt_sample_idx_for_planner,
                                            "dir_planner_data": dir_planner_data,
                                            "path_model_params": self.path_model_params,
                                            "path_var": self.path_var,
                                            "min_mpc_thread_loop_duration": self.min_mpc_thread_loop_duration})

        planner_cmd = [
            self.dir_dairlib + 'bazel-bin/examples/goldilocks_models/run_cassie_rom_planner_process',
            '--channel_x=%s' % self.ch.channel_x,
            '--channel_fsm_t=%s' % self.ch.channel_fsm_t,
            '--channel_y=%s' % self.ch.channel_y,
            '--lcm_url_port=%d' % port_idx,
            '--fix_duration=true',
            '--zero_touchdown_impact=true',
            '--log_solver_info=false',
            '--iter=1',  # We only use this for initial guess and regularization term if is_RL_training=True
            '--sample=%d' % trajopt_sample_idx_for_planner,
            '--knots_per_mode=%d' % self.knots_per_mode,
            '--n_step=%d' % self.n_step,
            '--feas_tol=%.6f' % self.feas_tol,
            '--opt_tol=%.6f' % self.opt_tol,
            '--stride_length=%.3f' % task_sl,
            '--pelvis_height=%.3f' % task_ph,
            '--time_limit=0',  # we always dynamically adjust the time limit
            '--realtime_rate_for_time_limit=%.3f' % self.realtime_rate_for_time_limit,
            '--init_file=%s' % planner_init_file,
            '--use_ipopt=%s' % ("true" if self.use_ipopt else str(get_init_file).lower()),
            '--run_one_loop_to_get_init_file=%s' % str(get_init_file).lower(),
            '--switch_to_snopt_after_first_loop=false',
            '--spring_model=%s' % str(self.spring_model).lower(),
            '--dir_and_prefix_FOM=%s' % dir_and_prefix_FOM_reg,
            '--dir_data=%s' % dir_planner_data,
            '--path_wait_identifier=%s' % planner_wait_identifier,
            '--print_level=0',
            '--completely_use_trajs_from_model_opt_as_target=false',   # completely_use_trajs_from_model_opt_as_target=false because of hybrid_rom_mpc
            '--close_sim_gap=%s' % str(self.close_sim_gap).lower(),
            '--log_data=%s' % str(get_init_file or not self.collect_rewards_via_lcmlogs).lower(),  # savig reward files requires this to be True
            '--is_RL_training=true',
            '--collect_rewards_via_lcmlogs=%s' % str(self.collect_rewards_via_lcmlogs).lower(),
            '--include_previous_vel_in_rl_state=%s' % str(self.include_previous_vel_in_rl_state).lower(),
            '--get_RL_gradient_offline=false',
            '--debug_mode=false',
            '--path_model_params=%s' % self.path_model_params,
            '--path_var=%s' % self.path_var,
            '--min_mpc_thread_loop_duration=%.3f' % self.min_mpc_thread_loop_duration,
            '--policy_output_noise_bound=%.6f' % self.policy_output_noise_bound,
            '--only_construct_to_get_RL_problem_size_so_do_not_simulate=%s' % str(self.only_construct_to_get_RL_problem_size_so_do_not_simulate).lower(),
            '--unit_testing=%s' % str(self.unit_testing).lower(),
            '--path_unit_testing_success=%s' % str(self.path_unit_testing_success).lower(),
            ]
        controller_cmd = [
            self.dir_dairlib + 'bazel-bin/examples/goldilocks_models/run_cassie_rom_controller',
            '--channel_x=%s' % self.ch.channel_x,
            '--channel_fsm_t=%s' % self.ch.channel_fsm_t,
            '--channel_y=%s' % self.ch.channel_y,
            '--channel_u=%s' % self.ch.channel_u,
            '--lcm_url_port=%d' % port_idx,
            '--turning_rate=%.3f' % task_tr,
            '--ground_incline=%.3f' % task_gi,
            '--close_sim_gap=%s' % str(self.close_sim_gap).lower(),
            '--init_traj_file_path=%s' % init_traj_file_path,
            '--spring_model=%s' % str(self.spring_model).lower(),
            '--get_swing_foot_from_planner=true',
            '--get_stance_hip_angles_from_planner=false',
            '--get_swing_hip_angle_from_planner=false',
            '--path_wait_identifier=%s' % control_wait_identifier,
            '--is_RL_training=true',
            '--path_model_params=%s' % self.path_model_params,
            '--publish_osc_data=false',  # avoid uncessary publishing, since we are not running lcm-logger
            ]
        simulator_cmd = [
            self.dir_dairlib + 'bazel-bin/examples/Cassie/multibody_sim_w_ground_incline',
            '--channel_x=%s' % self.ch.channel_x,
            '--channel_u=%s' % self.ch.channel_u,
            '--lcm_url_port=%d' % port_idx,
            '--end_time=%.3f' % self.sim_end_time,
            '--pause_second=%.3f' % pause_second,
            '--target_realtime_rate=%.3f' % self.target_realtime_rate,
            '--spring_model=%s' % str(self.spring_model).lower(),
            '--init_height=%.3f' % task_ph,
            '--pelvis_x_vel=%.3f' % ((init_x_vel_reduction_ratio * task_sl / task_du) if self.init_sim_vel else 0),
            '--pelvis_y_vel=%.3f' % 0.15,
            '--turning_rate=%.3f' % task_tr,
            '--ground_incline=%.3f' % task_gi,
            '--path_init_state=%s' % path_simulation_init_state,
            '--path_init_pose_success=%s' % path_init_pose_success,
            '--is_RL_training=true',
            '--publish_cassie_out=false',  # avoid uncessary publishing, since we are not using radio and state estimation
            ]
        visualizer_cmd = [
            self.dir_dairlib + 'bazel-bin/examples/Cassie/visualizer',
            '--com=False',
            '--channel=CASSIE_STATE_SIMULATION',
            '--lcm_url_port=%d' % port_idx,
            '--ground_incline=%.3f' % task_gi,
            ]
        lcm_logger_cmd = [
            'sleep %.1f && ' % (self.t_no_logging_at_start/self.target_realtime_rate),
            'lcm-logger',
            '--lcm-url=udpm://239.255.76.67:%d' % port_idx,
            '-f',
            self.LcmlogFilePath(dir_planner_data, rom_iter_idx, folder_idx),
            ]

        # Testing code to get command
        # if not get_init_file:
        #   print(' '.join(planner_cmd))
        #   print(' '.join(controller_cmd))
        #   print(' '.join(simulator_cmd))
        #   print(' '.join(lcm_logger_cmd))
        #   input("type anything to continue")
        # else:
        #   return

        path = self.dir_history_data + "%d_%d_commands.txt" % (rom_iter_idx, folder_idx)
        f = open(path, "a")
        f.write(' '.join(planner_cmd) + "\n")
        f.write(' '.join(controller_cmd) + "\n")
        f.write(' '.join(simulator_cmd) + "\n")
        f.write(' '.join(visualizer_cmd) + "\n")
        f.write(' '.join(lcm_logger_cmd) + "\n")
        f.write("---\n")
        f.close()

        # Remove file for init pose success/fail flag
        if os.path.exists(path_init_pose_success):
            self.RunCommand("rm " + path_init_pose_success)

        # Run all processes
        planner_process = self.RunCommand(planner_cmd, False, supress_standard_output=self.supress_simulation_standard_output)
        controller_process = self.RunCommand(controller_cmd, False, supress_standard_output=self.supress_simulation_standard_output)
        if not get_init_file:
            if self.collect_rewards_via_lcmlogs:
                logger_process = self.RunCommand(lcm_logger_cmd, False)
            if self.run_visualizer:
                visualizer_process = self.RunCommand(visualizer_cmd, False, supress_standard_output=self.supress_simulation_standard_output)
        # We don't run simulation thread until both planner and controller are waiting for simulation's message
        while not os.path.exists(planner_wait_identifier) or \
                not os.path.exists(control_wait_identifier):
            time.sleep(0.1)
        self.RunCommand("rm " + planner_wait_identifier)
        self.RunCommand("rm " + control_wait_identifier)
        simulator_process = self.RunCommand(simulator_cmd, False, supress_standard_output=self.supress_simulation_standard_output)

        # Message to return
        msg = "iteration #%d folder/log #%d: init pose solver failed to find a pose\n" % (rom_iter_idx, folder_idx)

        processes_list = []  # once the first process ends, we terminate the rest
        if get_init_file:
            processes_list.append(planner_process)
            processes_list.append(simulator_process)
        else:
            processes_list.append(simulator_process)
            processes_list.append(planner_process)
        processes_list.append(controller_process)
        if not get_init_file:
            if self.collect_rewards_via_lcmlogs:
                processes_list.append(logger_process)
            if self.run_visualizer:
                processes_list.append(visualizer_process)

        return (processes_list,
                [path_init_pose_success, msg],
                get_init_file,
                thread_idx,
                time.time())



# Grid tasks
class Tasks:
    # Constructor and builders
    def __init__(self):
        self.constructed = False
        self.names = []  # ordered names
        self.task_input_dict = {}
        self.grid2list_idx_map = {}
        self.list2grid_idx_map = {}

    def AddTaskDim(self, array, name, overwrite_existing=False):
        if self.constructed:
            raise ValueError("Cannot call this function after building the task obj")
        if not isinstance(array, (list, np.ndarray)):
            raise TypeError("array should be a list or numpy array")
        if len(np.array(array).shape) != 1:
            raise TypeError("array should be 1D")
        if not isinstance(name, str):
            raise TypeError("name should be a string")
        if not overwrite_existing:
            if name in self.task_input_dict:
                raise ValueError("%s is already a key in task_data" % name)

        if name not in self.names:
            self.names.append(name)
        self.task_input_dict[name] = np.array(array)

    def CreateTasklistViaDfs(self, level, grid_index):
        if level == self.n_dim:
            task = []
            # print("grid_index = " + str(grid_index))
            for i_dim in range(self.n_dim):
                task_idx = grid_index[i_dim]
                task.append(self.task_input_dict[self.names[i_dim]][task_idx])
            self.grid2list_idx_map[tuple(grid_index)] = len(self.task_list)
            self.list2grid_idx_map[len(self.task_list)] = tuple(grid_index)
            self.task_list.append(task)
        else:
            for _ in self.task_input_dict[self.names[level]]:
                # print("before " + str(grid_index))
                self.CreateTasklistViaDfs(level + 1,  copy.deepcopy(grid_index))
                grid_index[level] += 1
                # print("after " + str(grid_index))

    def Construct(self):
        self.constructed = True

        self.n_dim = len(self.task_input_dict)

        # Compute n_task
        self.n_task = 1
        for key in self.task_input_dict:
            self.n_task *= len(self.task_input_dict[key])

        # Create task list
        # self.task_list = np.zeros((self.n_task, self.n_dim))
        self.task_list = []
        level = 0
        grid_index = [0] * self.n_dim
        self.CreateTasklistViaDfs(level, copy.deepcopy(grid_index))
        self.task_list = np.array(self.task_list)

        print("task_list = \n" + str(self.task_list))

        if self.task_list.shape != (self.n_task, self.n_dim):
            raise ValueError("self.task_list.shape = " + str(self.task_list.shape) + ", but we expect (" + str(self.n_task) + ", " + str(self.n_dim) + ")")

    # Getters
    def tasks_info(self):
        output = ""
        for name in self.names:
            output += "%s ranges from %.3f to %.3f\n" % (name, self.task_input_dict[name][0], self.task_input_dict[name][-1])
        return output
    def get_task_dim(self):
        return self.n_dim
    def get_n_task(self):
        return self.n_task
    def GetDimIdxByName(self, name):
        if not (name in self.names):
            raise ValueError("%s doesn't exist in the tasks" % name)
        return self.names.index(name)
    def GetNameByDimIdx(self, task_dim_idx):
        assert (task_dim_idx >= 0) and (task_dim_idx < len(self.names))
        return self.names[task_dim_idx]
    def GetSizeByDimIdx(self, task_dim_idx):
        return len(self.task_input_dict[self.GetNameByDimIdx(task_dim_idx)])
    def GetTask(self, task_idx):
        return self.task_list[task_idx]
    def GetTaskList(self):
        return self.task_list
    def GetVaryingTaskElementName(self):
        name_list = []
        for key in self.task_input_dict:
            if len(self.task_input_dict[key]) > 1:
                name_list.append(key)
        return name_list
    def GetAllTaskNames(self):
        return self.task_input_dict.keys()
    
    def GetAdjacentTaskIndicesGivenATaskIndex(self, idx):
        assert idx >= 0 

        grid_idx_tuple = self.list2grid_idx_map[idx]

        adjacent_task_list_indices = []
        for i_dim in range(len(grid_idx_tuple)):
            idx_tuple_mutable = list(grid_idx_tuple)
            idx_tuple_mutable[i_dim] -= 1
            if idx_tuple_mutable[i_dim] >= 0:
                adjacent_task_list_indices.append(self.grid2list_idx_map[tuple(idx_tuple_mutable)])
            idx_tuple_mutable[i_dim] += 2
            if idx_tuple_mutable[i_dim] < self.GetSizeByDimIdx(i_dim):
                adjacent_task_list_indices.append(self.grid2list_idx_map[tuple(idx_tuple_mutable)])
            idx_tuple_mutable[i_dim] -= 1
        return adjacent_task_list_indices


class TasksSampler:
    def __init__(self, tasks_object_list, curriculum_learning=False, initial_task_box_range={}):
        ### Some checks ###
        # This classs assumes all task objects have been constructed (the task lists have been created)
        for task_object in tasks_object_list:
            assert task_object.constructed

        assert len(tasks_object_list) > 0

        for i in range(len(tasks_object_list) - 1):
            # I use list in the line below, in order to check the order of the keys too
            assert list(tasks_object_list[i].GetAllTaskNames()) == list(tasks_object_list[i+1].GetAllTaskNames()) 

        if curriculum_learning:
            assert len(initial_task_box_range) > 0
            for key in initial_task_box_range:
                assert key in tasks_object_list[0].GetAllTaskNames()
                assert initial_task_box_range[key][0] <= initial_task_box_range[key][1]

        ### Set up the object ###
        self.tasks_object_list = tasks_object_list

        # Construct `self.full_task_list`
        self.full_task_list = np.array(self.tasks_object_list[0].GetTaskList())
        for i in range(1, len(self.tasks_object_list)):
            self.full_task_list = np.vstack([self.full_task_list, self.tasks_object_list[i].GetTaskList()])

        # Set initial active flags
        self.task_active_flag = np.ones((len(self.full_task_list)), dtype=bool)  # Initialize all to True
        if curriculum_learning:
            for key in initial_task_box_range:
                task_dim_idx = self.tasks_object_list[0].GetDimIdxByName(key)
                self.task_active_flag[self.full_task_list[:,task_dim_idx] < initial_task_box_range[key][0]] = False
                self.task_active_flag[self.full_task_list[:,task_dim_idx] > initial_task_box_range[key][1]] = False

        assert len(self.full_task_list) == len(self.task_active_flag) 
        assert type(self.full_task_list) == np.ndarray
        assert type(self.task_active_flag) == np.ndarray

        task_list_idx_start = 0
        for task_object in self.tasks_object_list:
            # If you see assertion fail, it's possible that your slices didn't slice 0 value by accident (e.g. 0 rad ground incline)
            assert np.sum(self.task_active_flag[task_list_idx_start:task_list_idx_start+task_object.get_n_task()] == True) > 0  # we need at least one active task for each plane
            task_list_idx_start += task_object.get_n_task()

    def get_full_task_list(self):
        return self.full_task_list

    def get_active_task_list(self):
        return self.full_task_list[self.task_active_flag == True]        

    def get_active_flags(self):
        return self.task_active_flag

    def get_active_task_list_length(self):
        return np.sum(self.task_active_flag == True)

    def full_list_idx_to_Tasks_object_map(self, idx):
        assert idx >= 0
        task_list_idx_start = 0
        for task_object in self.tasks_object_list:
            if idx < task_list_idx_start + task_object.get_n_task():
                return task_object, task_list_idx_start
            else:
                task_list_idx_start += task_object.get_n_task()
        raise ValueError("task list index %d is too big. Total length is %d" % (idx, task_list_idx_start))

    # This method has too take in `full_list_idx` and not `active_list_idx`, because the active list set changes every time we call this method, but we run simulations in a batch
    def update_successful_flags(self, full_list_idx):
        task_object, task_list_idx_start = self.full_list_idx_to_Tasks_object_map(full_list_idx)
        local_task_list_idx = full_list_idx - task_list_idx_start
        local_adjacent_task_list_indices = task_object.GetAdjacentTaskIndicesGivenATaskIndex(local_task_list_idx)
        local_adjacent_task_list_indices = [idx + task_list_idx_start for idx in local_adjacent_task_list_indices]
        self.task_active_flag[local_adjacent_task_list_indices] = True


    def print_task_active_flags(self, dir_rl):
        self.print_full_task_list_flags(self.task_active_flag, dir_rl)

    def print_full_task_list_flags(self, full_task_list_flags, dir_rl="", write_to_file=True):
        assert len(full_task_list_flags) == len(self.full_task_list)
        if write_to_file:
            assert len(dir_rl) > 0
            f = open(dir_rl + "rl_sim_training_log.txt", "a")
            f.write("\n\n*************************************************************\n")
        task_list_idx_start = 0
        for task_object in self.tasks_object_list:
            names = [task_object.GetNameByDimIdx(i) for i in range(task_object.get_task_dim()) if task_object.GetSizeByDimIdx(i) != 1]
            flags = copy.deepcopy(full_task_list_flags[task_list_idx_start:task_list_idx_start+task_object.get_n_task()]).reshape([task_object.GetSizeByDimIdx(i) for i in range(task_object.get_task_dim()) if task_object.GetSizeByDimIdx(i) != 1])

            # Downsample if the resolution of the task grid is too high
            max_n_samples_per_dim_to_show = 15
            downsampled = False
            for i_dim in range(len(flags.shape)):
                if flags.shape[i_dim] > max_n_samples_per_dim_to_show:
                    downsampled = True
                    flags = np.take(flags, [int(round(faction)) for faction in np.linspace(0,flags.shape[i_dim]-1,max_n_samples_per_dim_to_show)], axis=i_dim)

            print(names, "(down-sampled)" if downsampled else "", "\n")
            print(flags.astype(int),"\n")  # convert to int for cleaner presentation
            if write_to_file:
                f.write(str(names) + ("(down-sampled)" if downsampled else "") + ":\n")
                f.write(str(flags.astype(int)) + ":\n")
            task_list_idx_start += task_object.get_n_task()
        if write_to_file:
            f.close()
        


def test_task_sampler():
    assert False  # implement a quick unit test for the sampler for curriculum learning 


def main():
    # Testing the code
    # TODO: this will become our unit test
    import torch
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = MPCPolicy(device=_device)

    data_collector = CassieRolloutDataCollector(policy, "testing_project", cmaes_or_ppo="ppo")
    list_of_dict = data_collector.run_multiple_rollouts(iter_idx=1, n_sample_needed=2)
    assert len(list_of_dict) > 0
    import pdb; pdb.set_trace()



if __name__ == "__main__":
    # main()
    test_task_sampler()



