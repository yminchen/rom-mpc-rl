import numpy as np
import time
import torch
from scipy.stats import multivariate_normal

from typing import Optional, Dict, List

import subprocess
from subprocess import DEVNULL
import psutil

class MPCPolicy:
    def __init__(
        self,
        device: str,
        init_params: Optional[np.array] = None,
        init_var: Optional[np.array] = None,
    ) -> None:
        self.device = device
        
        # these MPC parameters are adjustable
        self.cur_params = None if init_params is None else init_params.copy()
        self.cur_var = None if init_var is None else init_var.copy()

        # others
        self.dir_dairlib = ""  # this will be assigned later
        self.n_max_thread = psutil.cpu_count()

    # dict_data contains init_w, etc
    # dict_data should be a list of dictionary, because x here is mini-batch (list of x's)
    def get_action(self, x, action=None, dict_data=None):
        if dict_data is None:
            raise ValueError("must provide dict_data in MPCPolicy")
        if type(dict_data) == Dict:
            raise ValueError("wrong type -- should be a list of dictionary instead of a dict")
        assert len(x) == len(dict_data)

        max_var = 10.0 * np.ones_like(self.cur_var)
        min_var = 1e-10 * np.ones_like(self.cur_var)
        self.cur_var = np.clip(self.cur_var, a_min=min_var, a_max=max_var)  # ensure positive variance (one reason: multivariate_normal.logpdf())
        if action is not None:
            action = action.cpu().numpy()

        if type(x) is torch.Tensor:
            x = x.cpu().numpy()
        batch_size = x.shape[0]

        assert len(self.dir_dairlib) > 0
        assert self.n_max_thread > 0

        # Compute MPC outputs
        indices = list(range(batch_size))
        chunks = [indices[i:i + self.n_max_thread] for i in range(0, len(indices), self.n_max_thread)]
        mpc_outputs = []
        for chunk in chunks:
            # 1. solve MPC
            processes = [self.solve_mpc(x[idx], dict_data[idx]) for idx in chunk]
            # wait for all processes to finish
            while any([process.poll() is None for process in processes]):
                time.sleep(0.001)
            # 2. read the solution file
            for idx in chunk:
                mpc_outputs.append(self.read_mpc_solution_files(x[idx], dict_data[idx]))

        # Compute the rest of info
        mean_action = []
        sample_action = []
        log_prob = []
        entropy = []
        for i in range(batch_size):
            mean_action.append(mpc_outputs[i]["a"])

            if action is None:
                sample_action.append(mean_action[i] + mpc_outputs[i]["a_noise"])
            else:
                sample_action.append(action[i])

            # If the action is provided, then just compute the log prob and entropy
            # Note: In the case of action != None (i.e. policy update step), we get new probability density with new mean (i.e. mean_action[i]), and we evaluate the prob_den at old action. This is what Eq 3 is doing.
            log_prob.append(
                multivariate_normal.logpdf(sample_action[i], mean_action[i], np.diag(self.cur_var))
            )
            entropy.append(multivariate_normal.entropy(mean_action[i], np.diag(self.cur_var)))

        sample_action = torch.Tensor(np.array(sample_action)).to(self.device)
        log_prob = torch.Tensor(np.array(log_prob)).to(self.device)
        entropy = torch.Tensor(np.array(entropy)).to(self.device)
        mean_action = torch.Tensor(np.array(mean_action)).to(self.device)

        assert sample_action.shape[0] == batch_size
        assert log_prob.shape[0] == batch_size
        assert entropy.shape[0] == batch_size
        assert mean_action.shape[0] == batch_size

        return mpc_outputs, sample_action, log_prob, entropy, mean_action

    # `solve_mpc` is only called during the policy update steps and not the rollout
    # `dict_data` needs to contain the info for parameter file, and also the initial guess file
    def solve_mpc(self, x0: np.array, dict_data: Optional[Dict] = None):
        reevaluation_info = dict_data["reevaluation_info"]

        # parameters
        #self.dir_dairlib = reevaluation_info["dir_dairlib"]
        spring_model = reevaluation_info["spring_model"]
        close_sim_gap = reevaluation_info["close_sim_gap"]

        # planner arguments
        use_ipopt = reevaluation_info["use_ipopt"]
        knots_per_mode = reevaluation_info["knots_per_mode"]
        feas_tol = reevaluation_info["feas_tol"]
        opt_tol = reevaluation_info["opt_tol"]
        n_step = reevaluation_info["n_step"]

        # Extract tasks
        task_sl = reevaluation_info["task_sl"]
        task_ph = reevaluation_info["task_ph"]
        task_gi = reevaluation_info["task_gi"]
        task_tr = reevaluation_info["task_tr"]
        task_du = reevaluation_info["task_du"]

        # Regularization
        dir_and_prefix_FOM_reg = reevaluation_info["dir_and_prefix_FOM_reg"]
        trajopt_sample_idx_for_planner = reevaluation_info["trajopt_sample_idx_for_planner"]

        # For RL
        path_model_params = reevaluation_info["path_model_params"]
        path_var = reevaluation_info["path_var"]
        min_mpc_thread_loop_duration = reevaluation_info["min_mpc_thread_loop_duration"]

        dir_planner_data = reevaluation_info["dir_planner_data"]
        solve_idx_for_read_from_file = dict_data["time_idx"]

        planner_cmd = [
            'bazel-bin/examples/goldilocks_models/run_cassie_rom_planner_process',
            '--fix_duration=true',
            '--zero_touchdown_impact=true',
            '--log_solver_info=false',
            '--iter=1',  # We only use this for initial guess and regularization term if is_RL_training=True
            '--sample=%d' % trajopt_sample_idx_for_planner,
            '--knots_per_mode=%d' % knots_per_mode,
            '--n_step=%d' % n_step,
            '--feas_tol=%.6f' % feas_tol,
            '--opt_tol=%.6f' % opt_tol,
            '--stride_length=%.3f' % task_sl,
            '--pelvis_height=%.3f' % task_ph,
            '--time_limit=0',  # we always dynamically adjust the time limit
            '--use_ipopt=%s' % str(use_ipopt).lower(),
            '--log_data=true',
            '--spring_model=%s' % str(spring_model).lower(),
            '--dir_and_prefix_FOM=%s' % dir_and_prefix_FOM_reg,
            '--dir_data=%s' % dir_planner_data,
            '--print_level=0',
            '--completely_use_trajs_from_model_opt_as_target=false',   # completely_use_trajs_from_model_opt_as_target=false because of hybrid_rom_mpc
            '--close_sim_gap=%s' % str(close_sim_gap).lower(),
            '--is_RL_training=true',
            '--get_RL_gradient_offline=true',
            '--debug_mode=true',
            '--solve_idx_for_read_from_file=%d' % solve_idx_for_read_from_file,
            '--path_model_params=%s' % path_model_params,
            '--path_var=%s' % path_var,
            '--min_mpc_thread_loop_duration=%.3f' % min_mpc_thread_loop_duration,
            ]

        # Run process
        return self.RunCommand(planner_cmd, False, True)

    def read_mpc_solution_files(self, x0: np.array, dict_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        dir_planner_data = dict_data["reevaluation_info"]["dir_planner_data"]
        solve_idx_for_read_from_file = dict_data["time_idx"]

        path_a = dir_planner_data + '%d_a.csv' % solve_idx_for_read_from_file
        path_grad_a = dir_planner_data + '%d_grad_a_wrt_theta.csv' % solve_idx_for_read_from_file
        path_a_noise = dir_planner_data + '%d_a_noise.csv' % solve_idx_for_read_from_file
        a = np.loadtxt(path_a)
        grad_a = np.loadtxt(path_grad_a, delimiter=',')
        a_noise = np.loadtxt(path_a_noise)

        return {"a": a, "grad_a": grad_a, "a_noise": a_noise}

    def mpc_diff(self, mpc_outputs: Dict):
        # mpc_outputs contains minibatches' mpc_output
        gradients = []
        for mpc_output in mpc_outputs:
            gradients.append(mpc_output["grad_a"])
        gradients = np.array(gradients)
        return gradients


    ################## Copied and modified from `run_sim_cost_study.py` ######################
    # We assume cmd is string or a list of string
    # WARNING: when we use shell=True, p.kill() won't kill the process. Need to use the function `KillProcess` below.
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

