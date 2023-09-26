This is a repo for the preprint "Reinforcement Learning for Reduced-order Models of Legged Robots" by Yu-Ming Chen, Hien Bui and Michael Posa.

Warning: This code was used during the development phase, so it may appear somewhat disorganized.

# Installation

OS: Ubuntu 20.04

The code calls the controller and simulation from another repo `dairlib` (https://github.com/DAIRLab/dairlib/tree/goldilocks-model-dev), so it is necessary to build `dairlib` separately (see the installation instruction in that repo).

Below is a installation script for this repo (doesn't include `dairlib`):
```
sudo apt install python3.8-venv
python3.8 -m venv ~/venv/rom_rl
source ~/venv/rom_rl/bin/activate

pip install --upgrade pip

git clone git@github.com:yminchen/rom-mpc-rl.git
cd rom-mpc-rl
pip3 install -e .

pip3 install scipy
pip3 install psutil
pip3 install mujoco
pip3 install wandb
pip3 install tensorboard
pip3 install gym
pip3 install gymnasium
pip3 install torch torchvision torchaudio
pip3 install optuna  # for cma-es

```

# Running the code
Training with ppo:
```
source ~/venv/rom_rl/bin/activate
python3 scripts/test_train_cassie_optimal_rom_ppo.py --train_mode
```

Training with CMA-ES:
```
source ~/venv/rom_rl/bin/activate
python3 common/cma_es.py
```
