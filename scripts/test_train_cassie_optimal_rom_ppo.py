import os
import gin
import time
import click
from common.ppo_trainer import PPO


@click.command()
@click.option("--train_mode", is_flag=True, help="Choose between inference and training")
@click.option("--exp_name", default="None", help="Specify the name of experiment")
@click.option("--chkpt_name", default="None", help="Specify the name of checkpoint")
def main(train_mode: bool, exp_name: str, chkpt_name: str):
    # Notes: parameters in PPOConfigs class (experiment_configs.py) are overwritten if specified in
    # cassie_optimal_rom_ppo_config.gin
    path_to_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../configs/ppo_configs/cassie_optimal_rom_ppo_config.gin",
        )
    )
    gin.parse_config_file(path_to_config_file)

    # Notes -- boolean is default to False, so we need to set train_mode flag is we want to train
    #   E.g. python3 scripts/test_train_cassie_optimal_rom_ppo.py --train_mode
    if train_mode:
        ppo_trainer = PPO(gin.REQUIRED)
        ppo_trainer.train()
        # ppo_trainer.train(path_to_policy_chkpt="/home/username/workspace/dairlib_data/goldilocks_models/rl_training/cassie-rom-mpc-ppo_20230511_134705/checkpoints/policy_MPC_36000.pt",
        #                   path_to_critic_chkpt="/home/username/workspace/dairlib_data/goldilocks_models/rl_training/cassie-rom-mpc-ppo_20230511_134705/checkpoints/critic_36000.pt")
    else:
        # gin.bind_parameter("PPOConfigs.sync_with_wandb", False)
        # ppo_trainer = PPO(gin.REQUIRED)
        # path_to_policy_checkpoint = os.path.normpath(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         f"../data/checkpoints/{exp_name}/{chkpt_name}",
        #     )
        # )
        # #ppo_trainer.rollout_trajectory_with_learned_mpc(path_to_policy_checkpoint)
        pass


if __name__ == "__main__":
    main()
