import gym
import signal
import torch
import numpy as np

from per_dqn import PER_DQNTrainer
from isaac_env import signal_handler

signal.signal(signal.SIGINT, signal_handler)

device = "cuda"


def init_env():
    return gym.make(
        "isaac-v0",
        speed_hack=True,
        use_virtual_controller=True,
        max_timesteps=4000,
        max_enemies_to_observe=2,
        max_enemy_projectiles_to_observe=40,
        max_tears_to_observe=7,
    )


env = init_env()


trainer = PER_DQNTrainer(
    env,
    epsilon_decay_steps=750000,
    update_beta_timesteps=750000,
    epsilon_ini=1.0,
    epsilon_mini=0.01,
    gamma=0.95,
    lr=1e-4,
    startup_steps=250000,
    buffer_size=250000,
    batch_size=128,
    device=device,
    noisy=False,
    double=True,
    dueling=True,
    _backup_restart_env=init_env,
    max_grad_norm=10,
    update_freq=5,
    hidden_net_size=512,
)

trainer.train(10000, 5000)
