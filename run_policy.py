import torch
import gym
import isaac_env

nb_episodes = 10
model_path = "saves/best_model.pt"
model = torch.load(model_path)

env = gym.make(
    "isaac-v0",
    speed_hack=False,
    use_virtual_controller=True,
    max_timesteps=2000,
    max_enemies_to_observe=1,
    max_enemy_projectiles_to_observe=15,
    max_tears_to_observe=7,
    _alter_window_name=False,
)

for i in range(nb_episodes):
    state = env.reset()

    input()

    done = False
    total_reward = 0
    while not done:
        action = model.get_discrete_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

env.close()
