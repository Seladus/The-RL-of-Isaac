from time import sleep
import torch
import numpy as np
import torch.nn.functional as F
import tqdm
import random
import math

from torch.utils.tensorboard import SummaryWriter
from torch import nn
from utils.prioritized_replay_buffer import PrioritizedReplayBuffer
from utils.schedule import LinearSchedule


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(
            torch.full((out_features, in_features), sigma_init)
        )
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.deterministic = False
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def set_deterministic(self, deterministic):
        self.deterministic = deterministic

    def forward(self, input):
        if not self.deterministic:
            self.epsilon_weight.normal_()
            bias = self.bias
            if bias is not None:
                self.epsilon_bias.normal_()
                bias = bias + self.sigma_bias * self.epsilon_bias.data
            return F.linear(
                input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias
            )
        else:
            return F.linear(input, self.weight, self.bias)


class DQN(nn.Module):
    def __init__(
        self,
        inputs,
        outputs,
        M=[256, 256, 256],
        Mf=[256, 256],
        noisy=True,
        dueling=True,
        device="cuda",
    ):
        super(DQN, self).__init__()
        self.device = device
        self.noisy = noisy
        self.dueling = dueling

        if dueling:
            self.fdense1 = nn.Linear(inputs, Mf[0])
            self.fdense2 = nn.Linear(Mf[0], Mf[1])

        in_size = inputs if not dueling else Mf[-1]

        if noisy:
            self.dense1 = NoisyLinear(in_size, M[0])
            self.dense2 = NoisyLinear(M[0], M[1])
            self.dense3 = NoisyLinear(M[1], M[2])
            self.final = NoisyLinear(M[2], outputs)
            if dueling:
                self.vdense1 = NoisyLinear(in_size, M[0])
                self.vdense2 = NoisyLinear(M[0], M[1])
                self.vdense3 = NoisyLinear(M[1], M[2])
                self.vfinal = NoisyLinear(M[2], 1)
        else:
            self.dense1 = nn.Linear(in_size, M[0])
            self.dense2 = nn.Linear(M[0], M[1])
            self.dense3 = nn.Linear(M[1], M[2])
            self.final = nn.Linear(M[2], outputs)
            if dueling:
                self.vdense1 = nn.Linear(in_size, M[0])
                self.vdense2 = nn.Linear(M[0], M[1])
                self.vdense3 = nn.Linear(M[1], M[2])
                self.vfinal = nn.Linear(M[2], 1)
        self.layers = [self.dense1, self.dense2, self.dense3, self.final]

    def _deterministic_forward_dueling(self, x):
        x = F.relu(self.fdense1(x))
        x = F.relu(self.fdense2(x))
        for f in self.layers:
            f.set_deterministic(True)
        a = F.relu(self.dense1(x))
        a = F.relu(self.dense2(a))
        a = F.relu(self.dense3(a))
        a = self.final(a)
        v = F.relu(self.vdense1(x))
        v = F.relu(self.vdense2(v))
        v = F.relu(self.vdense3(v))
        v = self.vfinal(v)
        for f in self.layers:
            f.set_deterministic(False)
        return v + (a - a.mean())

    def _deterministic_forward(self, x):
        assert self.noisy
        for f in self.layers:
            f.set_deterministic(True)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))
        x = self.final(x)
        for f in self.layers:
            f.set_deterministic(False)
        return x

    def _forward_dueling(self, x):
        x = F.relu(self.fdense1(x))
        x = F.relu(self.fdense2(x))

        a = F.relu(self.dense1(x))
        a = F.relu(self.dense2(a))
        a = F.relu(self.dense3(a))
        a = self.final(a)

        v = F.relu(self.vdense1(x))
        v = F.relu(self.vdense2(v))
        v = F.relu(self.vdense3(v))
        v = self.vfinal(v)
        return v + (a - a.mean())

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, deterministic=False):
        x = x.to(self.device)
        if deterministic and self.noisy:
            if not self.dueling:
                return self._deterministic_forward(x)
            else:
                return self._deterministic_forward_dueling(x)
        else:
            if not self.dueling:
                x = F.relu(self.dense1(x))
                x = F.relu(self.dense2(x))
                x = F.relu(self.dense3(x))
                x = self.final(x)
                return x
            else:
                return self._forward_dueling(x)

    def get_discrete_action(self, x, deterministic=False):
        return (
            self.forward(
                torch.from_numpy(np.array(x)).float().to(self.device).unsqueeze(0),
                deterministic=deterministic,
            )
            .max(1)[1]
            .cpu()
            .item()
        )


class PER_DQNTrainer:
    def __init__(
        self,
        env,
        epsilon_mini=0.05,
        epsilon_ini=1.0,
        tau=0.05,
        lr=1e-4,
        startup_steps=1000,
        buffer_size=1000,
        update_freq=1,
        epsilon_decay_steps=100000,
        update_beta_timesteps=100000,
        batch_size=32,
        gamma=0.99,
        max_grad_norm=10,
        noisy=True,
        double=True,
        dueling=True,
        hidden_net_size=256,
        device="cuda",
        _backup_restart_env=None,
    ):
        self.writer = SummaryWriter()
        self._backup_restart_env = _backup_restart_env
        self.env = env
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.startup_steps = startup_steps
        self.beta = 0.5
        self.alpha = 0.7
        self.replay_buffer = PrioritizedReplayBuffer(
            self.buffer_size,
            self.alpha,
            device=device,
        )
        self.beta_schedule = LinearSchedule(
            update_beta_timesteps, initial_p=self.beta, final_p=1.0
        )
        self.epsilon_schedule = LinearSchedule(
            epsilon_decay_steps, initial_p=epsilon_ini, final_p=epsilon_mini
        )

        inputs = env.observation_space.shape[0]
        n_actions = env.action_space.n
        lr = lr
        m = hidden_net_size
        M = [m, m, m]
        Mf = [m, m]
        self.q = DQN(inputs, n_actions, M, Mf, noisy, dueling, device=device).to(device)
        self.q_target = DQN(inputs, n_actions, M, Mf, noisy, dueling, device=device).to(
            device
        )

        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.epsilon = epsilon_ini
        self.epsilon_mini = epsilon_mini
        self.tau = tau
        self.gamma = gamma
        self.noisy = noisy
        self.double = double
        # self.criterion = F.mse_loss
        self.criterion = F.smooth_l1_loss
        self.max_grad_norm = max_grad_norm

        self._current_step = 0

    def _compute_dqn_loss(self, obs, actions, next_obs, rewards, dones, td_error=False):
        q_pred = self.q(obs)
        q_next_pred = self.q_target(next_obs).detach()

        state_action_values = q_pred.gather(1, actions.unsqueeze(dim=-1))
        target_values = rewards + (1 - dones) * self.gamma * q_next_pred.max(1)[
            0
        ].unsqueeze(-1)
        loss = self.criterion(state_action_values, target_values)
        return loss, q_pred, state_action_values - target_values if td_error else None

    def _compute_double_dqn_loss(
        self, obs, actions, next_obs, rewards, dones, td_error=False
    ):
        q_pred = self.q(obs)
        best_action_idxs = q_pred.max(1, keepdim=True)[1]
        target_q_values = self.q_target(next_obs).gather(1, best_action_idxs).detach()
        y_target = rewards + (1.0 - dones) * self.gamma * target_q_values
        y_target = y_target.detach()
        # actions is a one-hot vector
        state_action_values = q_pred.gather(1, actions.unsqueeze(dim=-1))
        loss = self.criterion(state_action_values, y_target)

        return loss, q_pred, state_action_values - y_target if td_error else None

    def train_step(self):
        (
            weights,
            idxes,
            (obs, actions, next_obs, rewards, dones),
        ) = self.replay_buffer.sample(
            self.batch_size, self.beta_schedule.value(self._current_step)
        )
        # Compute Loss
        if self.double:
            loss, q_values, td_error = self._compute_double_dqn_loss(
                obs, actions, next_obs, rewards, dones, td_error=True
            )
        else:
            loss, q_values, td_error = self._compute_dqn_loss(
                obs, actions, next_obs, rewards, dones, td_error=True
            )
        with torch.no_grad():
            weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
        loss *= weight

        self.writer.add_scalar("train/q_loss", loss, self._current_step)
        self.writer.add_scalar("train/q_values", q_values.mean(), self._current_step)

        # Update q network
        self.q_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.q_optimizer.step()

        # Update PER
        self.replay_buffer.update_priorities(
            idxes, np.abs(td_error.detach().cpu().squeeze().numpy()) + 1e-6
        )

        # Update target q network
        soft_update_from_to(self.q, self.q_target, self.tau)

        self._current_step += 1

    def train(self, nb_epochs=10, nb_steps_per_epoch=1000, nb_episode_per_eval=5):
        self._current_step = 0
        best_eval = -100000000

        env = self.env
        # warmup steps
        print("Filling replay buffer")
        self.run_random_startup(self.startup_steps)
        i = 1
        while i <= nb_epochs:
            try:
                self.run_epoch(nb_steps_per_epoch, i, nb_epochs)
                ret_eval = self.run_eval(nb_episode_per_eval)
                if ret_eval > best_eval:
                    best_eval = ret_eval
                    torch.save(self.q, f"saves/best_model.pt")
                self.writer.add_scalar("evaluation/Average Reward", ret_eval, i)
                i += 1
            except:
                sleep(10)
                env.close()
                if self._backup_restart_env:
                    self.env = self._backup_restart_env()
                    env = self.env

        env.close()

    def run_random_startup(self, nb_steps):
        env = self.env
        next_obs = env.reset()
        obs = next_obs
        done = False

        pbar = tqdm.trange(0, nb_steps, position=0, disable=False)
        pbar.set_description(f"STARTUP EPOCH")
        for i in pbar:
            if done:
                env.reset()
            # chose action
            action = env.action_space.sample()
            obs = next_obs
            next_obs, reward, done, info = env.step(action)
            # add experience to replay buffer
            self.replay_buffer.add(obs, action, next_obs, reward, done)

    def run_epoch(self, nb_steps, current_epoch, nb_tot_epoch):
        env = self.env
        next_obs = env.reset()
        obs = next_obs
        done = False

        pbar = tqdm.trange(0, nb_steps, position=0, disable=False)
        pbar.set_description(f"epoch {current_epoch}/{nb_tot_epoch}")
        for i in pbar:
            for i in range(self.update_freq):
                if done:
                    env.reset()
                # chose action
                if not self.noisy:
                    if random.random() < self.epsilon_schedule.value(
                        self._current_step
                    ):
                        action = env.action_space.sample()
                    else:
                        action = self.q.get_discrete_action(obs)
                else:
                    action = self.q.get_discrete_action(obs)
                obs = next_obs
                next_obs, reward, done, info = env.step(action)
                # add experience to replay buffer
                self.replay_buffer.add(obs, action, next_obs, reward, done)
            self.train_step()
        torch.save(self.q, f"saves/{current_epoch}.pt")

    def run_eval(self, nb_eval, max_timesteps=2000):
        with torch.no_grad():
            env = self.env
            total_reward = 0
            pbar = tqdm.trange(nb_eval)
            pbar.set_description("Running evaluation...")
            for i in pbar:
                obs = env.reset()
                done = False
                steps = 0
                while not done and steps < max_timesteps:
                    action = self.q.get_discrete_action(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    steps += 1
            print(
                f"Evaluation for {nb_eval} episodes : Mean Reward {total_reward / nb_eval}"
            )
            return total_reward / nb_eval
