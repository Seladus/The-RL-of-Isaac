import sys
import signal
import socket
import gym
import json
import numpy as np
import cv2
import tqdm

from itertools import product
from time import sleep
from utils.virtual_controller import VirtualKeyboard
from utils.img import ImageCapture
from utils.utils import changeWindowName, kill_process, kill_steam, run_game, kill_game
from utils.wrappers import FrameStack, MaxAndSkipEnv, ScaledFloatFrame, WarpFrame
from speedhack.speedhack import run_speed_hack
from gym import spaces

gym.logger.set_level(40)

RECV_SIZE = 4096


class Isaac(gym.Env):
    def __init__(
        self,
        speed_hack=True,
        use_virtual_controller=True,
        max_timesteps=None,
        max_enemies_to_observe=1,
        max_enemy_projectiles_to_observe=15,
        max_tears_to_observe=5,
        _semaphore=None,
        _alter_window_name=True,
    ) -> None:
        super().__init__()
        if _semaphore:
            _semaphore.acquire(block=True)
        self.max_enemies = max_enemies_to_observe
        self.max_projectiles = max_enemy_projectiles_to_observe
        self.max_tears = max_tears_to_observe
        self.max_steps = max_timesteps
        self.use_virtual_controller = use_virtual_controller
        # generate action dict
        p = list(product(range(0, 5), repeat=2))
        self.action_dict = {
            k: self._tuple_to_vect(v) for k, v in zip(list(range(len(p))), p)
        }

        self.observation_shape = (
            2
            + 2
            + self.max_tears * 4
            + self.max_enemies * 4
            + self.max_projectiles * 4,
        )
        self.observation_space = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=self.observation_shape,
            dtype=np.float32,
        )

        # init action space
        self.action_space = spaces.Discrete(
            len(p),
        )

        self.gamestate = {}
        self.previous_gamestate = {}

        exe_path = "C:/program files (x86)/steam/Steam.exe"
        args = [
            "C:/program files (x86)/steam/Steam.exe",
            "-applaunch",
            "250900",
            "--set-stage=1",
            "--set-stage-type=0",
        ]

        kill_steam()
        self.game_process = run_game(exe_path, args)
        self.windown_name = (
            f"{self.game_process.pid}"
            if _alter_window_name
            else f"Binding of Isaac: Repentance"
        )
        if speed_hack:
            run_speed_hack("Binding of Isaac: Repentance", 500)

        # Open connection with game
        self.host, self.port = "127.0.0.1", 6942
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setblocking(True)
        self.connected = False
        while not self.connected:
            try:
                self.s.connect((self.host, self.port))
                self.connected = True
                # print("Successful connection with game.", flush=True)
                changeWindowName("Binding of Isaac: Repentance", self.windown_name)
            except:
                # print("Failed to connect to game, retrying...", flush=True)
                sleep(1)
        self.controller = VirtualKeyboard(self.windown_name, wait=0.5)
        self.capturer = ImageCapture(self.windown_name)

        self._current_step = 0

        if _semaphore:
            _semaphore.release()

    @staticmethod
    def _tuple_to_vect(tuple, sizes=[4, 4]):
        vectors = [np.zeros(i) for i in sizes]
        for idx, v in enumerate(tuple):
            if v != 0:
                vectors[idx][v - 1] = 1.0
        return np.concatenate(vectors)

    def compute_reward(self):
        if self.previous_gamestate:
            dmg_taken = (
                self.gamestate["damagesTaken"] - self.previous_gamestate["damagesTaken"]
            )
            mob_dmg_taken = (
                self.gamestate["mobDamagesTaken"]
                - self.previous_gamestate["mobDamagesTaken"]
            )
            mob_dmg_taken = 1.0 if mob_dmg_taken != 0 else 0.0
            mob_killed = (
                self.gamestate["mobKilled"] - self.previous_gamestate["mobKilled"]
            )
            # ======
            is_too_close = 0
            for e in range(len(self.gamestate["enemy_positions"])):
                dist_from_enemy = np.sqrt(
                    (
                        self.gamestate["player_position"][0]
                        - self.gamestate["enemy_positions"][e][0]
                    )
                    ** 2
                    + (
                        self.gamestate["player_position"][1]
                        - self.gamestate["enemy_positions"][e][1]
                    )
                    ** 2
                )
                is_too_close += (
                    1.0 if dist_from_enemy > 270 or dist_from_enemy < 160 else -1.0
                )
            # ======
            has_died = self.gamestate["isDead"]
            cleared = not self.gamestate["isDead"] and self.gamestate["done"]
            return (
                is_too_close * -0.2
                + dmg_taken * -20.0
                + mob_killed * 20.0
                + has_died * -0.0
                + cleared * 20.0
                + mob_dmg_taken * 5.0
            )
        return 0.0

    def observe(self, gamestate):
        observation = np.zeros(self.observation_space.shape)
        ptr = 0
        # player infos
        observation[ptr : ptr + 4] = (
            gamestate["player_position"] + gamestate["player_velocity"]
        )
        observation[ptr : ptr + 4] /= [570, 410, 10, 10]
        ptr = ptr + 4
        # enemies infos
        for i in range(self.max_enemies):
            if i < len(gamestate["enemy_positions"]):
                observation[ptr : ptr + 5] = (
                    gamestate["enemy_positions"][i]
                    + gamestate["enemy_velocities"][i]
                    + [gamestate["enemy_healths"][i]]
                )
                observation[ptr : ptr + 5] /= [570, 410, 10, 10, 300]
            ptr += 4
        # enemies projectiles infos
        for i in range(self.max_projectiles):
            if i < len(gamestate["enemy_projectiles_positions"]):
                observation[ptr : ptr + 4] = (
                    gamestate["enemy_projectiles_positions"][i]
                    + gamestate["enemy_projectiles_velocities"][i]
                )
                observation[ptr : ptr + 4] /= [570, 410, 10, 10]
            ptr += 4
        # tears infos
        for i in range(self.max_tears):
            if i < len(gamestate["tears_positions"]):
                observation[ptr : ptr + 4] = (
                    gamestate["tears_positions"][i] + gamestate["tears_velocities"][i]
                )
                observation[ptr : ptr + 4] /= [570, 410, 10, 10]
            ptr += 4
        return observation

    def reset(self, return_info=False, options=None, seed=None):
        self._current_step = 0
        self.gamestate = {}
        self.previous_gamestate = {}
        self.controller.update(self.controller.get_null_action())
        request = {
            "action": "reset",
            "use_virtual_controller": self.use_virtual_controller,
        }

        self.s.sendall((json.dumps(request) + "\n").encode("utf-8"))
        msg = self.s.recv(RECV_SIZE)
        if msg == b"":
            raise Exception("Connection Lost")
        gamestate = json.loads(msg)

        # observation = self.capturer.capture_frame() # => not reliable
        observation = self.observe(gamestate)
        if not return_info:
            return observation
        else:
            return observation, {}

    def step(self, action):
        """obs, reward, done, infos"""
        act_to_send_low = np.where(self.action_dict[action][0:4] == 1.0)[0]
        act_to_send_high = np.where(self.action_dict[action][4::] == 1.0)[0]
        if not act_to_send_low:
            act_to_send_low = -1
        if not act_to_send_high:
            act_to_send_low = -1
        # Get subsidiary information like reward or done
        ## Send instruction to game
        request = {
            "action": "step",
        }

        if not self.use_virtual_controller:
            request["player_move"] = int(act_to_send_high)
            request["player_fire"] = int(act_to_send_low)

        self.s.sendall((json.dumps(request) + "\n").encode("utf-8"))

        if self.use_virtual_controller:
            self.controller.update(self.action_dict[action])

        ## Receive gamestate from game (if state is terminal for example)
        msg = self.s.recv(RECV_SIZE)
        if msg == b"":
            raise Exception("Connection Lost")
        gamestate = json.loads(msg)

        # Observe the environment
        # observation = self.capturer.capture_frame() # => do not work reliably
        observation = self.observe(gamestate)

        if gamestate["done"]:
            # Send input
            if self.use_virtual_controller:
                self.controller.update(self.controller.get_null_action())

        self.previous_gamestate = self.gamestate
        self.gamestate = gamestate

        reward = self.compute_reward()

        if self.max_steps and self._current_step >= self.max_steps:
            return observation, reward, True, gamestate

        self._current_step += 1
        return observation, reward, gamestate["done"], gamestate

    def close(self):
        print("Closing environement")
        # request = {"action": "close"}
        # self.s.sendall((json.dumps(request) + "\n").encode("utf-8"))
        self.s.close()
        kill_process(self.windown_name)


def signal_handler(sig, frame):
    kill_game()
    sys.exit(0)


# register environment
gym.envs.register(id="isaac-v0", entry_point="isaac_env:Isaac")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # env = Isaac(speed_hack=True)
    env = gym.make("isaac-v0", speed_hack=False)
    obs = env.reset()
    done = False
    count = 0
    # for i in tqdm.trange(0, 1000):
    while True:
        if done:
            env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        # print(f"Step {count} => {action}")
        count += 1
