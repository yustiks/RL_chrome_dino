from stable_baselines.common.policies import CnnPolicy

import base64
import os
import time
from collections import deque
from io import BytesIO

import cv2
import gym
import numpy as np
from PIL import Image
from gym import spaces
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from stable_baselines import PPO2
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv


class EnvironmentChromeTRex(gym.Env):

    def __init__(self,
                 screen_width,  # width of the compressed image
                 screen_height,  # height of the compressed image
                 chromedriver_path: str = 'chromedriver'
                 ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.chromedriver_path = chromedriver_path
        self.num_observation = 0

        self.action_space = spaces.Discrete(3)  # set of actions: do nothing, jump, down
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_width, self.screen_height, 4),
            dtype=np.uint8
        )
        # connection to chrome
        _chrome_options = webdriver.ChromeOptions()
        _chrome_options.add_argument("--mute-audio")
        _chrome_options.add_argument("disable-infobars")
        # _chrome_options.add_argument("--disable-gpu") # if running on Windows

        self._driver = webdriver.Chrome(
            executable_path=self.chromedriver_path,
            options=_chrome_options
        )
        self.current_key = None
        # current state represented by 4 images
        self.state_queue = deque(maxlen=4)

        self.actions_map = [
            Keys.ARROW_RIGHT,  # do nothing
            Keys.ARROW_UP,  # jump
            Keys.ARROW_DOWN  # down
        ]
        action_chains = ActionChains(self._driver)
        self.keydown_actions = [action_chains.key_down(item) for item in self.actions_map]
        self.keyup_actions = [action_chains.key_up(item) for item in self.actions_map]

    def reset(self):
        try:
            self._driver.get('chrome://dino')
        except WebDriverException as e:
            print(e)

        WebDriverWait(self._driver, 10).until(
            EC.presence_of_element_located((
                By.CLASS_NAME,
                "runner-canvas"
            ))
        )

        # trigger game start
        self._driver.find_element_by_tag_name("body").send_keys(Keys.SPACE)

        return self._next_observation()

    def _get_image(self):
        LEADING_TEXT = "data:image/png;base64,"
        _img = self._driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL()"
        )
        _img = _img[len(LEADING_TEXT):]
        return np.array(
            Image.open(BytesIO(base64.b64decode(_img)))
        )

    def _next_observation(self):
        image = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2GRAY)
        image = image[:500, :480]  # cropping
        image = cv2.resize(image, (self.screen_width, self.screen_height))

        self.num_observation += 1
        self.state_queue.append(image)

        if len(self.state_queue) < 4:
            return np.stack([image] * 4, axis=-1)
        else:
            return np.stack(self.state_queue, axis=-1)

        return image

    def _get_score(self):
        try:
            num = int(''.join(self._driver.execute_script("return Runner.instance_.distanceMeter.digits")))
        except:
            num = 0
        return num

    def _get_done(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def step(self, action: int):
        self._driver.find_element_by_tag_name("body") \
            .send_keys(self.actions_map[action])

        obs = self._next_observation()

        done = self._get_done()
        reward = .1 if not done else -1

        time.sleep(.015)

        return obs, reward, done, {"score": self._get_score()}

    def render(self, mode: str = 'human'):
        img = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2RGB)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


import imageio

from tqdm import tqdm

if __name__ == '__main__':
    env_lambda = lambda: EnvironmentChromeTRex(
        screen_width=96,
        screen_height=96,
        chromedriver_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "chromedriver"
        )
    )
    do_train = False
    num_cpu = 1
    save_path = "chrome_dino_ppo_cnn_1980000_steps.zip"
    env = SubprocVecEnv([env_lambda for i in range(num_cpu)])

    if do_train:
        checkpoint_callback = CheckpointCallback(
            save_freq=200000,
            save_path='./.checkpoints/',
            name_prefix=save_path,
        )
        model = PPO2(
            CnnPolicy,
            env,
            verbose=1,
            tensorboard_log="./.tb_chromedino_env/",
        )
        model.learn(
            total_timesteps=2000000,           callback=[checkpoint_callback]
        )
        model.save(save_path)

    model = PPO2.load('./.checkpoints/'+save_path, env=env)

    images = []

    obs = env.reset()
    img = model.env.render(mode='rgb_array')

    for i in tqdm(range(500)):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        # env.render(mode='human')

        img = env.render(mode='rgb_array')

    imageio.mimsave('dino.gif', [np.array(img) for i, img in enumerate(images)], fps=15)

    exit()
