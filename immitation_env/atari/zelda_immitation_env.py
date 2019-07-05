import sys
import gym
from .. import ImmitationWrapper
from TDCFeaturizer import TDCFeaturizer
from train_featurizer import generate_dataset
sys.path.append("nnrunner/a2c_gvgai")

# import gym_gvgai.envs.GVGAI_Env

class ZeldaImmitationEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        print("ZeldaImmitationEnv")
        featurizer = TDCFeaturizer(84, 84, 84, 84, feature_vector_size=1024, learning_rate=0, experiment_name='PE')
        featurizer.load("zelda")
        video_dataset = generate_dataset('PE', framerate=60, width=84, height=84)[0]
        featurized_dataset = featurizer.featurize(video_dataset)
        self._env = ImmitationWrapper(gym.make('gvgai-zelda-lvl0-v0'), featurizer=featurizer, featurized_dataset=featurized_dataset)
        # self._env = gym.make('gvgai-zelda-lvl0-v0')
        self.observation_space = self._env.unwrapped.observation_space
        self.action_space = self._env.unwrapped.action_space

    def seed(self, seed = None):
        return self._env.seed(seed)

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self, mode='human'):
        return self._env.unwrapped.render(mode)

    def close(self):
        self._env.unwrapped.close()

    def get_action_meanings(self):
        return self._env.unwrapped.get_action_meanings()

    def _setLevel(self, level):
        return self._env.unwrapped._setLevel(level)