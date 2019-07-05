import sys
import os
import cv2
import gym_gvgai
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
from baselines import logger
from baselines.common.atari_wrappers import *
from baselines.common import set_global_seeds
from baselines.bench import Monitor
from util import obs_space_info, obs_to_dict, dict_to_obs, copy_obs_dict

sys.path.append("/home/jupyter/Notebooks/Chang/HardRLWithYoutube")

USE_IMMITATION_ENV = False
if USE_IMMITATION_ENV:
    print(sys.path)
    from TDCFeaturizer import TDCFeaturizer
    from train_featurizer import generate_dataset


def worker(remote, parent_remote, env_fn_wrapper, level_selector=None):
    parent_remote.close()
    env = env_fn_wrapper.x()
    level_selector = level_selector
    level = "/home/jupyter/Notebooks/Chang/HardRLWithYoutube/nnrunner/a2c_gvgai/examples/gridphysics/zelda_lvl0.txt"
    score = 0
    finished = False
    last_mes = None
    print("Worker launched")
    # print("worker level: ", level_selector.get_level() is None)
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            if finished:
                # Act like agent is still playing
                remote.send(last_mes)
            else:
                ob, reward, done, info = env.step(data)
                score += reward
                if done:
                    if level_selector is not None:
                        if level_selector.get_game() == "boulderdash":
                            level_selector.report(level, score >= 20)
                        else:
                            level_selector.report(level, False if info['winner'] == 'PLAYER_LOSES' else True)
                        try:
                            level = level_selector.get_level()
                        except Exception as e:
                            print("step: ", level_selector.get_level() is None)
                            raise e
                        
                        # print(level)
                        if level is not None:
                            # env.unwrapped._setLevel(level)
                            pass
                        else:
                            finished = True
                    ob = env.reset()
                    if finished:
                        last_mes = (ob, reward, False, info)
                    score = 0
                remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            if finished:
                remote.send(last_mes)
            else:
                # print("reset: ", level_selector.get_level() is None)
                if level_selector is not None:
                    try:
                        level = level_selector.get_level()                   
                    except Exception as e:
                        print(type(level))
                        print(level)
                        print(level, "file: " + str(os.path.isfile(level)))
                        raise e
                    env.unwrapped._setLevel(level)
                ob = env.reset()
                score = 0
                remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            score = 0
            break
        elif cmd == 'render':
            # env.render(**data)
            # env.render(mode='human')
            remote.send(env.render(**data))
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None, level_selector=None, experiment=None, dataset=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn), level_selector))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            print("start processes")
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        if USE_IMMITATION_ENV:
            self.featurizer = TDCFeaturizer(84, 84, 84, 84, feature_vector_size=1024, learning_rate=0, experiment_name=experiment)
            self.featurizer.load(dataset)
            video_dataset = generate_dataset(experiment, framerate=60, width=84, height=84)[0]
            self.featurized_dataset = self.featurizer.featurize(video_dataset)
            self.checkpoint_indexes = [0] * nenvs

            self.rewards = 0
            self.counter = 0

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        obs, rews, dones = np.stack(obs), np.stack(rews), np.stack(dones)
        # print("obs shape: ", obs.shape)
        if USE_IMMITATION_ENV:
            obs = cv2.resize(obs, (84, 84))
            state_feature_vectors = self.featurizer.featurize(obs)
            dot_products = [np.dot(state_feature_vectors[i], self.featurized_dataset[self.checkpoint_indexes[i]]) for i in range(self.nenvs)]
            gamma_threshold = 0.5
            immitation_rewards = [0.5 if dot_product > gamma_threshold else 0 for dot_product in dot_products]

            rews += immitation_rewards
            mean_rews = np.mean(rews)
            self.rewards += mean_rews
            self.counter += 1
            if self.counter == 10000:
                print('10000 rewards: ', self.rewards)
                self.rewards = 0
                self.counter = 0
            #if dot_products[0] > 0.5:
            #    print(dot_products, immitation_rewards[0], rews)
            self.checkpoint_indexes = [self.checkpoint_indexes[i] + 1 if immitation_rewards[i] > 0 else self.checkpoint_indexes[i] for i in range(self.nenvs)]
            #print(self.checkpoint_indexes[0])

            self.checkpoint_indexes = [0 if dones[i] else self.checkpoint_indexes[i] for i in range(self.nenvs)]
        return obs, rews, dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        print("reset")
        return np.stack([remote.recv() for remote in self.remotes])

    # def render(self):
    #     print("render")
    #     for remote in self.remotes:
    #         remote.send(('render', {'mode':'human'}))

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def get_images(self):
        for remote in self.remotes:
            remote.send(('render', {'mode':'rgb_array'}))
        imgs = [remote.recv() for remote in self.remotes]
        return imgs

class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns, spaces=None, level_selector=None, experiment=None, dataset=None):
        """
        Arguments:
        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(env_fns)
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

        self.finsihed = [False for _ in range(self.num_envs)]
        self.last_mes = [None for _ in range(self.num_envs)]
        self.level_selector = level_selector
        if USE_IMMITATION_ENV:
            self.featurizer = TDCFeaturizer(92, 92, 84, 84, feature_vector_size=1024, learning_rate=0, experiment_name=experiment)
            self.featurizer.load(dataset)
            video_dataset = generate_dataset(experiment, framerate=60, width=84, height=84)[0]
            self.featurized_dataset = self.featurizer.featurize(video_dataset)
            self.checkpoint_indexes = [0] * nenvs

            self.rewards = 0
            self.counter = 0

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            level = None
            if self.finsihed[e]:
                obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = zip(*self.last_mes[e])
            else:
                action = self.actions[e]
                # if isinstance(self.envs[e].action_space, spaces.Discrete):
                #    action = int(action)

                obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
                if self.buf_dones[e]:
                    if self.level_selector is not None:
                        self.level_selector.report(level, False if self.buf_infos[e]['winner'] == 'PLAYER_LOSES' else True)
                        level = self.level_selector.get_level()
                        if level is not None:
                            self.envs[e].unwrapped._setLevel(level)
                        else:
                            self.finsihed[e] = True
                    obs = self.envs[e].reset()
                    if self.finsihed[e]:
                        self.last_mes[e] = (obs, self.buf_rews[e], False, self.buf_infos[e])

                self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            level = None
            if self.finsihed[e]:
                obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = zip(*self.last_mes[e])
            else:
                if self.level_selector is not None:
                    level = self.level_selector.get_level()
                    self.envs[e].unwrapped._setLevel(level)
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def reset_task(self):
        for e in range(self.num_envs):
            ob = self.envs[e].reset_task()
            self._save_obs(e, ob)
        return self._obs_from_buf() 

    def close(self):
        pass


def wrap_gvgai(env, frame_stack=False, scale=False, clip_rewards=False, noop_reset=False, frame_skip=False, scale_float=False):
    """Configure environment for DeepMind-style Atari.
    """
    if scale_float:
        env = ScaledFloatFrame(env)
    if scale:
        env = WarpFrame(env)
    if frame_skip:
        env = MaxAndSkipEnv(env, skip=4)
    if noop_reset:
        env = NoopResetEnv(env, noop_max=30)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def make_gvgai_env(env_id, num_env, seed, start_index=0, level_selector=None, experiment="PE", dataset="zelda"):
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            # env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=True)
            return wrap_gvgai(env)
        return _thunk
    
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)], level_selector=level_selector, experiment=experiment, dataset=dataset)
    # return DummyVecEnv([make_env(i + start_index) for i in range(num_env)], level_selector=level_selector, experiment=experiment, dataset=dataset)


def make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return wrap_deepmind(env, **wrapper_kwargs)
        return _thunk
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

