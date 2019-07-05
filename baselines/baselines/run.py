import sys
import multiprocessing 
import os
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_mujoco_env, make_atari_env
from baselines.common.tf_util import save_state, load_state, get_session
from baselines import bench, logger
from importlib import import_module

from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import atari_wrappers, retro_wrappers

sys.path.append("/home/jupyter/Notebooks/Chang/HardRLWithYoutube/nnrunner/a2c_gvgai")
import nnrunner.a2c_gvgai.level_selector as ls

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

store = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires 
# importing retro here, and for some reason that crashes tensorflow 
# in ubuntu 
_game_envs['retro'] = set([
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
])


def train(args, extra_args):
    global store
    env_type, env_id = get_env_type(args.env)
        
    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    lvlFileName = "/home/jupyter/Notebooks/Chang/HardRLWithYoutube/nnrunner/a2c_gvgai/examples/gridphysics/zelda_lvl0.txt"
    selector = ls.LevelSelector.get_selector("map-elite", os.path.basename(lvlFileName),os.path.dirname(os.path.abspath(lvlFileName)), max=1)
    env = build_env(args, selector)
    env.reset()
    # store.reset()

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
 
       
    env.reset()
    print(type(env))
    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))
    model = learn(
        env=env,  
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args, selector=None):
    global store
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = args.seed    

    env_type, env_id = get_env_type(args.env)
    print(env_type, env_id, nenv, args.num_env)
    if env_type == 'mujoco':
        get_session(tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1, 
                                   inter_op_parallelism_threads=1))

        if args.num_env:
            env = SubprocVecEnv([lambda: make_mujoco_env(env_id, seed + i if seed is not None else None, args.reward_scale) for i in range(args.num_env)])    
        else:
            env = DummyVecEnv([lambda: make_mujoco_env(env_id, seed, args.reward_scale)])

        env = VecNormalize(env)

    elif env_type == 'atari':
        if alg == 'acer':
            env = make_atari_env(env_id, nenv, seed)#, wrapper_kwargs={'clip_rewards': False})
        elif alg == 'deepq':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir())
            env = atari_wrappers.wrap_deepmind(env, frame_stack=True, scale=True)
        elif alg == 'trpo_mpi':
            env = atari_wrappers.make_atari(env_id)
            env.seed(seed)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            env = atari_wrappers.wrap_deepmind(env)
            # TODO check if the second seeding is necessary, and eventually remove
            env.seed(seed)
        elif "Zelda" in env_id:
            sys.path.append("/home/jupyter/Notebooks/Chang/HardRLWithYoutube/nnrunner/a2c_gvgai")
            import nnrunner.a2c_gvgai.env as gvgai_env
            frame_stack_size = 4
            print("run zelda")
            env = VecFrameStack(gvgai_env.make_gvgai_env(env_id, nenv, seed, level_selector=selector, experiment="PE", dataset="zelda"), frame_stack_size)
            # env.reset()
            # store = env
        else:
            frame_stack_size = 4
            env = VecFrameStack(make_atari_env(env_id, nenv, seed), frame_stack_size)

    elif env_type == 'retro':
        import retro
        gamestate = args.gamestate or 'Level1-1'
        env = retro_wrappers.make_retro(game=args.env, state=gamestate, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE)
        env.seed(args.seed)
        env = bench.Monitor(env, logger.get_dir())
        env = retro_wrappers.wrap_deepmind_retro(env)
        
    elif env_type == 'classic_control':
        def make_env():
            e = gym.make(env_id)
            e = bench.Monitor(e, logger.get_dir(), allow_early_resets=True)
            e.seed(seed)
            return e
            
        env = DummyVecEnv([make_env])
        

    else:
        raise ValueError('Unknown env_type {}'.format(env_type))
    
    # env.reset()
    print("build env")
    # store.reset()
    # store.reset()

    return env


def get_env_type(env_id):
    if env_id in _game_envs.keys():
        env_type = env_id
        env_id =  [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break 
        assert env_type is not None, 'env_id {} is not recognized in env types{}'.format(env_id, _game_envs.items())

    return env_type, env_id

def get_default_network(env_type):
    if env_type == 'mujoco' or env_type == 'classic_control':
        return 'mlp'
    if env_type == 'atari' or env_type == 'gvgai':
        return 'cnn'

    raise ValueError('Unknown env_type {}'.format(env_type))
    
def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))
    
    return alg_module
        

def get_learn_function(alg):
    return get_alg_module(alg).learn

def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}       
    return kwargs
    
def parse(v): 
    '''
    convert value of a command-line arg to a python object if possible, othewise, keep as string
    '''

    assert isinstance(v, str)
    try:
        return eval(v) 
    except (NameError, SyntaxError): 
        return v


def main(custom_args=[]):
    # configure logger, disable logging in child MPI processes (with rank > 0) 
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = {}
    for arg in custom_args:
        if arg in vars(args).keys():
            vars(args)[arg] = custom_args[arg]
        else:
            extra_args[arg] = custom_args[arg]
    
    #extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}

    
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(format_strs = ['stdout', 'tensorboard'])
    else:
        logger.configure(format_strs = ['stdout', 'tensorboard'])
        rank = MPI.COMM_WORLD.Get_rank()

    model, _ = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)
    

    if args.play:
        logger.log("Running trained model")
        env = build_env(args)
        obs = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs, _, done, _  = env.step(actions)
            env.render()
            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                obs = env.reset()
            


if __name__ == '__main__':
    main()
