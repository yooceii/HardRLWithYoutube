from immitation_env.atari.montezuma_immitation_env import MontezumaImmitationEnv
from immitation_env.atari.zelda_immitation_env import ZeldaImmitationEnv
from gym.envs.registration import register
import gym

register(
    id='MontezumaImmitationNoFrameskip-v4',
    entry_point='immitation_env.atari:MontezumaImmitationEnv'
)

register(
    id="ZeldaImmitationNoFrameskip-v4",
    entry_point='immitation_env.atari:ZeldaImmitationEnv'
)
