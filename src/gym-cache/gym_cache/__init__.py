
from gym.envs.registration import register

register(
    id='cache-v0',
    entry_point='gym_cache.envs:CacheEnv',
)
