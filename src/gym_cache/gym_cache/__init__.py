
from gym.envs.registration import register

register(
    id='cache-v0',
    entry_point='gym_cache.envs:CacheEnv',
)

register(
    id='cache-episode-v0',
    entry_point='gym_cache.envs:CacheEpisodeEnv',
)

register(
    id='cache-guessing-game-v0',
    entry_point='gym_cache.envs:CacheGuessingGameEnv',
)

register(
    id='cache-guessing-game-simple-v0',
    entry_point='gym_cache.envs:CacheGuessingGameSimpleEnv',
)

register(
    id='test-lstm-v0',
    entry_point='gym_cache.envs:TestLstmEnv',
)

