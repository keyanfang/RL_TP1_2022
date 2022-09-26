import gym
from gym.utils.play import play

env = gym.make('Pong-v4')
env.reset()
play(env, zoom=3, fps=12)

env.close()