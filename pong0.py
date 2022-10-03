import gym
import ale_py

from gym.utils.play import play

env = gym.make("ALE/Pong-v5",render_mode='rgb_array')
env.reset()
play(env, zoom=3, fps=12)

env.close()
