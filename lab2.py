import gym
from gym.utils.play import play
import imageio
import _pickle as pickle
import numpy as np
import ale_py

D = 80 * 80
neuron = 100
model = {}
model['1'] = np.random.randn(neuron, D) / np.sqrt(D)
model['2'] = np.random.randn(neuron) / np.sqrt(neuron)


def prepro(obs_t):
    obs_t = np.array(obs_t[0])
    obs_t = obs_t[35:195]
    obs_t = obs_t[::2, ::2, 0]
    obs_t[obs_t == 144] = 0
    obs_t[obs_t == 109] = 0
    obs_t[obs_t != 0] = 1  #
    return obs_t.astype(np.float).ravel()


def policy_forward(x):
    h = np.dot(model['1'], x)
    h[h < 0] = 0
    logp = np.dot(model['2'], h)
    p = 1.0 / (1.0 + np.exp(logp))
    return p, h


def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'1': dW1, '2': dW2}


def mycallback(obs_t, action, rew, done):
    #print("action = ", action, " reward = ", rew, "done = ", done)
    imageio.imwrite("jeu.jpg", obs_t)
    jeu = imageio.imread("jeu.jpg")
    imageio.imwrite('outfile.png', obs_t[34:194:4, 12:148:2, 1])
    obs_tp1 = obs_t[34:194:4, 12:148:2, 1]
    np.savetxt("outfileX.txt", delimiter='', X=obs_t[34:194:4, 12:148:2, 1], fmt='%d')
    np.savetxt("outfileY.txt", delimiter='', X=[action], fmt='%d')


grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

if __name__ == "__main__":
    env = gym.make("ALE/Pong-v5", render_mode='rgb_array')
    LEARNING_RATE = 0.01
    GAMMA = 0.99
    obs = env.reset()
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 0
    begin = True


    while begin:
        env.render()
        cur_x = prepro(obs)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob, h = policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

        xs.append(x)
        hs.append(h)
        y = 1 if action == 2 else 0  # a "fake label"
        dlogps.append(y - aprob)

        state2, reward, terminated, truncated, info = env.step(action)

        mycallback(state2, action, reward, terminated)
        reward_sum += reward

        drs.append(reward)
        if terminated:
            episode_number += 1

            epx = np.vstack(xs)
            eph = np.vstack(hs)
            epdlogp = np.vstack(dlogps)
            epr = np.vstack(drs)
            xs, hs, dlogps, drs = [], [], [], []

            discounted_epr = epr
            discounted_epr -= np.mean(discounted_epr)

            running_reward = reward_sum if running_reward is None else running_reward * GAMMA + reward_sum * LEARNING_RATE
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
            reward_sum = 0
            observation = env.reset()
            prev_x = None

        if reward != 0:
            print('ep %d: game finished, reward: %f' % (episode_number, reward))

        if episode_number == 20:
            begin = False
