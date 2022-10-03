import gym
import numpy as np
import time, pickle, os

env = gym.make('FrozenLake-v1', render_mode='human',desc=None, map_name="4x4", is_slippery=True)

epsilon = 0.9
total_episodes = 10000
max_steps = 1000

lr_rate = 0.1
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learnWithQSARSA(state, state2, reward, action, action2):#SARSA
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)


# Start
for episode in range(total_episodes):
    state = env.reset()
    t = 0
    #print("Episode ",episode)   
 
    while t < max_steps:
        env.render()
        action = choose_action(state)  
        state2, reward, done, info = env.step(action)

        action2 = choose_action(state2)#SARSA

        learnWithQSARSA(state, state2, reward, action, action2)
        state = state2

        action = action2#SARSA

        t += 1
        if done:
            break
        #time.sleep(0.1)

print(Q)

with open("frozenLake_qTable.pkl", 'wb') as f:
    pickle.dump(Q, f)

