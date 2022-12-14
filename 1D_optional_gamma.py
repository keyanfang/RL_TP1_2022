# reference: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/1_command_line_reinforcement_learning/treasure_on_right.py


import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ["left", "right"]
EPSILON = 0.5
ALPHA = 0.1
# GAMMA = 0.9
MAX_EPISODE = 10
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,
    )

    return table


def choose_action(state, table):
    actions = table.iloc[state, :]
    if np.random.uniform() > EPSILON or actions.all() == 0:
        action = np.random.choice(ACTIONS)
    else:
        action = actions.idxmax()
    return action


def env_feedback(S, A):
    if A == "right":
        if S == N_STATES - 1:
            S_ = "terminal"
            reward = 1
        else:
            S_ = S + 1
            reward = -0.01
    else:
        if S == 0:
            S_ = "terminal"
            reward = -1
        else:
            S_ = S - 1
            reward = -0.01
    return S_, reward


def print_env(S, episode, step_counter):
    env_list = ['X'] + ['-'] * (N_STATES - 2) + ['G']
    if S == 'terminal':
        print("\n")
        print('Episode %s: total_steps = %s' % (episode + 1, step_counter))
        # interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        # print('\r{}'.format(interaction), end='')
        # print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def q_learning(GAMMA):
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODE):
        is_terminated = False
        step_counter = 0
        S = 2
        print_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, r = env_feedback(S, A)
            q_predict = q_table.loc[S, A]
            if S_ == "terminal":
                q_target = r
                is_terminated = True
            else:
                q_target = r + GAMMA * q_table.iloc[S_, :].max()
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            step_counter += 1
            print_env(S, episode, step_counter)
    return q_table


def different_gamma_result():
    GAMMA = 0.1
    while GAMMA < 1:
        print("gamma=", GAMMA)
        q_table = q_learning(GAMMA)
        print(q_table)
        GAMMA = GAMMA + 0.1


if __name__ == "__main__":
    # q_table = q_learning()
    # print(q_table)
    different_gamma_result()

# gamma est un facteur qui prend en compte les r??compenses futures.
# Si le gamma est trop petit, la r??compense positive au point final ne peut pas "se propager",
# c'est-??-dire que le robot risque de ne pas apprendre une strat??gie pour atteindre le point final.
