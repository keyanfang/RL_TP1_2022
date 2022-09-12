#reference: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/2_Q_Learning_maze
#reference: https://blog.csdn.net/qq_41314151/article/details/100045143


import numpy as np
import pandas as pd
import os
import time

np.random.seed(1)

possibility=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
WORLD_R = 4
WORLD_C = 4
ACTIONS = ['up','down','left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 75
FRESH_TIME = 0.1
end_pos_x = 3





def build_q_table(world_r,world_c,actions):
    k = 0
    I = np.zeros([world_r * world_c,2],int)
    for i in range(world_r):
        for j in range(world_c):
            I[k,0] = i
            I[k,1] = j
            k+=1
    I = np.transpose(I).tolist()
    table = pd.DataFrame(np.zeros((world_r * world_c, len(actions))),index=I ,columns=actions)
    return table
def choose_action(pos_x,pos_y, q_table):
    pos_actions = q_table.loc[(pos_x,pos_y), :]
    if (np.random.rand() > EPSILON) or (max(pos_actions) == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = pos_actions.idxmax()
    return action_name

def get_env_feedback(pos_x, pos_y, Action, ):
    if Action == 'up':
        if pos_y ==0 and pos_x == end_pos_x:  # next move is final
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = 1
        elif pos_y == 1 and pos_x == end_pos_x:  # next move is final
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = -1
        elif pos_y == 0:  # cannot go up anymore
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        elif pos_y == 2 and pos_x ==1:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        else:
            next_pos_y = pos_y - 1  # middle
            next_pos_x = pos_x
            Reward = -0.04
    elif Action == 'down':
        if pos_y == WORLD_R - 1:  # cannot go down anymore
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        elif pos_x==1 and pos_y ==0:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        elif pos_y ==2:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = 0
        else:
            next_pos_y = pos_y + 1  # middle
            next_pos_x = pos_x
            Reward = -0.04
    elif Action == 'left':
        if pos_x == 0:  # cannot go left anymore
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        elif pos_x==2 and pos_y ==1:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        else:
            next_pos_x = pos_x - 1  # middle
            next_pos_y = pos_y
            Reward = -0.04
    elif Action == 'right':
        if pos_x == end_pos_x - 1 and pos_y == 0:  # next move is final
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = 1
        elif pos_x == end_pos_x - 1 and pos_y == 1:  # next move is final
            next_pos_x = 'end'
            next_pos_y = 'end'
            Reward = -1
        elif pos_x==0 and pos_y ==1:
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        elif pos_x == WORLD_C - 1:  # cannot go right anymore
            next_pos_x = pos_x
            next_pos_y = pos_y
            Reward = -0.04
        else:
            next_pos_x = pos_x + 1  # 中间
            next_pos_y = pos_y
            Reward = -0.04
    return next_pos_x, next_pos_y, Reward

def update_env(pos_x,pos_y, episode, step_counter):
    if pos_x == 'end' and pos_y == 'end':
        os.system("cls")
        print('Episode %s: total_steps = %s' % (episode + 1, step_counter))
        time.sleep(2)
    else:
        os.system("cls")
        for i in range(WORLD_R):
            env = ['+'] * (WORLD_C)
            if i == 1:
                env = ['+'] + ['x'] + ['+'] + ['-']
            if i == 0:
                env[end_pos_x] = 'O'
            if i == pos_y:
                env[pos_x] = '@'
            if i == 3:
                env = [' ']*(WORLD_C)
            a = ''.join(env)

            print('{}'.format(a))
        time.sleep(FRESH_TIME)
def RL():
    q_table = build_q_table(WORLD_R,WORLD_C,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        pos_x = 0
        pos_y = 2
        is_terminated = False
        update_env(pos_x,pos_y, episode, step_counter)
        while not is_terminated:
            Action = choose_action(pos_x,pos_y, q_table)
            next_pos_x,next_pos_y, Reward = get_env_feedback(pos_x,pos_y, Action)
            q_predict = q_table.loc[(pos_x,pos_y),Action]
            if next_pos_x != 'end' and next_pos_y != 'end':
                q_target = Reward + GAMMA * q_table.loc[(next_pos_x,next_pos_y), :].max()
            else:
                q_target = Reward
                is_terminated = True
            q_table.loc[(pos_x,pos_y), Action] += ALPHA * (q_target - q_predict)
            pos_x = next_pos_x
            pos_y = next_pos_y
            update_env(pos_x,pos_y, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    print("EPSILON=", EPSILON, "ALPHA=", ALPHA, "GAMMA=", GAMMA)
    q_table = RL()
    os.system("cls")
    print('Q-table:')
    print(q_table)
