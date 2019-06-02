import numpy as np
import time
import sys

def get_table(states, actions):
		table = np.zeros([len(states), len(actions)])
		return table

gamma = 0.8
alpha = 0.6
epsilon = 0.1

states = list(range(10))
states[-1] = 'terminal'
actions = [0, 1] # 0 for Left and 1 for Right

table = get_table(states, actions)

def get_reward(action, state):
    if state == 'terminal':
        return None, 'terminal'
    elif state >= 10:
        return None, 'terminal'
    elif state < 0:
        return None, 'terminal'
    elif action == 0:
        return -1, state - 1
    elif action == 1:
        if state + 1 >= 10:
            return 1, 'terminal'
        else:
            return 1, state + 1

def render(state):
    possible_states = ['_' for i in range(10)]
    possible_states[state] = 'X'
    string = ''
    for i in range(len(possible_states)):
        string += '{:^2}'.format(possible_states[i])
    return string

total_rewards = []
for i in range(1000):
    current_state = 0
    action = None
    current_rewards = []
    steps = 0
    while True:
        message = "Step {:<3}".format(steps)
        random = np.abs(np.random.randn())
        if random < epsilon: # random move
            action = np.random.choice(actions)
        else:
            action = np.argmax(table[current_state])
        
        reward, next_state = get_reward(action, current_state)
        current_rewards.append(reward)
        if reward != None:
            current = table[current_state, action]
            update = gamma * table[current_state, np.argmax(table[current_state])]
            new_q = current + (alpha * (reward + update - current))
            table[current_state, action] = new_q

            message += '{}'.format(render(current_state))
            print (message, end="\r", flush=True)

            current_state = next_state

            if next_state == 'terminal':
                print ('Goal reached. Steps Taken: {}'.format(steps))
                break
            else:
                steps += 1
        else:
            break
    total_rewards.append(current_rewards)

print (table)