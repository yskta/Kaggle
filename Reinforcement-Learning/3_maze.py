import numpy as np
import matplotlib.pyplot as plt

# パラメータの設定
epsilon = 0.1
gamma = 0.9
alpha = 0.1
num_episodes = 1

# 迷路の設定
maze = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

start = (1, 1)
goal = (7, 8)
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 価値関数の初期化
V = np.zeros(maze.shape)

def get_possible_actions(state):
    possible_actions = []
    for action in actions:
        next_state = (state[0] + action[0], state[1] + action[1])
        if 0 <= next_state[0] < maze.shape[0] and 0 <= next_state[1] < maze.shape[1] and maze[next_state] == 1:
            possible_actions.append(action)
    return possible_actions

def epsilon_greedy_policy(state):
    possible_actions = get_possible_actions(state)
    if not possible_actions:
        return None
    if np.random.rand() < epsilon:
        return possible_actions[np.random.choice(len(possible_actions))]
    else:
        next_values = []
        for action in possible_actions:
            next_state = (state[0] + action[0], state[1] + action[1])
            next_values.append(V[next_state])
        return possible_actions[np.argmax(next_values)]

episode_lengths = []

for episode in range(num_episodes):
    state = start
    steps = 0
    while state != goal:
        action = epsilon_greedy_policy(state)
        if action is None:
            break
        next_state = (state[0] + action[0], state[1] + action[1])
        reward = 1 if next_state == goal else 0
        V[state] += alpha * (reward + gamma * V[next_state] - V[state])
        state = next_state
        steps += 1
    episode_lengths.append(steps)

# 学習曲線のプロット
plt.plot(episode_lengths)
plt.xlabel('Episode')
plt.ylabel('Steps to Goal')
plt.title('Learning Curve')
plt.show()