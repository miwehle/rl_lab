"""A KISS implementation of reinforcement learning.

It illustrates basic RL principles, especially Q-learning.

Task:
- world = regular grid with walls
- start at a position and learn to reach the goal position

Representation:
- states = positions, actions = moves
- reward: if goal is reached
"""

import numpy as np

# grid: (height, width)
grid = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
], dtype=bool)
height, width = grid.shape

# actions
num_actions = 4
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# q: (height, width, num_actions)
q = np.zeros((height, width, num_actions), dtype=float)
q[~grid] = np.nan

rng = np.random.default_rng(seed=42)

def move(state, action):
    y, x = state

    if action == UP:
        next_y, next_x = y - 1, x
    elif action == DOWN:
        next_y, next_x = y + 1, x
    elif action == LEFT:
        next_y, next_x = y, x - 1
    elif action == RIGHT:
        next_y, next_x = y, x + 1
    else:
        raise ValueError("invalid action")

    height, width = grid.shape

    if next_y < 0 or next_y >= height:
        return state

    if next_x < 0 or next_x >= width:
        return state

    if not grid[next_y, next_x]:
        return state

    return next_y, next_x

def choose_action(state, exploration_rate):
    if rng.random() < exploration_rate:
        return rng.integers(q.shape[2])

    action_values = q[state]
    best_value = np.nanmax(action_values)
    best_actions = np.flatnonzero(action_values == best_value)
    return rng.choice(best_actions)

def q_learning(
    start, goal,
    goal_reward,
    num_episodes, max_steps,
    lr, discount_factor, exploration_rate
):
    """
    cf.
    https://introml.mit.edu/notes/reinforcement_learning.html#sec-q_learning
    (https://web.stanford.edu/class/cs234/slides/lecture4pre.pdf, p. 35f.)
    """
    for _ in range(num_episodes):
        state = start

        for _ in range(max_steps):
            action = choose_action(state, exploration_rate)
            next_state = move(state, action)
            done = next_state == goal

            # core: Q-learning update rule
            reward = goal_reward if done else 0.0
            next_value = 0.0 if done else np.nanmax(q[next_state])
            target = reward + discount_factor * next_value
            error = target - q[state][action]
            q[state][action] += lr * error

            state = next_state

            if done:
                break

def print_policy(start, goal):
    symbols = {
        UP: "^",
        DOWN: "v",
        LEFT: "<",
        RIGHT: ">",
    }

    for y in range(height):
        row = []
        for x in range(width):
            if not grid[y, x]:
                row.append("#")
            elif (y, x) == start:
                row.append("S")
            elif (y, x) == goal:
                row.append("G")
            else:
                action_values = q[y, x]
                best_value = np.nanmax(action_values)
                best_actions = np.flatnonzero(action_values == best_value)

                if len(best_actions) > 1:
                    row.append("?")
                else:
                    row.append(symbols[best_actions[0]])
        print(" ".join(row))

start = (0, 0)
goal = (2, 2)

q_learning(
    start = start, goal = goal,
    goal_reward = 10,
    num_episodes = 20, max_steps = 50,
    lr=0.3, discount_factor=0.9, exploration_rate=0.2,
)

print_policy(start, goal)
