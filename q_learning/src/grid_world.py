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

_rng = np.random.default_rng(seed=42)

def reset(seed=42):
    """Reset the Q-table and random number generator."""
    global _rng

    q[:] = 0.0
    q[~grid] = np.nan
    _rng = np.random.default_rng(seed=seed)

def _move(y, x, action):
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

    is_outside_grid = (
        next_y < 0 or next_y >= height
        or next_x < 0 or next_x >= width
    )

    if is_outside_grid:
        return (y, x)

    if not grid[next_y, next_x]:
        return (y, x)

    return (next_y, next_x)

def _step(state, action, goal, goal_reward):
    y, x = state
    next_state = _move(y, x, action)
    done = next_state == goal
    reward = goal_reward if done else 0.0
    return next_state, reward, done

def _select_action(state, epsilon):
    if _rng.random() < epsilon:
        return _rng.integers(q.shape[2])

    action_values = q[state]
    best_value = np.nanmax(action_values)
    best_actions = np.flatnonzero(action_values == best_value)
    return _rng.choice(best_actions)

def q_learning(
    start, goal,
    goal_reward,
    num_episodes, max_steps,
    alpha, gamma, epsilon
):
    """
    Args:
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration rate

    References:
        https://introml.mit.edu/notes/reinforcement_learning.html#sec-q_learning
        (https://web.stanford.edu/class/cs234/slides/lecture4pre.pdf, p. 35f.)
    """
    for _ in range(num_episodes):
        state = start

        for _ in range(max_steps):
            action = _select_action(state, epsilon)
            next_state, reward, done = _step(state, action, goal, goal_reward)

            # core: Q-learning update rule (cf. 
            next_q_value = 0.0 if done else np.nanmax(q[next_state])
            td_target = reward + gamma * next_q_value
            td_error = td_target - q[state][action]
            q[state][action] += alpha * td_error

            state = next_state

            if done:
                break

def policy_as_string(start, goal):
    symbols = {
        UP: "^",
        DOWN: "v",
        LEFT: "<",
        RIGHT: ">",
    }

    lines = []
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
        lines.append(" ".join(row))
    return "\n".join(lines)
