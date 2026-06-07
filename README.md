## RL concepts

[q_learning](q_learning/grid_world.py) and [dqn](dqn/src/dqn/training.py) both use the following concepts.

The *TD target* estimates what $Q(s,a)$ should be by looking one step ahead:

$$
y = r + \gamma \max_{a'} Q(s',a')
$$

It considers two points in time:
1. Now: If I do $a$, I get reward $r$
2. One step ahead: The best estimated value from the next state $s'$ is $\max_{a'} Q(s',a')$

$\gamma$ is the discount factor (controlling how much the one-step-ahead estimate matters).

The *TD error* compares the target and the current estimate:

$$
\delta = y - Q(s,a)
$$

The *Q-learning update* adjusts the current estimate toward the target:

$$
Q[s,a] \leftarrow Q[s,a] + \alpha \cdot \delta
$$

$\alpha$ is the learning rate (controlling how strongly the TD error updates the old estimate).

TD stands for temporal difference.

## References

[MIT Intro to ML](https://introml.mit.edu/notes/reinforcement_learning.html)
