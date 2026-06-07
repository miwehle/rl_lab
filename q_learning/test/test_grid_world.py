import grid_world


EXPECTED_POLICY = (
    "S ? ? ? ?\n"
    "v # # # v\n"
    "v # G # v\n"
    "> > ^ < <\n"
    "? ^ ? ^ ?"
)


def test_q_learning():
    start = (0, 0)
    goal = (2, 2)

    grid_world.q_learning(
        start=start, goal=goal,
        goal_reward=10,
        num_episodes=20, max_steps=50,
        alpha=0.3, gamma=0.9, epsilon=0.2
    )

    policy = grid_world.policy_as_string(start, goal)
    #print("\n" + policy)
    assert policy == EXPECTED_POLICY
