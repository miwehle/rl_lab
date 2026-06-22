### Suchräume

{
    "learning_rate": 0.0022727854024196057,
    "batch_size": 512,
    "eps_end": 0.047716002108220544,
    "eps_decay": 43_214,
    "gamma": 0.99,
    "tau": 0.005,
    "learning_starts": 2_500,
    "optimize_every": 2,
    "replay_memory_capacity": 400_000,
    "num_episodes": 500,
}


| HP | S1 Flight Hours | S2 Exploration |
|---|---|---|
| num_episodes | categorical([500, 750, 1_000]) | incumbent |
| eps_end | incumbent | float(0.03, 0.07) |
| eps_decay | incumbent | int(30_000, 150_000, log=True) |
| alle anderen | incumbent | incumbent |
