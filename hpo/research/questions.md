# HPO Research Questions

| Nr | Question | Topics |
|---|---|---|
| [[#Q2 Training Nutzen Bei Epsilon Unter 0.05\|Q2]] | Training Nutzen Bei Epsilon Unter 0.05 | RL, HP |
| [[#Q1 SolarSystemLander Difficulty\|Q1]] | SolarSystemLander Difficulty | SSL |

Topics: `RL` = Reinforcement Learning, `SSL` = SolarSystemLander, `HP` = Hyperparameters.


## Q3 Verpasst Elise manchmal ohne Not gute Scores?

1. Wird das Training manchmal zu früh abgebrochen (vom Trainer, vom Gym?), so dass der Reward 100 für die Landung nicht addiert wird? Oder ist nur das Video zu kurz, so dass der Lande-Reward nich gefilmt wird?
2. Manchmal hält Elise nach der Landung eine Seitendüse betätigt. Wieviel Score kostet das?

## Q2 Training Nutzen Bei Epsilon Unter 0.05

Lohnt sich bei `epsilon < 0.05` Weitertrainieren überhaupt noch?

Beispiel: Study "Go, Optuna, go", Trial 24. Da lohnte es sich.

## Q1 SolarSystemLander Difficulty

What makes Earth and Venus hard for the small SSL DQN?

Possible factors:
- higher gravity
- wind and turbulence
- interaction of gravity and weather

| Gravity | Weather off | Weather on |
| ------- | ----------- | ---------- |
| Earth   |             |            |
| Venus   |             |            |
