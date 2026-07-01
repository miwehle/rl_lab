# HP-Bereiche

| HP              |      min |         max | Skala       | Erlaeuterung                                                                                                                              |
| --------------- | -------: | ----------: | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `learning_rate` |   `5e-4` |      `2e-3` | log         | Aktuell guter DQN/SSL-Bereich. Zu hoch kann gute Piloten wieder verschleifen; zu niedrig lernt zu langsam.                                |
| `gamma`         |   `0.99` |     `0.995` | categorical | Wirkt auf den Planungshorizont. `0.995` haelt spaete Landung/Crash-Rewards laenger relevant als `0.99`.                                   |
| `tau`           |  `0.002` |      `0.01` | categorical | Soft-Update des Target-Netzes. Kleinere Werte machen das Lernziel ruhiger; groessere Werte folgen schneller, koennen aber wackliger sein. |
| `eps_start`     |      `0` |       `1.0` | linear      | Exploration am Trainingsanfang. Oft `1.0` bei Training from scratch; niedriger bei Fine-Tuning oder vorhandener guter Policy.             |
| `eps_end`       |   `0.02` | oft: <`0.2` | linear      | Rest-Exploration am Trainingsende. Gute Earth/5-Welten-Kandidaten lagen bisher grob in diesem Bereich.                                    |
| `eps_decay`     | `25_000` |   `150_000` | log         | Steuert, wie schnell Exploration abfaellt. Muss zur Trainingslaenge und `num_envs` passen; gute Werte koennen stark vom Setup abhaengen.  |
