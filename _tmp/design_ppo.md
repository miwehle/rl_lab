# PPO im rl_lab

## Ausgangspunkt

PPO soll in `rl_lab` in mindestens zwei Stufen gelernt und aufgebaut werden. Das Thema ist komplex genug, dass ein direkter Sprung zur Quadrocopter-Steuerung zu viel Architektur, zu viele neue Abstraktionen und zu wenig anschauliches Feedback auf einmal erzeugen wuerde.

Die geplante Linie ist deshalb:

1. `ppo`: eine kleine, lesbare PPO-Implementierung fuer kontinuierliche Aktionen im SolarSystemLander.
2. `ppo_2`: ein spaeteres Folgeprojekt als moegliche Basis fuer echte Quadrocopter-Steuerung, insbesondere automatisiertes Landen.
3. `ppo_3`: ein moeglicher SB3-Pfad fuer schnelle robuste PPO-Baselines auf Basis von Stable-Baselines3.

KISS/YAGNI bleiben Hauptkriterien. Das erste Ziel ist nicht, sofort eine allgemeine RL-Plattform zu bauen, sondern PPO praktisch zu verstehen, Continuous Actions im bestehenden SSL-Kontext zu beherrschen und das vorhandene HPO-System wiederzuverwenden.

## Lektion 1: ppo

`ppo` ist die erste PPO-Stufe. Es soll ein eigenes kleines Projekt werden, analog zur Rolle von `dqn`, aber bewusst als Lern- und Versuchsstand geschnitten.

Ziel von `ppo`:

- PPO als Algorithmus sichtbar und nachvollziehbar implementieren.
- Continuous Actions mit `LunarLander-v3(continuous=True)` und danach mit dem SolarSystemLander trainieren.
- Den bestehenden SSL als Simulation zum Fliegen bringen.
- Frueh anschauliches Feedback ueber Lernkurven, Evaluation und Videos bekommen.
- Das vorhandene HPO-Projekt fuer PPO nutzbar machen, ohne HPO zu duplizieren.

### Basis

`ppo` soll auf CleanRLs `ppo_continuous_action.py` als Hauptvorlage aufbauen.

Grund:

- CleanRL bildet den PPO-Algorithmus aus dem Original-Paper sehr direkt ab.
- Die Hauptschleifen bleiben sichtbar: Rollout sammeln, Advantages berechnen, mehrere Optimierungs-Epochen, Minibatches, clipped surrogate loss.
- Es nutzt direkt Gymnasium- und PyTorch-nahe Konzepte.
- Es passt gut zum bestehenden `rl_lab`, weil der aktuelle SSL ebenfalls Gymnasium-Environments verwendet.

CleanRL bezeichnet sich selbst zwar als Library, ist hier aber nicht wirklich eine modulare Library mit einer stabilen importierbaren Trainings-API. CleanRL liefert vor allem gute, kompakte Single-File-Implementierungen. `ppo` soll deshalb aus `ppo_continuous_action.py` eigene kleine Module machen, aehnlich wie `dqn` aus dem PyTorch-DQN-Tutorial entstanden ist.

Andere CleanRL-Quellen, zum Beispiel `ppo.py` fuer diskrete Actions oder die CleanRL-Dokumentation, koennen zum Vergleich oder zur Einordnung nuetzlich sein. Sie sind aber vorerst nicht als Implementierungsbasis geplant.

Die Implementierung in `ppo` soll klein, lokal verstaendlich und an die bestehenden rl_lab-Konventionen angepasst sein. Dabei geht es nicht um blindes Duplizieren von CleanRL-Code, sondern um eine bewusst modularisierte rl_lab-Version der fuer SSL-continuous benoetigten Teile.

Das PyTorch-PPO-Tutorial ist fuer diese erste Stufe eher Konzept-Referenz als Implementierungsbasis. Es verwendet TorchRL und bringt damit viele zusaetzliche Abstraktionen mit, zum Beispiel Env-Transforms, TensorDict, Collector, ReplayBuffer, GAE-Modul und ClipPPOLoss-Modul. Diese Abstraktionen koennen spaeter wertvoll sein, sind fuer den ersten Einstieg aber eine groessere Lernkurve als noetig.

### Projektform

Vorgeschlagene Struktur:

```text
ppo/
  src/ppo/
    model.py
    training.py
    evaluation.py
  tests/
  README.md
```

Die Struktur soll klein starten. Weitere Module sollen nur entstehen, wenn sie aktuelle Komplexitaet reduzieren.

### Erste fachliche Schritte

1. PPO fuer `LunarLander-v3(continuous=True)` zum Lernen bringen.
2. Den bestehenden `SolarSystemLander` continuous-faehig machen.
3. PPO auf SSL-continuous trainieren.
4. Deterministische Evaluation und Videos fuer trainierte Policies ergaenzen.
5. HPO minimal an PPO anbinden.

Der SSL kann voraussichtlich ueber die bestehende EnvFactory-Linie erweitert werden, weil Gymnasium `LunarLander-v3` diskrete und kontinuierliche Aktionen unterstuetzt. Fuer continuous Actions wird `continuous=True` genutzt. Die Action Space ist dann eine Box statt einer diskreten Action Space.

### HPO-Anbindung

Das bestehende HPO-Projekt soll fuer `ppo` wiederverwendet werden. HPO darf nicht fuer PPO neu implementiert werden.

Die Designregel:

- HPO bleibt die Studien-, Optuna-, Dashboard-, Checkpoint- und Reporting-Schicht.
- DQN und PPO liefern algorithmusspezifische Trainings- und Evaluationsbausteine.

Wahrscheinlicher kleinster Schritt:

- Entweder ein eigenes `hpo`-Modul fuer PPO-Objectives, zum Beispiel `hpo/src/hpo/ppo_objective.py`.
- Oder eine kleine Entkopplung des bestehenden `hpo.objective`, falls das einfacher bleibt.

Der aktuelle DQN-Pfad ist stark auf `VectorTrainer`, `VectorTrainingConfig`, `q_net` und greedy Q-Net-Evaluation bezogen. Fuer PPO braucht es stattdessen:

- PPO-TrainingConfig.
- PPO-Trainer.
- Actor/Policy fuer kontinuierliche Aktionen.
- Deterministische Policy-Evaluation fuer continuous Actions.
- PPO-spezifische Hyperparameter.

Die gemeinsame HPO-Idee bleibt:

```text
StudyRunner / Optuna / Dashboard
  -> Objective
  -> Trainer
  -> Evaluation
  -> Trial attrs / Checkpoints / Reporting
```

Die algorithmusspezifische Differenz bleibt bewusst am Rand:

```text
DQN:
  Trainer -> q_net -> greedy Q evaluation

PPO:
  Trainer -> actor/policy -> deterministic continuous-action evaluation
```

Keine grosse `rl_core`-Bibliothek vorab bauen. Erst wenn DQN und PPO nebeneinander leben, ist sichtbar, welche gemeinsame Schnittstelle wirklich verdient ist.

### Was ppo nicht sein soll

`ppo` soll nicht direkt die finale Quadrocopter-Architektur sein.

Nicht jetzt:

- TorchRL als Pflichtbasis.
- Isaac/OmniDrones-Integration.
- Eine allgemeine RL-Framework-Schicht.
- Direkte Motor-/PWM-Steuerung fuer echte Hardware.
- Spekulative Adapter fuer spaetere Simulatoren.

`ppo` ist ein Fluglabor fuer PPO, Continuous Actions, SSL und HPO.

## Lektion 2: ppo_2

`ppo_2` ist das spaetere Folgeprojekt nach `ppo`.

Es beginnt erst, wenn PPO durch `ppo` praktisch verstanden ist und SSL-continuous laeuft. Ziel ist dann eine realistischere Basis fuer echte Quadrocopter-Steuerung, zuerst vermutlich fuer automatisiertes Landen.

Fuer `ppo_2` ist TorchRL als Basis deutlich plausibler als fuer `ppo`, weil TorchRL eine echte modulare Library ist und der echte Quadrocopter-Anwendungsfall mehr Infrastruktur braucht:

- saubere Specs fuer Observation und Actions,
- Observation- und Reward-Transforms,
- Normalisierung und Persistenz der Normalisierungsstatistiken,
- parallele Simulation,
- Simulatorwechsel,
- Domain Randomization,
- Continuous-Control-Policies,
- spaetere Integration mit Robotik-Simulationen wie Isaac Lab oder OmniDrones.

Der Punkt ist nicht, dass TorchRL PPO einfacher macht. CleanRL zeigt PPO einfacher und direkter. TorchRL kann spaeter wertvoll sein, weil es die umgebende Robotik- und Simulationsinfrastruktur besser traegt. Anders als bei CleanRL soll `ppo_2` TorchRL wirklich als Library nutzen, nicht TorchRL-Module lokal nachbauen.

Fuer echten Quadrocopter-Betrieb ist ausserdem eine sicherere Steuerungsarchitektur wahrscheinlich:

- PPO steuert nicht direkt Motor-PWM.
- PPO gibt begrenzte Lande-Kommandos oder Setpoints aus.
- Ein bewahrter Low-Level-Controller stabilisiert darunter den Copter.
- Safety Layer, harte Action-Grenzen und Fallback-Verhalten sind Pflicht.

`ppo_2` bleibt deshalb bewusst grob geplant. Seine konkrete Form soll erst aus den Erfahrungen von `ppo` entstehen.

## Moegliche dritte Linie: ppo_3

`ppo_3` ist ein moeglicher spaeterer oder paralleler Pfad auf Basis von Stable-Baselines3, kurz SB3.

SB3 ist eine echte RL-Library mit einer bewaehrten PPO-Implementierung. In diesem Pfad wuerde PPO nicht selbst implementiert oder aus CleanRL modularisiert, sondern als Library genutzt:

```python
from stable_baselines3 import PPO
```

Ziel von `ppo_3` waere nicht PPO-Verstehen, sondern schnell robuste PPO-Baselines fuer SSL-continuous und eventuell spaeter Quadrocopter-Landing zu bekommen.

Rolle von `ppo_3`:

- Leistungs-Benchmark fuer die eigene `ppo`-Implementierung.
- Pragmatische Alternative zu `ppo_2`, falls SB3 fuer die konkreten Gymnasium-kompatiblen Environments genuegt.
- Moeglicher schneller Weg zu guten Trainingsresultaten mit wenig eigenem PPO-Code.

Nicht geplant:

- SB3-Interna kopieren.
- RL Baselines3 Zoo als zweites HPO-System neben dem bestehenden `hpo` uebernehmen.
- `ppo_3` sofort als gleichwertiges grosses Projekt starten.

## Aktuelle Entscheidung

Jetzt starten mit:

```text
ppo = CleanRL-nahe PPO-Lern- und SSL-Plattform
```

Spaeter, nach belastbarer Erfahrung:

```text
ppo_2 = TorchRL-/Robotik-Basis fuer Quadrocopter-Landung
ppo_3 = SB3-basierte robuste PPO-Baseline oder pragmatische Alternative
```

Nicht jetzt:

```text
grosses gemeinsames RL-Framework
HPO duplizieren
Quadrocopter-Abstraktionen vor dem ersten continuous SSL
```
