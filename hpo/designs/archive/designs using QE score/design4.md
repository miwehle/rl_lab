## HPO Study Series 3 – SolarSystemLander 8D

| HP | S0 Incumbent | S1 Update-Ökonomie | S2 Exploration | S3 Replay-Kapazität | S4 Gemeinsame Feinsuche |
|---|---:|---|---|---|---|
| learning_rate | 0.00197885 | *float(incumbent * 0.75, incumbent * 1.25, log=True)* | incumbent | incumbent | *float(incumbent * 0.75, incumbent * 1.25, log=True)* |
| batch_size | 1_024 | *categorical([512, 1_024])* | incumbent | incumbent | *categorical(neighbors(incumbent, [256, 512, 1_024]))* |
| eps_end | 0.04522 | incumbent | *float(max(0.01, incumbent - 0.01), min(0.10, incumbent + 0.01))* | incumbent | *float(max(0.01, incumbent - 0.01), min(0.10, incumbent + 0.01))* |
| eps_decay | 52_496 | incumbent | *int(incumbent * 0.75, incumbent * 1.25, log=True)* | incumbent | *int(incumbent * 0.75, incumbent * 1.25, log=True)* |
| gamma | 0.99 | incumbent | incumbent | incumbent | incumbent |
| tau | 0.005 | incumbent | incumbent | incumbent | incumbent |
| learning_starts | 1_000 | *categorical([1_000, 2_500])* | incumbent | incumbent | *categorical(neighbors(incumbent, [1_000, 2_500, 5_000]))* |
| optimize_every | 4 | *categorical([2, 4, 8])* | incumbent | incumbent | *categorical(neighbors(incumbent, [2, 4, 8]))* |
| replay_memory_capacity | 400_000 | incumbent | incumbent | *categorical([200_000, 400_000, 800_000])* | incumbent |
| num_episodes | 500 | *categorical([500, 1_000])* | incumbent | incumbent | *categorical(neighbors(incumbent, [500, 1_000, 1_500]))* |

### Venus-Wetter und Evaluation

Die Diagramme unter **„3. Performance by world“** im Analyse-Notebook `hpo/notebooks/_archive/Analyze_SolarSystemLander.ipynb` zeigen, dass ==Venus für beide Lander besonders schwierig== war. Venus hat zugleich von allen Welten den stärksten Wind. Das ist ein starkes Indiz dafür, dass der extreme Wind ein wesentlicher Faktor ist und beim Training vermieden oder zumindest reduziert werden sollte. Die bisherigen Daten erlauben jedoch keine saubere Trennung von Wind und Turbulenz.

Eine spätere Diagnose kann pro Evaluations-Episode Welt, Windstärke, Turbulenzstärke und Return speichern, um die Wetterfaktoren getrennt zu untersuchen.

### Architektur-Befund

Die Diagramme unter **„3. Performance by world“** zeigen über alle fünf Studien ein ähnliches Muster: ==Die Lander funktionieren unter den mittleren Bedingungen von Merkur und Mars am besten==, während sie an den dynamischen Extremen deutlich schlechter abschneiden. Neben der stürmischen Venus betrifft dies insbesondere den physikalisch leichteren, aber vom mittleren Trainingsfall weit entfernten Mond.

Damit scheint die direkte Erweiterung vom LunarLander zum SolarSystemLander ohne Architekturänderung ein Irrweg zu sein: Eine einzige kleine Policy mit 256 Neuronen lernt offenbar vor allem eine Steuerung für mittlere Bedingungen und generalisiert nur unzureichend auf stark abweichende Dynamiken. Dieses Negativergebnis ist ein wesentliches Ergebnis der Study Series 2.

Für eine Folgestudie gilt die Hypothese: Bei gleichem gesamten Trainingsaufwand `E` erreichen fünf spezialisierte Policies – eine pro Welt – höhere Gym-Scores als eine gemeinsame Generalisten-Policy. Alternativ kann später eine kleinere Zahl von Spezialisten für ähnliche Dynamikklassen untersucht werden.

### Qualitätsgrenze ohne Aufwandstrafe

Die bisherigen Studien suchten eine eierlegende Wollmilchsau: hohe Landequalität bei geringem Trainingsaufwand (und das für fünf Welten). Dadurch ist noch offen, ob der Generalist grundsätzlich an seiner Kapazitätsgrenze scheitert oder ob der Quality-Effort Score lediglich zu wenig Training belohnt.

==Eine weitere Studienreihe soll deshalb ausschließlich den gemeinsamen Gym-Score maximieren==; Trainingsaufwand wird gemessen, aber nicht im Objective bestraft. Damit lässt sich untersuchen, welche Landequalität möglich ist, wenn Trainingskosten praktisch keine Rolle spielen. Erreicht auch dieser Generalist auf Mond, Erde und Venus keine hohen Scores, spricht das deutlich stärker für eine Architektur- beziehungsweise Kapazitätsgrenze.
