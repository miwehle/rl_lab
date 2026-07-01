

`gamma` und `tau` sitzen beide im Herz der DQN-Lernmechanik, also bei TD-Target, TD-Error und Target-Netz-Update.

`gamma`: Wie weit schaut der Lander in die Zukunft?

Das TD-Target ist grob:

```
target = reward + gamma * best_future_value
```

Wenn `gamma` von `0.99` auf `0.995` geht, wirkt das klein. Aber über viele Schritte potenziert sich das:

```
0.99^100  ≈ 0.37
0.995^100 ≈ 0.61
0.99^200  ≈ 0.13
0.995^200 ≈ 0.37
```

Bei `gamma=0.995` sind spätere Rewards also viel stärker “noch hörbar”. Für Lander wichtig: Die gute Landung oder der Crash kommen oft spät. Ein kleines `gamma` macht den Lander kurzsichtiger.

`tau`: Wie schnell folgt das Target-Netz?

DQN hat zwei Netze:

```
q_net        lernt ständig
target_net   liefert stabilere Ziele
```

Bei Soft Update:

```
target_net = tau * q_net + (1 - tau) * target_net
```

`tau=0.005` heißt: Target-Netz bewegt sich pro Update 0.5 Prozent Richtung neues Q-Netz.  
`tau=0.002` heißt: nur 0.2 Prozent.

Klingt winzig, aber das passiert sehr oft. Bei vielen Updates macht das enormen Unterschied: `tau=0.005` folgt deutlich schneller, `tau=0.002` ist träger und stabiler.

Warum das bei uns wichtig ist:

Der SSL fliegt mehrere Welten. Das Q-Net lernt ständig neue, teils widersprüchliche Erfahrungen. Wenn das Target-Netz zu schnell folgt, jagt das Lernziel hinter jedem Gewackel her. Dann kann Lernen instabil werden. Kleineres `tau` macht das Ziel ruhiger.

Kurz:

```
gamma = wie weit Zukunft zählt
tau   = wie ruhig das Lernziel bleibt
```

Und beide wirken nicht einmalig, sondern tausende Male in der TD-Lernschleife. Deshalb ist die dritte Nachkommastelle hier kein Schmuck, sondern Fahrwerksgeometrie.