# Package structure and LOC

hpo production total  3794
├─ hpo/__init__.py  8
└─ hpo/src/hpo/  *3786*
   ├─ evaluation/  *964*
   │  ├─ __init__.py  1
   │  ├─ checkpoint_robustness.py  264
   │  ├─ hp_robustness.py  156
   │  ├─ lander_rendering.py  316
   │  └─ video.py  227
   ├─ games/  165
   │  ├─ __init__.py  1
   │  └─ solar_system_lander.py  164
   ├─ lunar_lander/  78
   │  ├─ __init__.py  1
   │  ├─ environment.py  19
   │  └─ logging.py  58
   ├─ notebook/  *1085*
   │  ├─ dashboard/  *718*
   │  │  ├─ __init__.py  1
   │  │  ├─ current_hps.py  52
   │  │  ├─ current_study.py  85
   │  │  ├─ main.py  228
   │  │  ├─ robustness.py  154
   │  │  ├─ style.py  18
   │  │  └─ training_progress.py  180
   │  ├─ __init__.py  1
   │  ├─ colab.py  143
   │  ├─ optuna.py  49
   │  └─ plots.py  174
   ├─ solar_system_lander/  242
   │  ├─ __init__.py  1
   │  ├─ environment.py  201
   │  └─ reward_shaping.py  40
   ├─ __init__.py  7
   ├─ _api.py  45
   ├─ ==checkpointing.py==  395
   ├─ hyperparams.py  15
   ├─ ==objective.py==  292
   ├─ study_metadata.py  143
   ├─ study_reporting.py  91
   └─ ==study.py==  264

*größte Pakete in hpo*
==fachlich wichtigste Module==
