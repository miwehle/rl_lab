
# Package structure and LOC


hpo production total  3771
├─ hpo/\_\_init\_\_.py  5
└─ hpo/src/hpo/  *3776*
   ├─ evaluation/  *964*
   │  ├─ checkpoint_robustness.py  264
   │  ├─ hp_robustness.py  156
   │  ├─ lander_rendering.py  316
   │  └─ video.py  227
   ├─ notebook/  *1153*
   │  ├─ dashboard/  *787*
   │  │  ├─ main.py  244
   │  │  ├─ training_progress.py  180
   │  │  ├─ robustness.py  151
   │  │  ├─ current_study.py  85
   │  │  ├─ current_hps.py  52
   │  │  ├─ study_series.py  52
   │  │  ├─ style.py  18
   │  │  └─ \_\_init\_\_.py  5
   │  ├─ plots.py  174
   │  ├─ colab.py  143
   │  └─ optuna.py  49
   ├─ ==checkpointing.py==  395
   ├─ ==objective.py==  292
   ├─ ==study.py==  278
   ├─ solar_system_lander/  174
   │  └─ environment.py  173
   ├─ games/  165
   │  └─ solar_system_lander.py  164
   ├─ study_metadata.py  143
   ├─ study_reporting.py  91
   ├─ lunar_lander/  78
   │  ├─ logging.py  58
   │  └─ environment.py  19
   └─ hyperparams.py  15


*größte Pakete in hpo*
==fachlich wichtigste Module==