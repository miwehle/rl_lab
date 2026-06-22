# ----------------------------------------
# setup

!git clone https://github.com/miwehle/rl_lab.git
%cd rl_lab
!pip install -r hpo/requirements.txt

from pathlib import Path
import sys
from unittest.mock import Mock

import torch

sys.path.insert(0, "dqn/src")
sys.path.insert(0, "hpo/src")

from dqn.vector_training import VectorTrainingConfig
from hpo.evaluation.scoring import ScoringConfig
from hpo.lunar_lander.logging import configure_file_logging
from hpo.objective import TrialConfig
from hpo.solar_system_lander.environment import EnvFactory
from hpo.study import StudyRunner


LOCAL_DIR = Path("/content/rl_lab/hpo/runs")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_NAME = "kiss.log"
LOCAL_LOG = LOCAL_DIR / LOG_NAME
configure_file_logging(LOCAL_DIR, LOG_NAME)


# ----------------------------------------
# search space

class SearchSpace:
    def training_config(self, trial, _params):
        return VectorTrainingConfig(
            num_episodes=1_000,
            batch_size=512,
            gamma=0.99,
            eps_start=1.0,
            eps_end=trial.suggest_float("eps_end", 0.03, 0.08),
            eps_decay=trial.suggest_int(
                "eps_decay", 43_000, 150_000, log=True
            ),
            tau=0.005,
            learning_rate=0.0022727854024196057,
            learning_starts=2_500,
            optimize_every=2,
        )

    def replay_memory_capacity(self, _trial, _params):
        return 1_200_000

# ----------------------------------------
# run

env_factory = EnvFactory("8d")
study_runner = StudyRunner(
    database_path=lambda _name: LOCAL_DIR / "kiss.db",
    environment_factory=env_factory,
    trial_cfg=TrialConfig(
        num_envs=20,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ),
    incumbent_params={},
    reporter=Mock(),
    study_attrs=env_factory.metadata(),
    robust_candidates=1,
    extra_seeds=(),
)


scoring_cfg = ScoringConfig(
    quality_weight=0.95,
    eval_episodes=10,
    baseline_env_steps=112_314.33333333333,
    baseline_processed_samples=28_491_093.333333332,
)
study_runner.run("kiss_study", SearchSpace(), 25, scoring_cfg)


# ----------------------------------------
# copy db to Google drive

from google.colab import drive
from hpo.drive_backup import backup_to_drive

drive.mount("/content/drive")
DRIVE_DIR = Path("/content/drive/MyDrive/rl_lab/hpo")
backup_to_drive(
    local_database=LOCAL_DIR / "kiss.db",
    drive_database=DRIVE_DIR / "kiss.db",
    local_log=LOCAL_LOG,
    drive_log=DRIVE_DIR / LOG_NAME,
)
