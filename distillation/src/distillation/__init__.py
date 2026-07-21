"""Small teacher-student distillation API for Elise-like DQN pilots."""

from distillation.dataset import DatasetRef, collect_teacher_dataset, load_dataset, save_dataset
from distillation.evaluate import evaluate_student, evaluate_teacher
from distillation.infra_cfg import InfraCfg
from distillation.plots import plot_score_gaps, plot_score_quantiles
from distillation.train import StudentRef, train_student

__all__ = [
    "DatasetRef",
    "InfraCfg",
    "StudentRef",
    "collect_teacher_dataset",
    "evaluate_student",
    "evaluate_teacher",
    "load_dataset",
    "plot_score_gaps",
    "plot_score_quantiles",
    "save_dataset",
    "train_student",
]
