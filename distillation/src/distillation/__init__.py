"""Small teacher-student distillation API for Elise-like DQN pilots."""

from distillation.dataset import DatasetRef, collect_teacher_dataset, load_dataset, save_dataset
from distillation.evaluate import evaluate_student
from distillation.infra import InfraCfg
from distillation.models import StudentDQN
from distillation.train import StudentRef, train_student

__all__ = [
    "DatasetRef",
    "InfraCfg",
    "StudentDQN",
    "StudentRef",
    "collect_teacher_dataset",
    "evaluate_student",
    "load_dataset",
    "save_dataset",
    "train_student",
]
