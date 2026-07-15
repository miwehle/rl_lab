"""Study orchestration API."""

from hpo.study.infra_cfg import InfraCfg

__all__ = ["Baseline", "InfraCfg", "StudyRunner", "run_study"]


def __getattr__(name: str):
    if name in {"Baseline", "StudyRunner", "run_study"}:
        from hpo.study.study_runner import Baseline, StudyRunner, run_study

        exports = {
            "Baseline": Baseline,
            "StudyRunner": StudyRunner,
            "run_study": run_study,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
