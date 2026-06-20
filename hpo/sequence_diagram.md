```mermaid
%%{init: {"sequence": {"mirrorActors": false}}}%%
sequenceDiagram
    participant notebook as HPO notebook
    participant runner as StudyRunner
    participant study_module as study.py
    participant study as study : Study
    participant objective as objective.py
    participant trial as trial : Trial
    participant trainer as VectorTrainer

    notebook->>runner: StudyRunner(...)
    activate runner
    deactivate runner
    notebook->>runner: run(name, search_space, ...)
    activate runner
    runner->>study_module: run_study(...)
    activate study_module
    study_module->>objective: create_objective(...)
    activate objective
    deactivate objective

    study_module->>study: optimize(objective, n_trials=1)
    activate study
    study->>objective: objective(trial)
    activate objective
    objective->>trainer: VectorTrainer(...)
    activate trainer
    deactivate trainer
    objective->>trainer: train(training_config)
    activate trainer
    trainer-->>objective: training result
    deactivate trainer
    objective->>objective: evaluate_greedy_q_net(...)
    objective->>trial: set_user_attr(...)
    activate trial
    deactivate trial
    objective-->>study: objective score
    deactivate objective
    deactivate study
    deactivate study_module

    runner->>study_module: select_robust_best(...)
    activate study_module
    study_module->>objective: create_objective(...)
    activate objective
    deactivate objective
    study_module->>objective: objective(fixed_trial)
    activate objective
    deactivate objective
    study_module->>study: set_user_attr("robust_best_*", ...)
    activate study
    deactivate study
    deactivate study_module

    runner-->>notebook: study and selected parameters
    deactivate runner

```
