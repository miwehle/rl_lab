import numpy as np
import torch

from nn_viz.ablation import evaluate_input_ablations


class AxSensitiveNet(torch.nn.Module):
    def eval(self):
        return self

    def forward(self, x):
        q = torch.zeros((x.shape[0], 4), dtype=torch.float32)
        q[:, 0] = 0.5
        q[:, 1] = x[:, 8]
        return q


class FakeEnv:
    def reset(self, *, seed=None):
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], dtype=np.float32), {}

    def step(self, action):
        return np.zeros(10, dtype=np.float32), float(action == 1), True, False, {}

    def close(self):
        pass


class FakeEnvFactory:
    def make_env(self, world):
        return FakeEnv()


def test_evaluate_input_ablations_scores_and_action_agreement():
    rows = evaluate_input_ablations(AxSensitiveNet(), FakeEnvFactory(), ["moon"], [7])
    by_ablation = {row["ablation"]: row for row in rows}

    assert by_ablation["normal"]["mean_score"] == 1.0
    assert by_ablation["ax=0"]["mean_score"] == 0.0
    assert by_ablation["ax=0"]["delta_vs_normal"] == -1.0
    assert by_ablation["ax=0"]["action_agreement"] == 0.0
    assert by_ablation["ay=0"]["mean_score"] == 1.0
    assert by_ablation["ay=0"]["action_agreement"] == 1.0
