from hpo.notebook_utils import neighbors


def test_neighbors_returns_value_plus_direct_neighbors() -> None:
    assert neighbors(10_000, [2_500, 5_000, 10_000, 20_000]) == [
        5_000,
        10_000,
        20_000,
    ]
