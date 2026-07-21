from distillation.plots import plot_score_gaps, plot_score_quantiles


def test_plot_score_quantiles_compares_worlds_and_all_scores():
    teacher = _summary([("moon", 100), ("moon", 120), ("earth", 80), ("earth", 90)])
    student = _summary([("moon", 95), ("moon", 115), ("earth", 70), ("earth", 75)])

    figure = plot_score_quantiles(teacher, student)

    assert figure.layout.yaxis.ticktext == ("moon", "earth", "all")
    assert len(figure.data) == 8


def test_plot_score_gaps_compares_worlds_and_all_scores():
    teacher = _summary([("moon", 100), ("moon", 120), ("earth", 80), ("earth", 90)])
    student = _summary([("moon", 95), ("moon", 115), ("earth", 70), ("earth", 75)])

    figure = plot_score_gaps(teacher, student)

    assert tuple(figure.data[0].x) == ("moon", "earth", "all")
    assert tuple(figure.data[0].y) == (-5.0, -12.5, -8.75)


def _summary(scores):
    return {"rows": [{"world": world, "score": score} for world, score in scores]}
