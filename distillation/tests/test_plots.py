from distillation.plots import plot_score_gaps, plot_score_quantiles, score_comparison_table


def test_score_comparison_table_sorts_large_teacher_advantages_first():
    teacher = _summary([("moon", 7, 120), ("earth", 42, 80), ("venus", 7, 50)])
    student = _summary([("moon", 7, 100), ("earth", 42, 75), ("venus", 7, 70)])

    table = score_comparison_table(teacher, student, min_diff=10.0)

    assert table.to_dict("records") == [
        {
            "world": "moon",
            "seed": 7,
            "teacher_score": 120.0,
            "student_score": 100.0,
            "teacher_minus_student": 20.0,
        },
        {
            "world": "venus",
            "seed": 7,
            "teacher_score": 50.0,
            "student_score": 70.0,
            "teacher_minus_student": -20.0,
        },
    ]


def test_score_comparison_table_can_sort_student_advantages_first():
    teacher = _summary([("moon", 7, 120), ("venus", 7, 50)])
    student = _summary([("moon", 7, 100), ("venus", 7, 70)])

    table = score_comparison_table(teacher, student, ascending=True)

    assert table["world"].tolist() == ["venus", "moon"]


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
    return {
        "rows": [
            {"world": row[0], "seed": row[1] if len(row) == 3 else index, "score": row[-1]}
            for index, row in enumerate(scores)
        ]
    }
