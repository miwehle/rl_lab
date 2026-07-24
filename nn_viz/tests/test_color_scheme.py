from nn_viz.color_scheme import alpha, edge_width, heat_color, signed_color


def test_alpha_uses_log_magnitude_and_clips():
    assert alpha(0.0, 10.0) == 0
    assert 0 < alpha(1.0, 10.0) < alpha(5.0, 10.0) < 255
    assert alpha(-10.0, 10.0) == 255
    assert alpha(100.0, 10.0) == 255
    assert alpha(1.0, 0.0) == 0


def test_signed_color_maps_sign_to_blue_gray_red():
    assert signed_color(0.0, 10.0) == (128, 128, 128)
    assert signed_color(10.0, 10.0) == (220, 38, 38)
    assert signed_color(-10.0, 10.0) == (37, 99, 235)

    weak_red = signed_color(1.0, 10.0)
    assert weak_red[0] > 128
    assert weak_red[1] < 128
    assert weak_red[2] < 128


def test_heat_color_uses_nonnegative_heat_scale():
    assert heat_color(-1.0, 10.0) == heat_color(0.0, 10.0)
    assert heat_color(10.0, 10.0) == (255, 255, 235)
    low = heat_color(1.0, 10.0)
    high = heat_color(5.0, 10.0)
    assert low != high


def test_edge_width_uses_log_magnitude_and_clips():
    assert edge_width(0.0, 10.0) == 1.0
    assert 1.0 < edge_width(1.0, 10.0) < edge_width(5.0, 10.0) < 3.0
    assert edge_width(-10.0, 10.0) == 3.0
    assert edge_width(100.0, 10.0) == 3.0
    assert edge_width(1.0, 0.0) == 1.0
