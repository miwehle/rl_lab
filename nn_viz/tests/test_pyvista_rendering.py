import sys
from types import SimpleNamespace

import pytest

from nn_viz._pyvista_rendering import load_pyvista, render_layout_snapshot
from nn_viz.layout import Edge, NetworkLayout, Node


@pytest.fixture(autouse=True)
def clear_pyvista_cache():
    load_pyvista.cache_clear()
    yield
    load_pyvista.cache_clear()


def test_render_layout_snapshot_writes_offscreen_scene(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)
    output_path = tmp_path / "snapshot.png"

    result = render_layout_snapshot(_layout(), output_path, width=320, height=240, node_radius=0.2)

    plotter = fake_pyvista.plotters[0]
    assert result == output_path
    assert output_path.read_bytes() == b"fake png"
    assert plotter.off_screen is True
    assert plotter.window_size == (320, 240)
    assert plotter.background == "white"
    assert plotter.parallel_projection_enabled is True
    assert plotter.view == "isometric"
    assert plotter.reset_camera_called is True
    assert plotter.closed is True


def test_render_layout_snapshot_uses_node_z_and_edge_endpoints(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    render_layout_snapshot(_layout(), tmp_path / "snapshot.png")

    assert fake_pyvista.spheres == [
        {"radius": 0.055, "center": (0.0, 0.0, 0.0)},
        {"radius": 0.055, "center": (1.0, 2.0, 3.0)},
    ]
    assert fake_pyvista.lines == [((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))]


def _layout() -> NetworkLayout:
    return NetworkLayout(
        nodes=(
            Node("in", 0, "x", 0.0, 0.0, 0.1, z=0.0),
            Node("out", 1, "left", 1.0, 2.0, 0.2, z=3.0),
        ),
        edges=(Edge("in", 0, "out", 1, 0.5, 0.5, 0.5),),
    )


class FakePyVista:
    def __init__(self):
        self.plotters = []
        self.spheres = []
        self.lines = []

    def Plotter(self, *, off_screen, window_size):
        plotter = FakePlotter(off_screen, tuple(window_size))
        self.plotters.append(plotter)
        return plotter

    def Sphere(self, *, radius, center):
        sphere = {"radius": radius, "center": center}
        self.spheres.append(sphere)
        return SimpleNamespace(kind="sphere", **sphere)

    def Line(self, pointa, pointb):
        line = (pointa, pointb)
        self.lines.append(line)
        return SimpleNamespace(kind="line", pointa=pointa, pointb=pointb)


class FakePlotter:
    def __init__(self, off_screen, window_size):
        self.off_screen = off_screen
        self.window_size = window_size
        self.meshes = []
        self.closed = False
        self.parallel_projection_enabled = False
        self.reset_camera_called = False
        self.view = None

    def set_background(self, background):
        self.background = background

    def add_mesh(self, mesh, **kwargs):
        self.meshes.append((mesh, kwargs))

    def enable_parallel_projection(self):
        self.parallel_projection_enabled = True

    def view_isometric(self):
        self.view = "isometric"

    def reset_camera(self):
        self.reset_camera_called = True

    def screenshot(self, *, filename):
        with open(filename, "wb") as f:
            f.write(b"fake png")

    def close(self):
        self.closed = True
