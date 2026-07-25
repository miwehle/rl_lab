import sys
from types import SimpleNamespace

import numpy as np
import pytest

from nn_viz._pyvista_rendering import load_pyvista, render_layout_snapshot, render_state_snapshot
from nn_viz._rendering import NetworkState
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

    render_layout_snapshot(_layout(), tmp_path / "snapshot.png", edge_geometry="line")

    assert fake_pyvista.spheres == [
        {"radius": 0.055, "center": (0.0, 0.0, 0.0)},
        {"radius": 0.055, "center": (1.0, 2.0, 3.0)},
    ]
    assert fake_pyvista.lines == [((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))]


def test_render_layout_snapshot_defaults_to_tubes_with_weighted_edge_width(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    render_layout_snapshot(_layout(), tmp_path / "snapshot.png")

    assert fake_pyvista.tubes == [{"radius": pytest.approx(0.026), "n_sides": 8}]
    edge_mesh, edge_kwargs = fake_pyvista.plotters[0].meshes[0]
    assert edge_mesh.kind == "tube"
    assert edge_kwargs == {"color": "#dc2626", "smooth_shading": True}


def test_render_layout_snapshot_can_use_thin_lines(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    render_layout_snapshot(_layout(), tmp_path / "snapshot.png", edge_geometry="line")

    edge_mesh, edge_kwargs = fake_pyvista.plotters[0].meshes[0]
    assert edge_mesh.kind == "line"
    assert edge_kwargs == {"color": "#dc2626", "line_width": 3}


def test_render_layout_snapshot_rejects_unknown_edge_geometry(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    with pytest.raises(ValueError, match="edge_geometry"):
        render_layout_snapshot(_layout(), tmp_path / "snapshot.png", edge_geometry="weird")


def test_render_state_snapshot_colors_edges_by_current_contribution(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    render_state_snapshot(
        _layout(),
        _state(input_value=-1.0),
        tmp_path / "snapshot.png",
        edge_geometry="line",
        scales={"input": [1.0], "output": 1.0, "activation": 1.0, "weight": 0.5},
    )

    edge_mesh, edge_kwargs = fake_pyvista.plotters[0].meshes[0]
    assert edge_mesh.kind == "line"
    assert edge_kwargs == {"color": "#2563eb", "line_width": 3, "opacity": 1.0}


def test_render_state_snapshot_can_map_edge_intensity_to_opacity(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    render_state_snapshot(
        _layout(),
        _state(input_value=0.1),
        tmp_path / "snapshot.png",
        edge_intensity="opacity",
        scales={"input": [1.0], "output": 1.0, "activation": 1.0, "weight": 0.5},
    )

    _, edge_kwargs = fake_pyvista.plotters[0].meshes[0]
    assert edge_kwargs["opacity"] < 1.0


def test_render_state_snapshot_rejects_unknown_edge_intensity(monkeypatch, tmp_path):
    fake_pyvista = FakePyVista()
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    with pytest.raises(ValueError, match="edge_intensity"):
        render_state_snapshot(_layout(), _state(), tmp_path / "snapshot.png", edge_intensity="weird")


def _layout() -> NetworkLayout:
    return NetworkLayout(
        nodes=(
            Node("in", 0, "x", 0.0, 0.0, 0.1, z=0.0),
            Node("out", 1, "left", 1.0, 2.0, 0.2, z=3.0),
        ),
        edges=(Edge("in", 0, "out", 1, 0.5, 0.5, 0.5),),
    )


def _state(*, input_value: float = 1.0) -> NetworkState:
    return NetworkState(
        inputs=np.array([input_value], dtype=np.float32),
        h1=np.array([], dtype=np.float32),
        h2=np.array([], dtype=np.float32),
        q_values=np.array([0.0, 1.0], dtype=np.float32),
        action=1,
    )


class FakePyVista:
    def __init__(self):
        self.plotters = []
        self.spheres = []
        self.lines = []
        self.tubes = []

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
        return FakeLine(self, pointa, pointb)


class FakeLine:
    kind = "line"

    def __init__(self, pyvista, pointa, pointb):
        self.pyvista = pyvista
        self.pointa = pointa
        self.pointb = pointb

    def tube(self, *, radius, n_sides):
        tube = {"radius": radius, "n_sides": n_sides}
        self.pyvista.tubes.append(tube)
        return SimpleNamespace(kind="tube", **tube)


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
