import polars as pl

from src.econ.local_projection import half_life
from src.econ.weights import edges_to_neighbors


def test_edges_to_neighbors_undirected_and_includes_isolates():
    edges = pl.DataFrame({"src": ["a", "b"], "dst": ["b", "c"]})
    neighbors = edges_to_neighbors(edges, ["a", "b", "c", "d"])
    assert neighbors["a"] == ["b"]
    assert neighbors["b"] == ["a", "c"]
    assert neighbors["c"] == ["b"]
    assert neighbors["d"] == []


def test_edges_to_neighbors_drops_self_loops():
    edges = pl.DataFrame({"src": ["a", "b"], "dst": ["a", "c"]})
    neighbors = edges_to_neighbors(edges, ["a", "b", "c"])
    assert neighbors["a"] == []
    assert neighbors["b"] == ["c"]


def test_half_life_finds_decay_after_peak():
    irf = pl.DataFrame({
        "horizon_minutes": [0, 5, 10, 15, 20],
        "beta": [1.0, 4.0, 2.0, 1.0, 0.5],
    })
    # peak is at horizon 5 (beta=4.0); half of peak is 2.0; first horizon
    # after the peak below that is horizon 15 (beta=1.0).
    assert half_life(irf) == 15.0


def test_half_life_returns_none_for_zero_response():
    irf = pl.DataFrame({"horizon_minutes": [0, 5, 10], "beta": [0.0, 0.0, 0.0]})
    assert half_life(irf) is None


def test_half_life_returns_none_if_never_decays():
    irf = pl.DataFrame({"horizon_minutes": [0, 5, 10], "beta": [1.0, 2.0, 3.0]})
    assert half_life(irf) is None
