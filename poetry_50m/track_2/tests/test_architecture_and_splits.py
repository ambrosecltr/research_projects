from __future__ import annotations

from genome.architecture_graph import build_architecture_graph
from genome.splits import assign_records_by_source, deterministic_source_split, validate_source_isolation


def test_architecture_graph_contains_ties_and_layer_edges(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    graph = build_architecture_graph(specimen.inventory, specimen.tied_groups)
    tensors = graph.as_tensors()
    assert len(graph.nodes) == len(specimen.inventory)
    assert tensors["node_features"].shape[0] == len(specimen.inventory)
    assert tensors["edge_index"].shape[0] == 2
    assert any(edge.relation == "tied" for edge in graph.edges)
    assert any(edge.relation == "same_role_next_layer" for edge in graph.edges)


def test_source_level_split_is_deterministic_and_isolated():
    sources = [f"poem:{index}" for index in range(30)]
    first = deterministic_source_split(
        sources, seed=1701, fractions={"fit": 0.6, "probe": 0.2, "hidden": 0.2}
    )
    second = deterministic_source_split(
        reversed(sources), seed=1701, fractions={"fit": 0.6, "probe": 0.2, "hidden": 0.2}
    )
    assert first == second
    records = [
        {"record_id": f"record:{index}", "source_id": sources[index // 2]}
        for index in range(60)
    ]
    assignments = assign_records_by_source(records, first)
    validate_source_isolation(assignments)
    for source in sources:
        assert len({item.split for item in assignments if item.source_id == source}) == 1
