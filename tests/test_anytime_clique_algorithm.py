import json
import random
import time
from pathlib import Path

from CliqueAI.clique_algorithms import anytime_clique_algorithm, networkx_algorithm
from CliqueAI.graph.codec import GraphCodec
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph


def is_valid_maximal_clique(adjacency_list: list[list[int]], clique: list[int]) -> bool:
    node_set = set(clique)
    if not clique or len(node_set) != len(clique):
        return False
    if any(node < 0 or node >= len(adjacency_list) for node in clique):
        return False
    adjacency_sets = [set(neighbors) for neighbors in adjacency_list]
    for i, left in enumerate(clique):
        for right in clique[i + 1 :]:
            if right not in adjacency_sets[left]:
                return False
    return not any(
        node not in node_set and all(node in adjacency_sets[member] for member in clique)
        for node in range(len(adjacency_list))
    )


def load_adjacency(path: str) -> tuple[int, list[list[int]]]:
    synapse = MaximumCliqueOfLambdaGraph.model_validate(json.loads(Path(path).read_text()))
    codec = GraphCodec()
    adjacency_list = codec.matrix_to_list(codec.decode_matrix(synapse.encoded_matrix))
    return synapse.number_of_nodes, adjacency_list


def make_dense_graph(n: int, p: float, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    adjacency_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)
    return adjacency_list


def test_anytime_returns_valid_maximal_clique_on_fixture() -> None:
    n, adjacency_list = load_adjacency("test_data/general_0.4.json")
    start = time.perf_counter()
    clique = anytime_clique_algorithm(
        n, adjacency_list, deadline=start + 1.0, seed_material="fixture"
    )
    assert time.perf_counter() - start < 1.2
    assert is_valid_maximal_clique(adjacency_list, clique)


def test_anytime_beats_networkx_baseline_on_dense_fixture() -> None:
    n, adjacency_list = load_adjacency("test_data/general_0.4.json")
    anytime_clique = anytime_clique_algorithm(
        n, adjacency_list, deadline=time.perf_counter() + 1.0, seed_material="fixture"
    )
    networkx_clique = networkx_algorithm(n, adjacency_list)
    assert is_valid_maximal_clique(adjacency_list, anytime_clique)
    assert len(anytime_clique) >= len(networkx_clique)


def test_anytime_handles_900_node_short_deadline() -> None:
    adjacency_list = make_dense_graph(900, 0.735, seed=900)
    start = time.perf_counter()
    clique = anytime_clique_algorithm(
        900, adjacency_list, deadline=start + 7.5, seed_material="900"
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 7.5
    assert is_valid_maximal_clique(adjacency_list, clique)
    assert len(clique) >= 24
