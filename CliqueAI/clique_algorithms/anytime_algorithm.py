import hashlib
import random
import time
from collections.abc import Sequence


def _maximalize(clique: list[int], candidates: int, adj_bits: Sequence[int]) -> list[int]:
    """Extend a clique until no remaining node connects to every clique node."""
    while candidates:
        # Prefer nodes that keep the largest future candidate set.
        best_node = -1
        best_score = -1
        scan = candidates
        while scan:
            lowest = scan & -scan
            node = lowest.bit_length() - 1
            score = (candidates & adj_bits[node]).bit_count()
            if score > best_score:
                best_node = node
                best_score = score
            scan ^= lowest
        clique.append(best_node)
        candidates &= adj_bits[best_node]
    return clique


def _greedy_from_candidates(
    candidates: int,
    adj_bits: Sequence[int],
    rng: random.Random,
    deadline: float,
    sample_width: int,
) -> list[int]:
    clique: list[int] = []
    while candidates and time.perf_counter() < deadline:
        # Select among a small pool of high-potential candidates. This keeps the
        # construction cheap but avoids returning the exact same clique every run.
        best: list[tuple[int, int]] = []
        scan = candidates
        seen = 0
        while scan:
            lowest = scan & -scan
            node = lowest.bit_length() - 1
            score = (candidates & adj_bits[node]).bit_count()
            if len(best) < sample_width:
                best.append((score, node))
                if len(best) == sample_width:
                    best.sort(reverse=True)
            elif score > best[-1][0]:
                best[-1] = (score, node)
                best.sort(reverse=True)
            scan ^= lowest
            seen += 1
            # For very dense 900-node graphs, scanning every candidate every
            # iteration is still acceptable, but this guard keeps short 6s jobs
            # responsive if Python gets CPU-starved.
            if seen % 128 == 0 and time.perf_counter() >= deadline:
                break
        if not best:
            break
        top_k = min(len(best), max(1, sample_width // 2))
        _, node = best[rng.randrange(top_k)]
        clique.append(node)
        candidates &= adj_bits[node]
    return _maximalize(clique, candidates, adj_bits)


def _is_clique(clique: Sequence[int], adj_bits: Sequence[int]) -> bool:
    bits = 0
    for node in clique:
        if bits & ~adj_bits[node]:
            return False
        bits |= 1 << node
    return True


def _is_maximal(clique: Sequence[int], all_bits: int, adj_bits: Sequence[int]) -> bool:
    candidates = all_bits
    for node in clique:
        candidates &= adj_bits[node]
    for node in clique:
        candidates &= ~(1 << node)
    return candidates == 0


def _ensure_maximal(clique: list[int], all_bits: int, adj_bits: Sequence[int]) -> list[int]:
    candidates = all_bits
    for node in clique:
        candidates &= adj_bits[node]
    for node in clique:
        candidates &= ~(1 << node)
    return _maximalize(clique, candidates, adj_bits)


def anytime_clique_algorithm(
    number_of_nodes: int,
    adjacency_list: list[list[int]],
    deadline: float | None = None,
    time_limit: float | None = None,
    seed_material: str | None = None,
) -> list[int]:
    """Deadline-aware dense-graph maximum-clique heuristic.

    The validator rewards valid maximal cliques by relative size. This algorithm
    is designed for mining: always keep a valid maximal fallback, then spend the
    remaining budget on randomized greedy improvements. It returns before the
    provided absolute ``deadline`` where possible.
    """
    started = time.perf_counter()
    if deadline is None:
        budget = 5.0 if time_limit is None else max(0.05, float(time_limit))
        deadline = started + budget
    else:
        deadline = float(deadline)
    budget = max(0.0, deadline - started)

    if number_of_nodes <= 0:
        return []

    adj_bits: list[int] = []
    for neighbors in adjacency_list:
        bits = 0
        for node in neighbors:
            if 0 <= node < number_of_nodes:
                bits |= 1 << node
        adj_bits.append(bits)

    all_bits = (1 << number_of_nodes) - 1
    degrees = [bits.bit_count() for bits in adj_bits]
    order = sorted(range(number_of_nodes), key=lambda node: degrees[node], reverse=True)

    # Immediate safe answer: a high-degree greedy maximal clique. Even if the
    # remaining budget is tiny, this avoids empty deadline misses on hard jobs.
    best = _greedy_from_candidates(all_bits, adj_bits, random.Random(0), deadline, sample_width=1)
    if not best:
        return []

    # Seed RNG deterministically per graph/request enough to get diversity across
    # different graphs while keeping reproducible behavior for debugging.
    seed_bytes = (seed_material or str(number_of_nodes)).encode()
    seed = int.from_bytes(hashlib.blake2s(seed_bytes, digest_size=8).digest(), "big")
    rng = random.Random(seed)

    # Try strong starts from high-degree vertices first, then randomized restarts.
    seed_nodes = order[: min(number_of_nodes, 160)]
    idx = 0
    sample_width = 6 if number_of_nodes < 600 else 10
    attempts = 0
    last_improved = time.perf_counter()
    if number_of_nodes >= 850:
        effort_fraction = 0.45
        effort_cap = 6.0
        stagnation_limit = 1.75
    elif number_of_nodes >= 650:
        effort_fraction = 0.35
        effort_cap = 4.0
        stagnation_limit = 1.25
    elif number_of_nodes >= 450:
        effort_fraction = 0.30
        effort_cap = 3.0
        stagnation_limit = 0.90
    else:
        effort_fraction = 0.20
        effort_cap = 1.50
        stagnation_limit = 0.45
    minimum_runtime = min(max(0.15, budget * effort_fraction), effort_cap)
    while time.perf_counter() < deadline:
        attempts += 1
        if idx < len(seed_nodes):
            start = seed_nodes[idx]
            idx += 1
        else:
            # Bias toward high-degree vertices but occasionally sample the tail
            # for diversity/escape from identical top-degree cliques.
            cutoff = min(number_of_nodes, max(32, number_of_nodes // 3))
            start = order[rng.randrange(cutoff)] if rng.random() < 0.85 else rng.randrange(number_of_nodes)

        clique = [start]
        candidates = adj_bits[start]
        candidate = _greedy_from_candidates(candidates, adj_bits, rng, deadline, sample_width)
        clique.extend(candidate)
        clique = _ensure_maximal(clique, all_bits, adj_bits)

        if len(clique) > len(best) and _is_clique(clique, adj_bits) and _is_maximal(clique, all_bits, adj_bits):
            best = clique
            last_improved = time.perf_counter()

        now = time.perf_counter()
        if (
            attempts >= 64
            and now - started >= minimum_runtime
            and now - last_improved >= stagnation_limit
        ):
            break

    return best
