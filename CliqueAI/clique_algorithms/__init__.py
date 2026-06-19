from .anytime_algorithm import anytime_clique_algorithm
from .networkx_algorithm import networkx_algorithm


def scattering_clique_algorithm(*args, **kwargs):
    """Lazy-load the optional torch/GNN algorithm only when explicitly used."""
    from .gnn_algorithm import scattering_clique_algorithm as _scattering_clique_algorithm

    return _scattering_clique_algorithm(*args, **kwargs)
