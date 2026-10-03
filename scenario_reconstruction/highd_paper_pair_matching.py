"""Exact disjoint matching of current following chains, before action draws."""
from .highd_dynamic_pairs import pair_key


def maximum_weight_following(candidates, previous=()):
    """Maximize total nonnegative edge weight; retain old pairs only on ties.

    Nearest-leader relations form paths, so a small dynamic program is exact.
    No future state, sampled action or score-reference histogram enters here.
    """
    previous = set(previous)
    outgoing = {c['actor_ids'][0]: c for c in candidates}
    incoming = {c['actor_ids'][1] for c in candidates}
    if len(outgoing) != len(candidates) or len(incoming) != len(candidates):
        raise ValueError('Expected disjoint directed following paths')
    selected = []
    visited = set()
    for start in sorted(set(outgoing)-incoming):
        chain = []
        actor = start
        while actor in outgoing:
            if actor in visited:
                raise ValueError('Cyclic following relation')
            visited.add(actor)
            edge = outgoing[actor]
            chain.append(edge)
            actor = edge['actor_ids'][1]
        # (total weight, retained pair count, selected edge indices).
        best = [(0., 0, ())]
        for i, edge in enumerate(chain):
            weight = float(edge['weight'])
            if not 0 <= weight < float('inf'):
                raise ValueError('Matching weights must be finite and nonnegative')
            prior = best[max(0, i-1)]
            take = (prior[0]+weight, prior[1]+int(pair_key(edge) in previous), prior[2]+(i,))
            skip = best[-1]
            best.append(take if weight > 0 and take[:2] > skip[:2] else skip)
        selected.extend(chain[i] for i in best[-1][2])
    if len(visited) != len(candidates):
        raise ValueError('Cyclic following relation')
    return selected
