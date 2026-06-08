"""
metaheuristicas.py
==================
Metaheuristic optimisers for binary-vector search spaces.

All optimisers share the same interface:

    best_solution, solutions_dict = optimiser(
        objective_function,   # f(vector) -> (cost: float, aux: any)
        chromosome_length,    # int
        **kwargs
    )

The objective_function is MINIMISED (cost = -metric).
"""

import math
import random
from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_lookup(objective_function, vec):
    cache = getattr(objective_function, "cache", None)
    key = str(vec)
    if cache is not None and key in cache:
        return cache[key]["cost"]
    cost, _ = objective_function(vec)
    if cache is not None and key in cache:
        return cache[key]["cost"]
    return cost


def _batch_eval(objective_function, candidates):
    """Evaluate candidates sequentially, skipping cached ones."""
    cache = getattr(objective_function, "cache", None)
    for vec in candidates:
        if cache is None or str(vec) not in cache:
            objective_function(vec)


def _all_single_flip_neighbors(vector: List[int]):
    """Return full 1-flip neighbourhood as (bit_idx, candidate) tuples."""
    return [
        (i, vector[:i] + [vector[i] ^ 1] + vector[i + 1:])
        for i in range(len(vector))
    ]


def _repair_zero_vector(vector, preferred_indices=None):
    """Activate one bit if the vector is all-zero."""
    if not vector or any(vector):
        return vector[:]
    repaired = vector[:]
    if preferred_indices:
        for idx in preferred_indices:
            if 0 <= idx < len(repaired):
                repaired[idx] = 1
                return repaired
    repaired[0] = 1
    return repaired


def _perturb_solution(vector, fraction, rng):
    """Flip a random fraction of bits."""
    out = vector[:]
    n_flip = max(1, min(len(out), int(round(fraction * len(out)))))
    for idx in rng.sample(range(len(out)), n_flip):
        out[idx] ^= 1
    return out


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    objective_function,
    chromosome_length,
    initial_solution=None,
    initial_temp=100.0,
    cooling_rate=0.95,
    stopping_temp=1e-3,
    max_iterations=10,
    num_neighbors=5,
    seed=12345,
    **kwargs,
):
    """Simulated annealing for binary vectors."""
    random.seed(seed)
    solutions_dict = getattr(objective_function, "cache", {})

    current = (
        initial_solution[:]
        if initial_solution is not None
        else [random.randint(0, 1) for _ in range(chromosome_length)]
    )
    current_cost = _cache_lookup(objective_function, current)
    best, best_cost = current[:], current_cost

    temp = initial_temp
    iteration = 0
    while temp > stopping_temp and iteration < max_iterations:
        candidates = []
        for _ in range(num_neighbors):
            c = current[:]
            c[random.randint(0, chromosome_length - 1)] ^= 1
            candidates.append(c)

        _batch_eval(objective_function, candidates)

        for candidate in candidates:
            candidate_cost = solutions_dict[str(candidate)]["cost"]
            delta = candidate_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current, current_cost = candidate[:], candidate_cost
            if current_cost < best_cost:
                best, best_cost = current[:], current_cost

        temp *= cooling_rate
        iteration += 1
        print(f"  [SA] iter={iteration:>3}  T={temp:.4f}  best_cost={best_cost:.4f}")

    return best, solutions_dict


# ---------------------------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------------------------

def tabu_search(
    objective_function,
    chromosome_length,
    initial_solution=None,
    max_iterations=12,
    tabu_tenure=4,
    max_no_improve=4,
    neighborhood_sample_size=None,
    restart_fraction=0.2,
    seed=12345,
    **kwargs,
):
    """
    Tabu Search for binary optimisation.

    Uses 1-bit-flip neighbourhood with memory (tabu list), aspiration
    criterion, and random multi-bit restart on stagnation.
    """
    random.seed(seed)
    np.random.seed(seed)

    solutions_dict = getattr(objective_function, "cache", {})
    current = (
        initial_solution[:]
        if initial_solution is not None
        else [random.randint(0, 1) for _ in range(chromosome_length)]
    )
    current_cost = _cache_lookup(objective_function, current)
    best, best_cost = current[:], current_cost

    tabu_until = {}
    no_improve = 0

    for iteration in range(1, max_iterations + 1):
        neighbours = _all_single_flip_neighbors(current)
        if neighborhood_sample_size is not None and neighborhood_sample_size < len(neighbours):
            neighbours = random.sample(neighbours, neighborhood_sample_size)

        _batch_eval(objective_function, [cand for _, cand in neighbours])

        best_admissible = None
        best_admissible_cost = float("inf")
        best_any = None
        best_any_cost = float("inf")
        chosen_bit = None

        for bit_idx, cand in neighbours:
            cand_cost = solutions_dict[str(cand)]["cost"]
            if cand_cost < best_any_cost:
                best_any, best_any_cost = cand[:], cand_cost

            is_tabu = tabu_until.get(bit_idx, 0) > iteration
            aspiration = cand_cost < best_cost
            if is_tabu and not aspiration:
                continue
            if cand_cost < best_admissible_cost:
                best_admissible, best_admissible_cost, chosen_bit = cand[:], cand_cost, bit_idx

        if best_admissible is None:
            best_admissible, best_admissible_cost = best_any[:], best_any_cost
            chosen_bit = next(
                i for i, (a, b) in enumerate(zip(current, best_admissible)) if a != b
            )

        current, current_cost = best_admissible[:], best_admissible_cost
        tabu_until[chosen_bit] = iteration + max(1, int(tabu_tenure))

        if current_cost < best_cost:
            best, best_cost = current[:], current_cost
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= max_no_improve:
            restart = best[:]
            for bit_idx in random.sample(range(chromosome_length),
                                         max(1, int(round(restart_fraction * chromosome_length)))):
                restart[bit_idx] ^= 1
            _cache_lookup(objective_function, restart)
            current = restart
            current_cost = solutions_dict[str(restart)]["cost"]
            no_improve = 0
            print(f"  [TS] iter={iteration:>3}  diversify  best_cost={best_cost:.4f}  evals={len(solutions_dict)}")
            continue

        print(f"  [TS] iter={iteration:>3}  best_cost={best_cost:.4f}  current_cost={current_cost:.4f}  evals={len(solutions_dict)}")

    return best, solutions_dict


# ---------------------------------------------------------------------------
# Iterated Local Search
# ---------------------------------------------------------------------------

def iterated_local_search(
    objective_function,
    chromosome_length,
    initial_solution=None,
    n_restarts=5,
    perturbation_strength=0.3,
    local_search_iters=15,
    local_search_initial_temp=5.0,
    local_search_cooling_rate=0.85,
    local_search_stopping_temp=1e-3,
    local_search_num_neighbors=3,
    seed=12345,
    warm_start=None,
    repair=True,
    **kwargs,
):
    """
    Iterated Local Search: repeated SA local searches with perturbation.

    Uses a warm-start mask (from correlation strengths) when available.
    Repairs all-zero masks to avoid degenerate solutions.
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    solutions_dict = getattr(objective_function, "cache", {})

    preferred_source = warm_start if warm_start is not None else initial_solution
    preferred_indices = [i for i, bit in enumerate(preferred_source or []) if bit == 1]

    if warm_start is not None:
        incumbent = warm_start[:]
    elif initial_solution is not None:
        incumbent = initial_solution[:]
    else:
        incumbent = [rng.randint(0, 1) for _ in range(chromosome_length)]

    if repair:
        incumbent = _repair_zero_vector(incumbent, preferred_indices)

    best = incumbent[:]
    best_cost = _cache_lookup(objective_function, best)

    for restart_idx in range(max(1, int(n_restarts))):
        seed_solution = (
            incumbent[:]
            if restart_idx == 0
            else _repair_zero_vector(
                _perturb_solution(best, perturbation_strength, rng),
                preferred_indices,
            ) if repair else _perturb_solution(best, perturbation_strength, rng)
        )

        local_best, _ = simulated_annealing(
            objective_function,
            chromosome_length,
            initial_solution=seed_solution,
            initial_temp=local_search_initial_temp,
            cooling_rate=local_search_cooling_rate,
            stopping_temp=local_search_stopping_temp,
            max_iterations=local_search_iters,
            num_neighbors=local_search_num_neighbors,
            seed=seed + restart_idx,
        )
        local_cost = _cache_lookup(objective_function, local_best)

        if local_cost < best_cost:
            best, best_cost = local_best[:], local_cost

        print(f"  [ILS] restart={restart_idx + 1:>3}/{n_restarts}  best_cost={best_cost:.4f}")

    return best, solutions_dict


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

def genetic_algorithm(
    objective_function,
    chromosome_length,
    population_size=20,
    n_generations=10,
    crossover_rate=0.8,
    mutation_rate=None,
    tournament_size=3,
    elitism_count=1,
    seed=12345,
    **kwargs,
):
    """
    Steady-state genetic algorithm for binary optimisation.

    Available for ablations; for SEQUENT's expensive objective, SA/ILS/TS
    typically outperform GA by spending fewer evaluations on diverse candidates.
    """
    random.seed(seed)
    np.random.seed(seed)

    if mutation_rate is None:
        mutation_rate = 1.0 / chromosome_length

    solutions_dict = getattr(objective_function, "cache", {})

    def _eval_pop(pop):
        _batch_eval(objective_function, pop)
        return [solutions_dict[str(ind)]["cost"] for ind in pop]

    population = [
        [random.randint(0, 1) for _ in range(chromosome_length)]
        for _ in range(population_size)
    ]
    fitness = _eval_pop(population)

    best_idx = int(np.argmin(fitness))
    best, best_cost = population[best_idx][:], fitness[best_idx]
    print(f"  [GA] gen=  0  best_cost={best_cost:.4f}")

    for gen in range(1, n_generations + 1):
        ranked = sorted(range(population_size), key=lambda i: fitness[i])
        new_population = [population[i][:] for i in ranked[:elitism_count]]

        while len(new_population) < population_size:
            pa = population[min(random.sample(range(population_size), tournament_size), key=lambda i: fitness[i])][:]
            pb = population[min(random.sample(range(population_size), tournament_size), key=lambda i: fitness[i])][:]
            child = (
                [pa[i] if random.random() < 0.5 else pb[i] for i in range(chromosome_length)]
                if random.random() < crossover_rate else pa[:]
            )
            child = [bit ^ 1 if random.random() < mutation_rate else bit for bit in child]
            new_population.append(child)

        population = new_population
        fitness = _eval_pop(population)

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_cost:
            best_cost = fitness[gen_best_idx]
            best = population[gen_best_idx][:]

        print(f"  [GA] gen={gen:>3}  best_cost={best_cost:.4f}  gen_best={fitness[gen_best_idx]:.4f}  evals={len(solutions_dict)}")

    return best, solutions_dict
