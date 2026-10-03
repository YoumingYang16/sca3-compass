"""Transparent adaptive-learning engine with Bayesian Knowledge Tracing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def update_bkt(
    prior: float,
    *,
    correct: bool,
    p_learn: float,
    p_guess: float,
    p_slip: float,
) -> dict[str, float]:
    """One auditable Bayesian Knowledge Tracing update."""
    for name, value in {
        "prior": prior,
        "p_learn": p_learn,
        "p_guess": p_guess,
        "p_slip": p_slip,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if correct:
        likelihood_known = 1.0 - p_slip
        likelihood_unknown = p_guess
    else:
        likelihood_known = p_slip
        likelihood_unknown = 1.0 - p_guess
    evidence = prior * likelihood_known + (1.0 - prior) * likelihood_unknown
    posterior_observation = (
        prior * likelihood_known / evidence if evidence > 0 else prior
    )
    posterior_learning = posterior_observation + (1.0 - posterior_observation) * p_learn
    return {
        "prior": prior,
        "likelihood": evidence,
        "posterior_after_observation": posterior_observation,
        "posterior_after_learning": posterior_learning,
    }


def _entropy(probability: float) -> float:
    value = min(max(probability, 1e-12), 1.0 - 1e-12)
    return -(value * math.log2(value) + (1.0 - value) * math.log2(1.0 - value))


def expected_information_gain(prior: float, concept: dict[str, Any]) -> float:
    p_correct = prior * (1.0 - concept["p_slip"]) + (1.0 - prior) * concept["p_guess"]
    correct_update = update_bkt(
        prior,
        correct=True,
        p_learn=0.0,
        p_guess=concept["p_guess"],
        p_slip=concept["p_slip"],
    )["posterior_after_observation"]
    incorrect_update = update_bkt(
        prior,
        correct=False,
        p_learn=0.0,
        p_guess=concept["p_guess"],
        p_slip=concept["p_slip"],
    )["posterior_after_observation"]
    expected_entropy = p_correct * _entropy(correct_update) + (
        1.0 - p_correct
    ) * _entropy(incorrect_update)
    return max(0.0, _entropy(prior) - expected_entropy)


def initial_mastery(model: dict[str, Any]) -> dict[str, float]:
    return {item["id"]: float(item["p_init"]) for item in model["concepts"]}


def select_next_item(
    model: dict[str, Any],
    mastery: dict[str, float],
    history: list[str] | None = None,
) -> dict[str, Any] | None:
    """Select an eligible item using need and expected information gain."""
    completed = set(history or [])
    concepts = {item["id"]: item for item in model["concepts"]}
    candidates: list[tuple[float, float, str, dict[str, Any]]] = []
    for item in model["items"]:
        if item["id"] in completed:
            continue
        concept = concepts[item["concept_id"]]
        if any(mastery.get(parent, 0.0) < 0.65 for parent in concept["prerequisites"]):
            continue
        probability = mastery.get(concept["id"], concept["p_init"])
        if probability >= concept["mastery_threshold"]:
            continue
        information = expected_information_gain(probability, concept)
        difficulty_match = 1.0 - abs(float(item["difficulty"]) - (1.0 - probability))
        score = (
            (1.0 - probability) * 0.55 + information * 0.30 + difficulty_match * 0.15
        )
        candidates.append((score, information, item["id"], item))
    if not candidates:
        # Small item bank: offer explicit review instead of silently dead-ending
        # after two errors exhaust a prerequisite concept's questions.
        eligible = [
            item
            for item in model["items"]
            if mastery.get(item["concept_id"], 0)
            < concepts[item["concept_id"]]["mastery_threshold"]
            and all(
                mastery.get(parent, 0) >= 0.65
                for parent in concepts[item["concept_id"]]["prerequisites"]
            )
        ]
        if not eligible:
            return None
        recent = history or []
        selected = min(
            eligible,
            key=lambda item: max(
                (i for i, value in enumerate(recent) if value == item["id"]), default=-1
            ),
        )
        return {
            **selected,
            "selection": {
                "policy": "prerequisite repair / least-recently-seen review",
                "review": True,
                "expected_information_gain_bits": 0.0,
            },
        }
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2]))
    selected = dict(candidates[0][3])
    selected["selection"] = {
        "policy": "prerequisite-constrained need + expected information gain",
        "score": candidates[0][0],
        "expected_information_gain_bits": candidates[0][1],
    }
    return selected


def evaluate_response(
    model: dict[str, Any],
    *,
    mastery: dict[str, float] | None,
    history: list[str] | None,
    item_id: str,
    selected_index: int,
) -> dict[str, Any]:
    concepts = {item["id"]: item for item in model["concepts"]}
    items = {item["id"]: item for item in model["items"]}
    if item_id not in items:
        raise ValueError(f"Unknown item id: {item_id}")
    item = items[item_id]
    if selected_index < 0 or selected_index >= len(item["options"]):
        raise ValueError("selected_index is outside the item options")
    state = {**initial_mastery(model), **(mastery or {})}
    if any(
        key not in concepts or not math.isfinite(value) or not 0 <= value <= 1
        for key, value in state.items()
    ):
        raise ValueError(
            "Mastery must contain known concepts with finite probabilities"
        )
    concept = concepts[item["concept_id"]]
    correct = selected_index == item["correct_index"]
    review = item_id in (history or [])
    update = update_bkt(
        state[concept["id"]],
        correct=correct,
        p_learn=concept["p_learn"],
        p_guess=max(concept["p_guess"], 0.6) if review else concept["p_guess"],
        p_slip=concept["p_slip"],
    )
    state[concept["id"]] = update["posterior_after_learning"]
    new_history = [*(history or []), item_id]
    misconception = None
    if not correct:
        misconception = item["misconceptions"][selected_index]
    return {
        "item_id": item_id,
        "concept_id": concept["id"],
        "correct": correct,
        "review": review,
        "review_assumption": "Repeated-item guess probability raised to .6 to acknowledge memory; heuristic, not fitted"
        if review
        else None,
        "rationale": item["rationale"],
        "misconception": misconception,
        "mastery": state,
        "update": update,
        "history": new_history,
        "next_item": select_next_item(model, state, new_history),
        "persistence": "none; caller-controlled state",
        "evidence_use": "learning interaction only; not scientific evidence",
    }


def _simulate_policy(
    model: dict[str, Any], *, adaptive: bool, learners: int, steps: int, seed: int
) -> dict[str, Any]:
    concepts = {item["id"]: item for item in model["concepts"]}
    item_sequence = [item["id"] for item in model["items"]]
    items = {item["id"]: item for item in model["items"]}
    post_accuracy: list[float] = []
    brier: list[float] = []
    mastered: list[float] = []
    concept_position = {concept_id: i for i, concept_id in enumerate(concepts)}
    learner_streams = np.random.SeedSequence(seed).spawn(learners)
    for learner in range(learners):
        # Each policy receives the same learner and exogenous random numbers,
        # indexed by step and concept even if its chosen actions differ.
        rng = np.random.default_rng(learner_streams[learner])
        belief = initial_mastery(model)
        truth = {
            concept_id: bool(rng.random() < concept["p_init"])
            for concept_id, concept in concepts.items()
        }
        history: list[str] = []
        for step in range(steps):
            response_uniforms = rng.random(len(concepts))
            learning_uniforms = rng.random(len(concepts))
            if adaptive:
                selected = select_next_item(model, belief, history)
                if selected is None:
                    break
                item = selected
            else:
                eligible = [
                    items[item_id]
                    for item_id in item_sequence
                    if belief[items[item_id]["concept_id"]]
                    < concepts[items[item_id]["concept_id"]]["mastery_threshold"]
                    and all(
                        belief[parent] >= 0.65
                        for parent in concepts[items[item_id]["concept_id"]][
                            "prerequisites"
                        ]
                    )
                ]
                if not eligible:
                    break
                item = min(
                    eligible,
                    key=lambda candidate: (
                        history.count(candidate["id"]),
                        item_sequence.index(candidate["id"]),
                    ),
                )
            concept = concepts[item["concept_id"]]
            is_known = truth[concept["id"]]
            guess = (
                max(0.6, concept["p_guess"])
                if item["id"] in history
                else concept["p_guess"]
            )
            probability_correct = 1.0 - concept["p_slip"] if is_known else guess
            position = concept_position[concept["id"]]
            correct = bool(response_uniforms[position] < probability_correct)
            update = update_bkt(
                belief[concept["id"]],
                correct=correct,
                p_learn=concept["p_learn"],
                p_guess=guess,
                p_slip=concept["p_slip"],
            )
            belief[concept["id"]] = update["posterior_after_learning"]
            if (
                not truth[concept["id"]]
                and learning_uniforms[position] < concept["p_learn"]
            ):
                truth[concept["id"]] = True
            history.append(item["id"])
        probabilities = np.asarray(
            [
                1.0 - concept["p_slip"] if truth[concept_id] else concept["p_guess"]
                for concept_id, concept in concepts.items()
            ]
        )
        post_accuracy.append(float(np.mean(probabilities)))
        brier.append(
            float(
                np.mean(
                    [
                        (belief[concept_id] - float(truth[concept_id])) ** 2
                        for concept_id in concepts
                    ]
                )
            )
        )
        mastered.append(
            float(
                np.mean(
                    [
                        belief[concept_id] >= concept["mastery_threshold"]
                        for concept_id, concept in concepts.items()
                    ]
                )
            )
        )
    return {
        "expected_posttest_accuracy": float(np.mean(post_accuracy)),
        "mastery_fraction": float(np.mean(mastered)),
        "latent_state_brier": float(np.mean(brier)),
        "_per_learner": {
            "expected_posttest_accuracy": post_accuracy,
            "mastery_fraction": mastered,
            "latent_state_brier": brier,
        },
    }


def simulate_policy_comparison(
    model: dict[str, Any], *, learners: int = 1000, steps: int = 12, seed: int = 202709
) -> dict[str, Any]:
    if learners < 100 or learners > 100_000:
        raise ValueError("learners must be between 100 and 100,000")
    if steps < 1 or steps > 100:
        raise ValueError("steps must be between 1 and 100")
    if learners * steps > 300_000:
        raise ValueError(
            "Limit policy experiments to 300,000 learner-steps per request"
        )
    adaptive = _simulate_policy(
        model, adaptive=True, learners=learners, steps=steps, seed=seed
    )
    fixed = _simulate_policy(
        model, adaptive=False, learners=learners, steps=steps, seed=seed
    )
    adaptive_scores = adaptive.pop("_per_learner")
    fixed_scores = fixed.pop("_per_learner")
    intervals = {}
    for key in adaptive_scores:
        differences = np.asarray(adaptive_scores[key]) - np.asarray(fixed_scores[key])
        standard_error = float(np.std(differences, ddof=1) / np.sqrt(learners))
        mean = float(differences.mean())
        intervals[key] = {
            "mean": mean,
            "mc_se": standard_error,
            "low": mean - 1.96 * standard_error,
            "high": mean + 1.96 * standard_error,
        }
    return {
        "model": "Bayesian Knowledge Tracing policy simulation",
        "learners": learners,
        "steps": steps,
        "seed": seed,
        "adaptive": adaptive,
        "fixed_sequence": fixed,
        "comparison_design": "Paired simulated learners; common random numbers indexed by learner/step/concept; both policies obey the same prerequisites and stopping thresholds",
        "comparator_policy": "Least-practiced eligible item, original item-bank order as tie-breaker",
        "paired_difference_intervals": intervals,
        "difference": {
            key: adaptive[key] - fixed[key]
            for key in (
                "expected_posttest_accuracy",
                "mastery_fraction",
                "latent_state_brier",
            )
        },
        "provenance_tier": "SIMULATION_OUTPUT",
        "interpretation_boundary": (
            "This tests algorithm behavior under synthetic learner assumptions; "
            "it is not evidence that real people learn more. The simulator shares the BKT learner assumptions; independent human and model-misspecification validation remain outstanding."
        ),
    }
