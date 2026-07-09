"""Tests for the Router (Phase 1: rules + default)."""

from __future__ import annotations

import pytest

from llmrouter import Router, RouterConfig, Tier


@pytest.fixture(scope="module")
def router() -> Router:
    return Router.from_config(RouterConfig())


def test_cheap_prompt_routes_cheap(router: Router) -> None:
    d = router.route("Classify this ticket by urgency.")
    assert d.tier is Tier.CHEAP
    assert d.classifier_used == "rules"
    assert d.model.tier is Tier.CHEAP


def test_frontier_prompt_routes_frontier(router: Router) -> None:
    d = router.route("Design a consensus protocol for this cluster.")
    assert d.tier is Tier.FRONTIER
    assert d.model.tier is Tier.FRONTIER


def test_rule_miss_flows_to_embedding_scorer(router: Router) -> None:
    # Phase 2: a prompt no rule catches is now scored by the embedding stage,
    # not sent straight to the default tier.
    d = router.route("Tell me a fun fact about honey.")
    assert d.classifier_used == "embedding"
    assert d.score is not None


def test_default_tier_used_when_embedding_unavailable(router: Router) -> None:
    # If the embedding scorer can't be built (offline / no model cache), a
    # rule miss degrades gracefully to the default tier — never fails.
    stub = Router(router.registry, router.policy, router.config)
    stub._embedding_failed = True  # simulate unavailable scorer
    d = stub.route("Tell me a fun fact about honey.")
    assert d.classifier_used == "default"
    assert d.tier is stub.policy.default_tier


def test_chosen_model_is_cheapest_in_tier(router: Router) -> None:
    d = router.route("Prove this identity.")
    cheapest = router.registry.cheapest_in_tier(Tier.FRONTIER)
    assert d.model.name == cheapest.name


def test_route_decision_is_frozen(router: Router) -> None:
    d = router.route("Extract the entities.")
    with pytest.raises(Exception):
        d.tier = Tier.FRONTIER  # type: ignore[misc]


def test_from_config_loads_registry_and_policy(router: Router) -> None:
    assert len(router.registry.models) >= 3
    assert {m.tier for m in router.registry.models} == set(Tier)
