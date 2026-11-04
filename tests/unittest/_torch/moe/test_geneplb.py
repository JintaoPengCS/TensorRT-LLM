# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB (policy adaptive): geometry, planner decisions, the planner worker, the config switch.

The planner side is standalone (numpy + gcc), so most of this file runs on a CPU-only host; the config
test needs an importable tensorrt_llm and is skipped otherwise.
"""

import importlib
import random
import sys
from pathlib import Path

import pytest

GENEPLB = (
    Path(__file__).resolve().parents[4]
    / "tensorrt_llm"
    / "_torch"
    / "moe"
    / "fused_moe"
    / "geneplb"
)


@pytest.fixture(scope="module")
def mods(tmp_path_factory):
    sys.path.insert(0, str(GENEPLB))
    try:
        geometry = importlib.import_module("geometry")
        planner = importlib.import_module("planner")
        shares = importlib.import_module("shares")
        shares.configure(str(tmp_path_factory.mktemp("geneplb-cache")))
        yield geometry, planner, shares
    finally:
        sys.path.remove(str(GENEPLB))


def _geo(geometry, experts=384, ranks=32, num_slots=448):
    m = geometry.local_slot_count(experts, ranks, num_slots)
    return geometry.Geometry(
        experts=experts, ranks=ranks, local_slots=m, active_slots=num_slots, top_k=6
    )


def test_local_slot_count(mods):
    geometry = mods[0]
    assert geometry.local_slot_count(384, 32, 448) == 16  # the shipped geometry: 12 homes + 4 spare
    assert geometry.local_slot_count(384, 16, 416) == 28
    assert geometry.local_slot_count(256, 4, 264) == 68
    with pytest.raises(ValueError):
        geometry.local_slot_count(384, 32, 384)
    with pytest.raises(ValueError):
        geometry.local_slot_count(384, 5, 448)


def test_home_layout(mods):
    geo = _geo(mods[0])
    home = geo.home_layout()
    assert len(home) == 512 and sorted(e for e in home if e >= 0) == list(range(384))
    assert all(home[geo.home_slot(e)] == e for e in range(384))
    assert sum(e < 0 for e in home) == 128


def test_do_replication(mods):
    planner = mods[1]
    load = [0.0] * 384
    load[7], load[9] = 100.0, 50.0
    counts = planner.do_replication(load, 448, 384)
    assert sum(counts) == 448 and min(counts) == 1
    assert counts[7] > counts[9] > 1
    assert planner.do_replication([0.0] * 384, 448, 384) == [2] * 64 + [1] * 320


def test_fixedhome_and_sticky(mods):
    geometry, planner, _ = mods
    geo = _geo(geometry)
    rng = random.Random(7)
    load = [rng.expovariate(1.0) * (20.0 if e % 37 == 0 else 1.0) for e in range(384)]
    plan = planner.fixedhome_targets(load, geo, 448, cap=4)
    assert sum(len(v) for v in plan.values()) == 64
    for e, ranks in plan.items():
        assert (
            e // geo.homes not in ranks or len(ranks) == geo.ranks
        )  # never on the home rank while another has room
    per_rank = [sum(r == k for v in plan.values() for r in v) for k in range(geo.ranks)]
    assert max(per_rank) <= 4
    cur = geo.home_layout()
    new, copies = planner.sticky_layout(cur, plan, geo)
    assert len(copies) == 64 and all(cur[s] == -1 and new[s] == e for e, s in copies)
    # a second identical plan keeps every replica in place: nothing to copy
    again, copies2 = planner.sticky_layout(new, plan, geo)
    assert again == new and copies2 == []
    # a rank whose spare slots are taken (or retired) cannot take a new replica in this wave: it is left out
    rank0_spare = list(range(geo.homes, geo.local_slots))
    _, copies3 = planner.sticky_layout(cur, {200: [0]}, geo, retired=rank0_spare)
    assert copies3 == []
    _, copies4 = planner.sticky_layout(cur, {200: [0]}, geo)
    assert copies4 == [(200, geo.homes)]


def test_remaining_steps(mods):
    planner = mods[1]
    assert planner.remaining_steps(4000.0, 4, 2.0) == 500.0
    assert planner.remaining_steps(10.0, 0, 2.0) is None


@pytest.mark.parametrize("mode", ["C", "B"])
def test_decide(mods, mode):
    geometry, planner, _ = mods
    geo = _geo(geometry)
    rng = random.Random(11)
    # per-launch rows of a large decode tier: hot experts far above one 256-row tile, so replicas pay under form T too
    loads = {
        li: [100.0 * rng.expovariate(1.0) * (30.0 if e % 41 == li else 1.0) for e in range(384)]
        for li in range(3)
    }
    import costmodel

    b = costmodel.prior_params(64) if mode == "B" else None
    req = dict(
        geo=dict(experts=384, ranks=32, local_slots=16, active_slots=448, top_k=6),
        mode=mode,
        b_params=b,
        gate_model=b,
        b_layers={li: mode == "B" for li in loads},
        totals=loads,
        scales={li: 1.0 for li in loads},
        layouts={li: geo.home_layout() for li in loads},
        n_rem=300.0,
        c_gpu_ms=None,
        total_active=448,
        cap=4,
        install=True,
        max_layers=3,
    )
    req["retired"] = {}
    out = planner.decide(req)
    assert set(out["tables"]) == set(loads)
    assert out["installs"], (
        "an imbalanced layer with 300 steps left and no measured copy cost installs"
    )
    for ins in out["installs"]:
        assert all(req["layouts"][ins["layer"]][s] == -1 for _, s in ins["copies"])
    # deterministic: every rank computes the same decision from the same inputs
    again = planner.decide(req)
    assert [i["copies"] for i in again["installs"]] == [i["copies"] for i in out["installs"]]
    assert all(
        (a == b).all() for li in loads for a, b in zip(again["tables"][li], out["tables"][li])
    )
    # no remaining steps: shares only
    assert planner.decide(dict(req, n_rem=None))["installs"] == []


def test_worker_roundtrip(mods, tmp_path):
    sys.path.insert(0, str(GENEPLB))
    try:
        worker = importlib.import_module("worker")
        w = worker.Worker(str(tmp_path))
        req = dict(
            geo=dict(experts=384, ranks=32, local_slots=16, active_slots=448, top_k=6),
            mode="C",
            b_params=None,
            gate_model=None,
            b_layers={0: False},
            totals={0: [1.0] * 384},
            scales={0: 1.0},
            layouts={0: _geo(mods[0]).home_layout()},
            retired={},
            n_rem=None,
            c_gpu_ms=None,
            total_active=448,
            cap=4,
            install=True,
            max_layers=1,
        )
        w.request("decide", req)
        _, reply = w.wait()
        assert set(reply["tables"]) == {0} and reply["installs"] == []
        w.close()
    finally:
        sys.path.remove(str(GENEPLB))


def test_config_switch():
    llm_args = pytest.importorskip("tensorrt_llm.llmapi.llm_args")
    cfg = llm_args.MoeConfig(load_balancer={"num_slots": 448, "policy": "adaptive"}).load_balancer
    assert cfg.mode == "standard" and cfg.policy == "adaptive" and cfg.cost_model_b is None
    cfg = llm_args.MoeConfig(
        load_balancer={"num_slots": 448, "policy": "adaptive", "cost_model_b": {"source": "prior"}}
    ).load_balancer
    assert cfg.cost_model_b == {"source": "prior"}
    with pytest.raises(ValueError):
        llm_args.MoeConfig(load_balancer={"num_slots": 448, "cost_model_b": {"source": "prior"}})
    with pytest.raises(ValueError):
        llm_args.MoeConfig(
            load_balancer={"num_slots": 448, "policy": "adaptive", "layer_updates_per_iter": 1}
        )
    with pytest.raises(ValueError):
        llm_args.MoeConfig(load_balancer={"policy": "adaptive"})
    assert llm_args.MoeConfig(load_balancer={"num_slots": 448}).load_balancer.policy is None
