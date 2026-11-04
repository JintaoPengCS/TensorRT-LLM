# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slot geometry of GenEPLB (standard library only: the planner worker imports it standalone).

Every rank holds its H = E / EP home experts in local slots 0..H-1 and S spare landing slots after
them, M = H + S physical slots per rank. ``num_slots`` (the yaml value) counts the ACTIVE slots, every
expert once plus its replicas; the spare slots are twice the active replicas per rank so a new replica
always lands in a slot that no rank routes to while the old one keeps serving (the shipped geometry:
E 384, EP 32, num_slots 448 -> H 12, S 4, M 16, 512 physical slots).
"""

from dataclasses import dataclass


def local_slot_count(num_experts: int, ep_size: int, num_slots: int) -> int:
    """Physical slots per rank for ``num_slots`` active slots."""
    if ep_size <= 0 or num_experts % ep_size:
        raise ValueError(
            f"GenEPLB needs num_experts ({num_experts}) divisible by EP size ({ep_size})"
        )
    if not num_experts < num_slots:
        raise ValueError(
            f"GenEPLB num_slots ({num_slots}) must exceed num_experts ({num_experts}): "
            "it counts every expert once plus at least one replica"
        )
    replicas_per_rank = -(-(num_slots - num_experts) // ep_size)
    return num_experts // ep_size + 2 * replicas_per_rank


@dataclass(frozen=True)
class Geometry:
    experts: int
    ranks: int
    local_slots: int  # M
    active_slots: int  # num_slots of the yaml: E + active replicas
    top_k: int

    @property
    def homes(self) -> int:  # H
        return self.experts // self.ranks

    @property
    def slots(self) -> int:
        return self.ranks * self.local_slots

    def home_slot(self, expert: int) -> int:
        return (expert // self.homes) * self.local_slots + expert % self.homes

    def home_layout(self) -> list:
        """The start layout: every expert at its home slot, every spare slot FREE (-1)."""
        h, m = self.homes, self.local_slots
        return [r * h + j if j < h else -1 for r in range(self.ranks) for j in range(m)]
