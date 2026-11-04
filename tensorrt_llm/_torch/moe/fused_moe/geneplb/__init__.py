# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB: ``moe_config.load_balancer {num_slots, policy: adaptive[, cost_model_b: {source: prior}]}``."""

__all__ = ["GenEplbMoeLoadBalancer"]


def __getattr__(name):
    if name == "GenEplbMoeLoadBalancer":
        from .balancer import GenEplbMoeLoadBalancer

        return GenEplbMoeLoadBalancer
    raise AttributeError(name)
