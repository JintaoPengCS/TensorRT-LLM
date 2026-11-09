# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GenEPLB on TRT-LLM's load-balancer framework (``moe_config.load_balancer.policy: adaptive``).

Two levels, renewed together: replicas (which spare slot holds a copy of which busy expert; changed only
when the predicted saving pays for the copy) and shares (how a replicated expert's tokens split over its
copies; re-solved from the measured load at every renewal). Per decode step the captured route kernel
maps expert ids to slots through per-layer tables and a statistics kernel counts the real rows each
expert receives. Every renewal: a window of decode replays is counted, every rank's counts are gathered,
every rank's planner worker computes the same decision, shares are written between two replays, and new
replicas are copied GPU to GPU (NCCL point-to-point on a side stream) into slots no rank routes to, then
published once every rank's copies have landed.
"""

from __future__ import annotations

import functools
import hashlib
import os
from typing import Any, Optional

import numpy as np
import torch

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned
from tensorrt_llm.logger import logger

from ..moe_load_balancer import BaseMoeLoadBalancer
from . import costmodel, native, planner, shares
from .geometry import Geometry
from .worker import Worker

# The probe window (decode replays counted per renewal), the opening's (the first decision reads the first
# replay) and the minimum renewal interval (1 + the window + 2 from gather to apply + 1).
WINDOW = 8
OPENING_WINDOW = 1
RENEWAL_FLOOR = 12
# Dormant (no layer holds a replica and the last gate admitted none): one counted window per this many replays, main's
# plain routing (no padding spread) and an idle statistics kernel (enable word 0) in between.
DORMANT_PERIOD = 256
# Installs: replays between the last rank's ready vote and the agreed first copy step, and the
# layers whose replicas are copied between two consecutive replays.
INSTALL_MARGIN = 8
LAYERS_PER_COPY_STEP = 8
# Copy time an install may add to one replay: a large install is spread over more replays when copies are slow.
COPY_BUDGET_MS = 4.0
# Live serving (requests keep arriving after the first renewal): a placement serves until the next install, not just
# the requests in flight, so installs are priced over at least LIVE_HORIZON replays instead of the in-flight requests'
# remaining steps (the fixed-batch rule), and a layer installs at most once per LIVE_COOLDOWN renewals (~500 replays; no
# churn on noisy windows). Counted in renewal decisions, which every rank makes in the same sequence: a window's place
# in replays is rank-local (each rank opens its next window after it applied the decision, at its own replay).
LIVE_HORIZON = 2048
LIVE_COOLDOWN = 32
# Live serving below the cost model's measured tiers (T < 32 real decode rows of one replay on the busiest rank; the
# prior there is the T = 32 row, never measured): a renewal re-solves the shares and installs no new replica.
# Measured at T 4..24 (GEN DEP32, 388 lanes, ~2.6 requests per rank): ~1400 installs per run moved ~115k replicas and
# cost +1.7..6 % mean GEN step against no balancer, while a run that installed nothing matched it.
# TLLM_GENEPLB_LIVE_MIN_T overrides the floor (0: off).
LIVE_MIN_T = int(os.environ.get("TLLM_GENEPLB_LIVE_MIN_T", "32"))
# ---------------------------------------------------------------------------------------------------------------------------
# RUN-TIME CONTROL SWITCHES (policy and diagnostics). Without TLLM_GENEPLB_CTL_FILE and TLLM_GENEPLB_FORCE_LIVE
# every path below is the shipped one (route launches without the control word, the shipped gate, gathered tail and log line).
#   TLLM_GENEPLB_FORCE_LIVE=1: every renewal is priced as live serving from the first one (a fill / lockstep gen-only run sees
#     no arrival after the first renewal, so the shipped rule stays on the fixed-batch path and never applies the low-T guard).
#   TLLM_GENEPLB_CTL_FILE=<path>: every rank re-reads the file when it packs its renewal statistics (on a change of its mtime /
#     size / inode: write it atomically, tmp + rename); the codes ride in the gathered tail and every rank takes the MAX over
#     ranks, so every rank makes the same decision (a file change mid-gather cannot split the plan digest). Tokens 'key=value'
#     (CTL_SPEC; unknown keys and malformed values are ignored, '#' starts a comment, no file = every default = shipped). The
#     per-replay fields (pad, identity) reach the captured route kernels through the device word t_ctl, written on the forward
#     stream between two replays together with the renewal's tables; each rank mirrors its applied fields into
#     <CTL_FILE>.rank<r> on a change, and rank 0's renewal line carries them with the real / padded T and the gate's sums.
#   TLLM_GENEPLB_NANPROBE=1 (with the control file): a captured count of non-finite router weights over every row (padding
#     rows included) of every staged layer, summed over ranks in the renewal line and the status files.
FORCE_LIVE = os.environ.get("TLLM_GENEPLB_FORCE_LIVE", "0") == "1"
CTL_FILE = os.environ.get("TLLM_GENEPLB_CTL_FILE", "")
NANPROBE = bool(CTL_FILE) and os.environ.get("TLLM_GENEPLB_NANPROBE", "0") == "1"
# (field, choices text -> code | number type, default code). Code -1 is the shipped value: min_t LIVE_MIN_T, gate_tau the prior
# row's tau ('prior'), gate_horizon 'fixed' (the shipped N_rem: max(N_rem, LIVE_HORIZON) once live).
CTL_SPEC = (
    ("install", {"normal": 0, "freeze": 1}, 0),  # freeze: shares only, no new replica
    ("min_t", int, -1),  # the low-T guard's floor (LIVE_MIN_T override)
    ("guard_t", {"real": 0, "padded": 1}, 0),  # the T the guard tests: real rows, or the padded rows MegaMoE runs
    ("pad", {"spread": 0, "drop": 1, "stock": 2}, 0),  # padding rows: shipped spread | sentinel + weight 0 | own ids
    ("identity", {"0": 0, "1": 1}, 0),  # every id to its home slot (fast path), no installs, held replicas unused
    ("gate_tau", float, -1),  # the gate's LSE tau in us (0: the hard max); 'prior': the prior row's
    ("gate_margin_us", float, 0),  # gain per layer-step required on top of the copy-basis test
    ("gate_hyst", int, 1),  # consecutive renewals a layer's plan (same digest) must pass before it installs
    ("gate_horizon", int, -1),  # the payback horizon in replays; 'fixed': the shipped one
    ("copy", {"nccl": 0, "ce": 1}, 0),  # reserved for the copy prototype (this build copies with NCCL either way)
    # The P5 / P6 policies at run time: the decide request carries them (with pad) and
    # the planner worker uses them instead of TLLM_GENEPLB_OBJECTIVE / TLLM_GENEPLB_PLACEMENT (planner.py _run_controls);
    # placement=flex is inert (lpt) unless objective=ef and pad=drop. Cost model B only (mode C ignores both).
    ("objective", {"shipped": 0, "ef": 1}, 0),  # shipped cost model B on the window means | P5 eF (expected-max) objective
    ("placement", {"lpt": 0, "flex": 1}, 0),  # shipped rows-LPT | P6 flex options (every rank's 2 busiest homes)
)
CTL_DEFAULT = tuple(float(d) for _, _, d in CTL_SPEC)
CTL_PAD_WORD = {"spread": 0, "drop": 1, "stock": 2}  # l3b_route_kernels.cu LB_PAD_* (control word bits 0-1)
CTL_IDENTITY_BIT = 4  # l3b_route_kernels.cu LB_CTL_IDENTITY


def parse_ctl(text: str):
    """(codes in CTL_SPEC order, the ignored tokens of known keys) of a control file's text."""
    vals, bad = list(CTL_DEFAULT), []
    index = {name: i for i, (name, _, _) in enumerate(CTL_SPEC)}
    for line in text.splitlines():
        for tok in line.split("#", 1)[0].split():
            key, sep, value = tok.partition("=")
            i = index.get(key) if sep else None
            if i is None:
                continue  # not 'key=value' of a known key: ignored
            kind = CTL_SPEC[i][1]
            try:
                if isinstance(kind, dict):
                    code = float(kind[value])
                elif (key, value) in (("gate_horizon", "fixed"), ("gate_tau", "prior"), ("min_t", "default")):
                    code = -1.0
                else:
                    code = float(kind(value))
                    if not (code >= (1.0 if key in ("gate_hyst", "gate_horizon") else 0.0) and code < 1e15):
                        raise ValueError(value)
            except (KeyError, ValueError):
                bad.append(tok)
                continue
            vals[i] = code
    return tuple(vals), bad


def ctl_fields(codes) -> dict:
    """The control fields of (the MAX over ranks of) a code vector, with the shipped values resolved."""
    c = {name: float(v) for (name, _, _), v in zip(CTL_SPEC, codes)}
    return dict(
        install="freeze" if c["install"] >= 1 else "normal",
        min_t=LIVE_MIN_T if c["min_t"] < 0 else int(c["min_t"]),
        guard_t="padded" if c["guard_t"] >= 1 else "real",
        pad=("spread", "drop", "stock")[min(2, max(0, int(c["pad"])))],
        identity=1 if c["identity"] >= 1 else 0,
        gate_tau=None if c["gate_tau"] < 0 else c["gate_tau"],
        gate_margin_us=max(0.0, c["gate_margin_us"]),
        gate_hyst=max(1, int(c["gate_hyst"])),
        gate_horizon=None if c["gate_horizon"] < 1 else int(c["gate_horizon"]),
        copy="ce" if c["copy"] >= 1 else "nccl",
        objective="ef" if c["objective"] >= 1 else "shipped",
        placement="flex" if c["placement"] >= 1 else "lpt",
    )


def ctl_text(f: dict) -> str:
    """The canonical control line of the fields (a control file holding it parses back to the same fields)."""
    return (
        f"install={f['install']} min_t={f['min_t']} guard_t={f['guard_t']} pad={f['pad']} identity={f['identity']} "
        f"gate_tau={'prior' if f['gate_tau'] is None else repr(f['gate_tau'])} gate_margin_us={f['gate_margin_us']!r} "
        f"gate_hyst={f['gate_hyst']} gate_horizon={'fixed' if f['gate_horizon'] is None else f['gate_horizon']} "
        f"copy={f['copy']} objective={f['objective']} placement={f['placement']}"
    )
# After the final warmup: bytes exchanged with every peer to connect it, and the layers the startup copy probe times
# (the shipped startup probe: the first gate prices installs with a measured copy cost).
CONNECT_BYTES = 4 << 20
PROBE_LAYERS = 2
# The install gate prices copies with the median of the last COPY_HISTORY measured copy times (startup probe
# included): copy cost is only re-measured by a published install, so a single outlier copy must not price every
# later install out (a last-value estimate froze installs for good after one 36 ms/layer copy).
COPY_HISTORY = 9
# Replica copy transport. TLLM_GENEPLB_COPY = nccl (default: the shipped NCCL
# batch_isend_irecv) | ce (copy-engine pull through a symmetric staging buffer, fastcopy.py; no NCCL P2P
# pre-connect) | both (both armed, NCCL agreed at start). With both armed, a 'copy=ce|nccl' field in
# TLLM_GENEPLB_CTL_FILE is this rank's request: every renewal's ready vote carries it (MAX over ranks: CE wins)
# and the agreed transport prices the gate and copies from the next renewal on, the same on every rank.
# TLLM_GENEPLB_COPY_BUDGET_MS overrides COPY_BUDGET_MS (the copy time one replay may absorb).
COPY_NCCL, COPY_CE = 0, 1
COPY_MODE_NAMES = ("nccl", "ce")
COPY_ARMED = {"nccl": ((COPY_NCCL,), COPY_NCCL), "ce": ((COPY_CE,), COPY_CE), "both": ((COPY_NCCL, COPY_CE), COPY_NCCL)}
COPY_ARM = os.environ.get("TLLM_GENEPLB_COPY", "nccl").strip().lower()
if COPY_ARM not in COPY_ARMED:
    raise ValueError(f"TLLM_GENEPLB_COPY={COPY_ARM!r}: one of {sorted(COPY_ARMED)}")
COPY_CTL_FILE = os.environ.get("TLLM_GENEPLB_CTL_FILE", "")
COPY_BUDGET_MS = float(os.environ.get("TLLM_GENEPLB_COPY_BUDGET_MS", COPY_BUDGET_MS))


class GenEplbLayer:
    """One MoE layer's handle: TRT-LLM calls ``route``; the balancer owns its tables and slots."""

    produces_replica_plan = False

    def __init__(
        self,
        owner: "GenEplbMoeLoadBalancer",
        index: int,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        repeat_count: int,
    ):
        self.owner = owner
        self.index = index
        self.expert_count = expert_count
        self.top_k = top_k
        self.local_slots = slot_count_per_rank
        self.repeat_count = repeat_count
        self.moe = None
        self.initial_assignments = None

    # Static at the layer level: TRT-LLM then only calls route(); no host weight pool, no stat hooks.
    def is_static_routing(self) -> bool:
        return True

    def is_dynamic_routing(self) -> bool:
        return False

    def need_load_shared_weights(self) -> bool:
        return False

    def get_repeat_count(self) -> int:
        return self.repeat_count

    def set_initial_weight_assignments(self, assignments) -> None:
        self.initial_assignments = list(assignments)

    def bind_moe(self, moe) -> None:
        self.moe = moe

    def route(
        self, token_selected_experts: torch.Tensor, use_dp: bool, overlap_fn=None
    ) -> torch.Tensor:
        if overlap_fn is not None:
            overlap_fn()
        return self.owner.route(self, token_selected_experts)


class GenEplbMoeLoadBalancer(BaseMoeLoadBalancer):
    """Model-level owner of the GenEPLB layers, their device tables and the renewal pipeline."""

    def __init__(self, *, ep_rank: int, ep_size: int, config: Any, mapping: Optional[Any] = None):
        self.ep_rank, self.ep_size = int(ep_rank), int(ep_size)
        self.config, self.mapping = config, mapping
        self.mode = "B" if config.cost_model_b is not None else "C"
        self.active_slots = int(config.num_slots)
        self.layers: list[GenEplbLayer] = []
        self.next_repeat: Optional[int] = None
        self._previous_balancer = None
        self.enable_statistic = False
        self.enable_update_weights = False
        self.finalized = False
        self.is_shutdown = False

    def __repr__(self) -> str:
        return (
            f"GenEplbMoeLoadBalancer(mode={self.mode}, ep_rank={self.ep_rank}, ep_size={self.ep_size}, "
            f"num_slots={self.active_slots})"
        )

    # ------------------------------------------------------------------ framework interface
    def is_static_routing(self) -> bool:
        return False  # the iteration context drives end_iter

    def balances(self, moe, model_config) -> bool:
        """The target model's routed MoE layers are balanced; the MTP / draft MoE layers (layer index past the
        body) keep TRT-LLM's plain expert partition."""
        body = getattr(getattr(model_config, "pretrained_config", None), "num_hidden_layers", None)
        return body is None or int(moe.layer_idx) < int(body)

    def set_repeated_for_next_layer(self, repeated_count: int) -> None:
        self.next_repeat = int(repeated_count)

    def add_layer(
        self,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> GenEplbLayer:
        layer = GenEplbLayer(
            self, len(self.layers), expert_count, top_k, slot_count_per_rank, self.next_repeat or 1
        )
        self.next_repeat = None
        self.layers.append(layer)
        return layer

    def register_weight_slots_after_to_cuda(self) -> None:
        pass

    def set_warm_up_iter_count(self, iter_count: int) -> None:
        pass

    def start_iter(self) -> None:
        pass

    def finalize_model(self) -> None:
        """Tables at the home layout, kernels, per-slot weight views, communicators and the planner worker."""
        first = self.layers[0]
        if any(
            (li.expert_count, li.local_slots, li.top_k)
            != (first.expert_count, first.local_slots, first.top_k)
            for li in self.layers
        ):
            raise ValueError("GenEPLB needs one slot geometry for every balanced layer")
        geo = self.geo = Geometry(
            experts=first.expert_count,
            ranks=self.ep_size,
            local_slots=first.local_slots,
            active_slots=self.active_slots,
            top_k=first.top_k,
        )
        self.device = torch.device("cuda", torch.cuda.current_device())
        L = self.num_layers = len(self.layers)
        home = geo.home_layout()
        self.layouts = [list(home) for _ in range(L)]
        self.retired = {}  # layer -> slots its last install freed (no new copy lands there for one renewal)
        count, start, slots, cumw = planner.tables(home, shares.even_weights(home), geo.experts)
        self.h_count, self.h_start, self.h_slots, self.h_cumw = (
            np.stack([a] * L) for a in (count, start, slots, cumw)
        )
        dev = self.device
        self.t_count = torch.from_numpy(self.h_count).to(dev)
        self.t_start = torch.from_numpy(self.h_start).to(dev)
        self.t_slots = torch.from_numpy(self.h_slots).to(dev)
        self.t_cumw = torch.from_numpy(self.h_cumw).to(dev)
        self.t_wmode = torch.zeros(L, dtype=torch.int32, device=dev)
        self.valid_rows = torch.full((1,), native.VALID_ALL, dtype=torch.int32, device=dev)
        self.enable = torch.zeros(1, dtype=torch.int32, device=dev)
        self._valid_host, self._enable_host = native.VALID_ALL, 0
        # the control word of the CTL route launches (cost model B's kernel only) and the NaN probe's counter.
        self.t_ctl = torch.zeros(1, dtype=torch.int32, device=dev) if CTL_FILE and self.mode == "B" else None
        self._nan_acc = torch.zeros(1, dtype=torch.int32, device=dev) if NANPROBE else None
        self.stats = native.StatsKernel(L, geo.experts, geo.ranks, dev, geo.top_k, slots=geo.slots)
        self.kernel = native.RouteKernel(
            self.mode, geo.experts, geo.slots, geo.ranks, self.ep_rank, dev, geo.top_k
        )
        self.out_dtype = (
            torch.int64 if self.mode == "B" else torch.int32
        )  # as the shipped arms (B's int64 cut)
        self._rows_ref = None
        self._held = {}
        # B's stage cut: a MegaMoE layer routes inside its run_moe, straight into the launch bucket's staging
        # (after its input is quantized there), instead of a route launch plus four staging fills / copies.
        self.staged = set()
        if self.mode == "B":
            for layer in self.layers:
                backend = getattr(layer.moe, "backend", None)
                if hasattr(backend, "_route_stager"):
                    backend._route_stager = functools.partial(self._stage, layer)
                    self.staged.add(layer.index)
        self.slot_tensors = [self._slot_tensors(layer) for layer in self.layers]
        self.b_layers = {
            li.index: self.mode == "B" for li in self.layers
        }  # B prices every balanced (body) layer
        self._init_pipeline()
        self.finalized = True
        if self.ep_rank == 0:
            names = ",".join(n for n, _ in self.slot_tensors[0])
            logger.info(
                f"GenEPLB ready: mode {self.mode}, {L} layers, E {geo.experts}, EP {geo.ranks}, "
                f"{geo.local_slots} slots/rank ({geo.homes} homes), active {geo.active_slots}, slot planes [{names}], "
                f"staged {len(self.staged)} layers"
            )

    def _slot_tensors(self, layer: GenEplbLayer) -> list:
        """Every per-slot tensor of the layer's backend (first dim = local slots): what an install copies."""
        module = getattr(layer.moe, "backend", None) or layer.moe
        out, seen = [], set()
        for name, t in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)
        ):
            t = t.data if isinstance(t, torch.nn.Parameter) else t
            if (
                t.is_cuda
                and t.dim() >= 1
                and t.shape[0] == layer.local_slots
                and t.numel() > 0
                and t.data_ptr() not in seen
            ):
                seen.add(t.data_ptr())
                out.append((name, t))
        if not out:
            raise RuntimeError(
                f"GenEPLB: layer {layer.index} has no per-slot weight tensors to copy"
            )
        return sorted(out)

    # ------------------------------------------------------------------ the renewal pipeline
    def _init_pipeline(self) -> None:
        import torch.distributed as dist

        from tensorrt_llm._utils import mpi_comm

        # One communicator per collective stream: each is called once per renewal / install on every rank, so
        # their orders agree even though ranks reach them at different steps.
        session = mpi_comm()
        if session.Get_size() != self.ep_size:
            raise RuntimeError(
                f"GenEPLB: the MPI session has {session.Get_size()} ranks, EP is {self.ep_size}"
            )
        self.comm_gather, self.comm_ready, self.comm_done = (
            session.Dup(),
            session.Dup(),
            session.Dup(),
        )
        if not (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() == self.ep_size
        ):
            raise RuntimeError(
                "GenEPLB copies replicas with torch.distributed NCCL: WORLD must equal EP"
            )
        self.copy_group = dist.new_group(ranks=list(range(self.ep_size)), backend="nccl")
        # Create the NCCL communicator now, with every rank present: a later install may give a rank no copy.
        dist.all_reduce(torch.zeros(1, device=self.device), group=self.copy_group)
        self._probe = torch.zeros(
            1 + self.ep_size, device=self.device
        )  # _copy_due's connection probes
        cache = os.environ.get(
            "TRTLLM_GENEPLB_CACHE_DIR", os.path.expanduser("~/.cache/tensorrt_llm/geneplb")
        )
        self.worker = Worker(cache)
        L, E = self.num_layers, self.geo.experts
        self.snap_load = torch.empty((L, E), dtype=torch.float32, pin_memory=prefer_pinned())
        self.snap_steps = torch.empty((L,), dtype=torch.int32, pin_memory=prefer_pinned())
        self.forwards = 0  # forwards since start-up (the same on every rank)
        self.replays = 0  # counted decode replays
        self.window_start = 0  # replay index of the next window's first replay
        self.window_end = None  # replay index after which the window is read
        self.window_open = False
        self.window_rows = 0  # the largest real decode rows of one replay in the open window
        self.snap_event = None
        self.gather = None  # (request, recv buffer, local size, send buffer)
        self.decision = None  # True while the worker computes
        self.ready_vote = (
            None  # installs waiting for every rank's decision: it agrees the first copy step
        )
        self.installs = []  # in flight: plans, agreed first copy step, chunks issued, events, the done vote
        self.pending_writes = False
        self.renewals = self.install_count = 0
        self.c_gpu_ms = None
        self.copy_ms_history = []  # measured copy ms/layer (max over ranks), newest last
        # Copy transport: the armed ones (same env on every rank), the agreed one (prices the gate, copies the next
        # install), one measured-copy history per transport, this rank's CTL request, the CE copier.
        self._copy_armed, self._copy_mode = COPY_ARMED[COPY_ARM]
        self._copy_default = self._copy_mode
        self._copy_hist = {COPY_NCCL: self.copy_ms_history}
        self._copy_ctl = (None, self._copy_mode)  # (CTL file mtime, requested transport)
        self._ce = None  # fastcopy.CeCopier, armed in after_warmup
        self.connected = False
        # No replica held and the last gate admitted none (set by each renewal).
        self.dormant = False
        self.live = FORCE_LIVE  # requests arrived after the first renewal on some rank (from the gathered counts)
        self.arrivals = 0  # this rank's requests that started decoding since the last gather
        self.decisions = 0  # renewal decisions requested (the same sequence on every rank)
        self.installed_at = {}  # layer -> the decision that installed it
        self.budgets = {}  # request id -> tokens it had produced at the previous step
        self.accept = None  # EMA (accepted tokens, request steps), weight 0.95 per step
        self.rem_sum, self.rem_count = 0.0, 0
        self.draft_len = 0
        self.lifetime = None  # EWMA of finished requests' output tokens (None until one finishes)
        # the control channel: the codes read last and the file's change key, the applied status (None before the
        # first decision), the device word / drop flag in force and due at the next table write, the window's padded rows, the
        # gate's hysteresis state (layer -> (plan digest, consecutive passes)), the NaN probe's host snapshot.
        self._ctl_codes, self._ctl_key, self._ctl_applied = CTL_DEFAULT, None, None
        self._ctl_word, self._ctl_word_next, self._drop_words, self._drop_next = 0, 0, False, False
        self.window_prows = 0
        self._hyst = {}
        self._snap_nan = torch.zeros(1, dtype=torch.int32, pin_memory=prefer_pinned()) if NANPROBE else None

    def begin_step(self, scheduled_requests, draft_len: int, graph: bool, is_dummy: bool) -> None:
        """Before a forward: the valid-rows / enable words, the writes due now, the window, the budgets."""
        if not self.finalized or torch.cuda.is_current_stream_capturing():
            return
        # Every rank runs the same forwards (attention DP keeps an idle rank in step with dummy requests):
        # installs are scheduled on this count, counted decode replays drive the statistics windows.
        self.forwards += 1
        counted = bool(graph and self.enable_statistic and self.enable_update_weights)
        gen = [r for r in scheduled_requests.generation_requests if not _is_dummy(r)]
        self.draft_len = int(draft_len)
        rows = len(gen) * (1 + self.draft_len)
        if counted:
            self._track_budgets(gen)
            if self.replays == self.window_start and self.window_end is None:
                self.stats.reset()  # the window's EMA starts from zero, stream-ordered before this replay
                n = OPENING_WINDOW if self.renewals == 0 else WINDOW
                self.window_end, self.window_open, self.window_rows = self.replays + n, True, 0
                self.window_prows = 0
            if self.window_open:
                self.window_rows = max(self.window_rows, rows)
                if CTL_FILE:  # the rows MegaMoE runs: graph padding and attention-DP dummies included
                    self.window_prows = max(
                        self.window_prows,
                        len(scheduled_requests.generation_requests) * (1 + self.draft_len),
                    )
        # The statistics kernel counts only a window's replays; a dormant balancer routes like main between its
        # windows (no padding spread), so its words stay put and no step writes them.
        # pad=drop: the real rows on every counted replay, dormant ones included (padding always dropped).
        window = counted and self.window_open
        spread = window or (counted and (not self.dormant or self._drop_words))
        self._set_words(rows if spread else native.VALID_ALL, 1 if window else 0)
        self._advance()
        self._copy_due()
        if self.pending_writes:
            self._write_tables()
        if counted:
            self.replays += 1

    def end_iter(self) -> None:
        """After a forward is enqueued: snapshot the window once its last replay is on the stream."""
        if not self.finalized or torch.cuda.is_current_stream_capturing():
            return
        if self.window_open and self.replays == self.window_end:
            self.window_open = False
            self.snap_load.copy_(self.stats.load, non_blocking=True)
            self.snap_steps.copy_(self.stats.steps, non_blocking=True)
            if self._snap_nan is not None:
                self._snap_nan.copy_(self._nan_acc, non_blocking=True)
            self.snap_event = torch.cuda.Event()
            self.snap_event.record()

    def _advance(self) -> None:
        """Move the renewal pipeline forward without blocking the serving thread (rank-local timing:
        every rank makes the same collective calls in the same order and computes the same decision)."""
        if self.snap_event is not None and self.gather is None and self.snap_event.query():
            self._start_gather()
        if (
            self.gather is not None
            and not self.installs
            and self.ready_vote is None
            and self.gather[0].Test()
        ):
            self._start_decision()  # after every earlier install published: the same layouts on every rank
        if self.decision is not None:
            got = self.worker.poll()
            if got is not None:
                self._apply_decision(got[1])
        if self.ready_vote is not None and self.ready_vote[0].Test():
            _, (_, out), plans, chunk = self.ready_vote
            self.ready_vote = None
            if (
                out[1] != -out[3] or out[2] != -out[4]
            ):  # MAX != MIN of the decision index or the plan digest
                raise RuntimeError(
                    f"GenEPLB rank {self.ep_rank}: the ranks' renewal decisions differ (decision "
                    f"{-out[3]:.0f}..{out[1]:.0f}, plan digest {-out[4]:.0f}..{out[2]:.0f}); copying now would "
                    "deadlock or copy the wrong experts"
                )
            if plans:
                first = int(out[0]) + INSTALL_MARGIN
                self.installs.append(
                    dict(plans=plans, first=first, issued=0, events=[], vote=None, chunk=chunk,
                         copy=self._copy_mode)
                )
            self._set_copy_mode(int(out[5]))  # the agreed request (MAX over ranks), from the next renewal on
        self._poll_installs()

    def route(self, layer: GenEplbLayer, ids: torch.Tensor) -> torch.Tensor:
        """Map the layer's expert ids to slots; inside a CUDA-graph capture also record the statistics: every
        layer's ids are held and ONE fused count kernel over all layers is captured after the last layer (the
        shipped 'fused_ptr' mode). Eager steps count nothing."""
        staged = layer.index in self.staged
        if ids.numel() == 0:
            return ids if staged else ids.to(self.out_dtype)
        if layer.index == 0 or self._rows_ref is None:
            self._rows_ref = int(
                ids.shape[0]
            )  # the main pass's rows: the padding rule spreads against them
        if torch.cuda.is_current_stream_capturing():
            if layer.index == 0:
                self._held = {}
            self._held.setdefault(layer.index, ids)  # the first invocation per forward
            if layer.index == self.num_layers - 1 and len(self._held) == self.num_layers:
                held = [self._held[li] for li in range(self.num_layers)]
                self.stats.record_fused(
                    [int(t.data_ptr()) for t in held],
                    [int(t.shape[0]) for t in held],
                    self.valid_rows,
                    self.enable,
                )
                self._held = {}  # the launch is recorded; stream order protects the reads
        if staged:
            return ids  # expert ids: the backend's run_moe routes them into its staging (_stage)
        return self.kernel(
            ids, self._tables(layer.index), self._rows_ref, self.valid_rows, self.out_dtype, ctl=self.t_ctl
        )

    def _stage(self, layer: GenEplbLayer, ids, weights, idx_out, w_out) -> None:
        if self._nan_acc is not None and weights.numel():  # non-finite router weights, every row (capturable)
            self._nan_acc += (~torch.isfinite(weights)).sum(dtype=torch.int32)
        self.kernel.stage(
            ids,
            self._tables(layer.index),
            self._rows_ref or 1,
            self.valid_rows,
            idx_out,
            weights,
            w_out,
            ctl=self.t_ctl,
        )

    def _tables(self, li: int):
        return native.LayerTables(
            expert_count=self.t_count[li],
            expert_start=self.t_start[li],
            slot_ids=self.t_slots[li],
            cumw=self.t_cumw[li],
            wmode=self.t_wmode[li : li + 1],
        )

    def _set_words(self, rows: int, enable: int) -> None:
        if rows != self._valid_host:
            self.valid_rows.fill_(rows)
            self._valid_host = rows
        if enable != self._enable_host:
            self.enable.fill_(enable)
            self._enable_host = enable

    def _track_budgets(self, gen) -> None:
        """Remaining output budgets and accepted tokens per request per step: the shipped N_rem inputs. Live
        traffic: once requests have finished, a budget counts at most the EWMA of finished requests' outputs
        (a client max_tokens far above the real lengths would otherwise inflate the horizon)."""
        app = rc = 0
        seen, rem = {}, 0.0
        if self.renewals:
            self.arrivals += sum(r.py_request_id not in self.budgets for r in gen)
        for r in gen:
            rid = r.py_request_id
            produced = int(r.get_num_tokens(0)) - int(r.py_prompt_len)
            last = self.budgets.pop(rid, None)
            if last is not None and produced > last:
                app += produced - last
                rc += 1
            seen[rid] = produced
            budget = int(r.py_max_new_tokens)
            if self.lifetime is not None:
                budget = min(budget, max(produced + 1, int(self.lifetime)))
            rem += max(0, budget - produced)
        for produced in self.budgets.values():  # requests that left since the previous step
            self.lifetime = (
                produced if self.lifetime is None else 0.95 * self.lifetime + 0.05 * produced
            )
        self.budgets = seen
        if rc:
            a = self.accept
            self.accept = (
                (app, rc) if a is None else (0.95 * a[0] + 0.05 * app, 0.95 * a[1] + 0.05 * rc)
            )
        self.rem_sum, self.rem_count = rem, len(gen)

    def _start_gather(self) -> None:
        acc = self.accept or (0.0, 0.0)
        extra = []
        if CTL_FILE:  # the control codes, the window's padded rows and the NaN count ride in the tail
            self._ctl_read()
            nan = float(self._snap_nan[0]) if self._snap_nan is not None else 0.0
            extra = [*self._ctl_codes, float(self.window_prows), nan]
        local = np.concatenate(
            [
                self.snap_load.numpy().astype(np.float64).ravel(),
                self.snap_steps.numpy().astype(np.float64),
                np.array(
                    [self.rem_sum, self.rem_count, acc[0], acc[1], self.window_rows, self.arrivals]
                    + extra,
                    dtype=np.float64,
                ),
            ]
        )
        recv = np.empty(self.ep_size * local.size, dtype=np.float64)
        self.gather = (
            self.comm_gather.Iallgather(local, recv),
            recv,
            local.size,
            local,
        )  # buffers live until done
        self.snap_event = None
        self.arrivals = 0

    def _start_decision(self) -> None:
        _, recv, n, _ = self.gather
        self.gather = None
        L, E = self.num_layers, self.geo.experts
        rows = recv.reshape(self.ep_size, n)
        # As shipped: every expert's statistics EMA summed over ranks (the totals) and one EMA scale per layer from its
        # launches since the window reset (the same on every rank) -- the same reduction of the same gathered bytes on
        # every rank.
        totals = rows[:, : L * E].reshape(self.ep_size, L, E).sum(axis=0)
        steps = rows[:, L * E : L * E + L].max(axis=0)
        scales = [(1.0 - 0.95) / (1.0 - 0.95 ** int(st)) if st > 0 else 0.0 for st in steps]
        tail = rows[:, L * E + L :]
        rem_sum, count, app, rc = (float(tail[:, i].sum()) for i in range(4))
        tokens = int(tail[:, 4].max())  # T: the largest real decode rows of one replay on any rank
        ctl = None
        if CTL_FILE:  # the MAX over ranks of every rank's codes: the same fields on every rank
            k = len(CTL_SPEC)
            ctl = ctl_fields(tail[:, 6 : 6 + k].max(axis=0))
            ctl["T_pad"] = int(tail[:, 6 + k].max())  # padded T: the largest padded rows of one replay on any rank
            ctl["nan"] = int(tail[:, 7 + k].sum())  # non-finite router weights so far, summed over ranks
        if tokens == 0 or not totals.any():
            # An idle window (no real decode rows on any rank): nothing to renew.
            self._next_window()
            return
        accepted = (
            (app / rc) if rc > 0 else float(self.draft_len + 1)
        )  # before any step is measured: the bound
        n_rem = (rem_sum / accepted / count) if count > 0 and accepted > 0 else None
        # Live once any rank saw a request start decoding after the first renewal (the same gathered counts on every
        # rank, so every rank prices the same way).
        self.live = self.live or bool(tail[:, 5].sum() > 0)
        self.decisions += 1
        frozen = []
        if self.live:
            n_rem = max(n_rem or 0.0, float(LIVE_HORIZON))
            frozen = [
                li for li, at in self.installed_at.items() if self.decisions - at < LIVE_COOLDOWN
            ]
        geo = self.geo
        b_params = costmodel.prior_params(max(1, tokens)) if self.mode == "B" else None
        install = not (self.live and tokens < LIVE_MIN_T)
        self._last = dict(n_rem=n_rem, tokens=tokens, install=install)
        if ctl is not None:  # the guard on the real or padded T against min_t, freeze / identity, the horizon
            t_guard = ctl["T_pad"] if ctl["guard_t"] == "padded" else tokens
            install = not (self.live and t_guard < ctl["min_t"])
            why = "" if install else f"{ctl['guard_t']} T {t_guard} < min_t {ctl['min_t']}"
            if ctl["identity"] or ctl["install"] == "freeze":
                install, why = False, ("identity" if ctl["identity"] else "freeze")
            if ctl["gate_horizon"] is not None and n_rem is not None:
                n_rem = float(ctl["gate_horizon"])
            self._last = dict(
                n_rem=n_rem, tokens=tokens, install=install, ctl=ctl, t_guard=t_guard, why=why
            )
        self.worker.request(
            "decide",
            dict(
                geo=dict(
                    experts=geo.experts,
                    ranks=geo.ranks,
                    local_slots=geo.local_slots,
                    active_slots=geo.active_slots,
                    top_k=geo.top_k,
                ),
                mode=self.mode,
                b_params=b_params,
                gate_model=b_params,
                b_layers=self.b_layers,
                totals={li: totals[li].tolist() for li in range(L)},
                scales={li: scales[li] for li in range(L)},
                layouts={li: self.layouts[li] for li in range(L)},
                retired=self.retired,
                n_rem=n_rem,
                frozen=frozen,
                c_gpu_ms=self.c_gpu_ms,
                total_active=geo.active_slots,
                cap=geo.local_slots - geo.homes,
                install=install,
                max_layers=L,
                **(
                    {}
                    if ctl is None
                    else dict(
                        ctl=dict(
                            gate_tau=ctl["gate_tau"],
                            gate_margin_us=ctl["gate_margin_us"],
                            gate_hyst=ctl["gate_hyst"],
                            hyst=self._hyst,
                            objective=ctl["objective"],  # (p56ctl) the planner's run-time P5 / P6 switch
                            placement=ctl["placement"],
                            pad=ctl["pad"],  # placement=flex is inert unless pad=drop
                        )
                    )
                ),
            ),
        )
        self.decision = True

    def _apply_decision(self, reply: dict) -> None:
        self.decision = None
        self.renewals += 1
        self.retired = {}  # a full renewal since the last install: its freed slots are unrouted everywhere
        for li, arrays in reply["tables"].items():
            self._stage_layer(li, arrays)
        self.pending_writes = True
        # Every renewal votes, with its index and a digest of its plan: the agreed copy step (MAX of the forwards
        # + INSTALL_MARGIN) only pairs identical plans of the same renewal; any divergence raises before a copy.
        from mpi4py import MPI

        for ins in reply["installs"]:
            self.installed_at[ins["layer"]] = self.decisions
        c = self.c_gpu_ms
        chunk = (
            max(1, min(LAYERS_PER_COPY_STEP, int(COPY_BUDGET_MS / c)))
            if c
            else LAYERS_PER_COPY_STEP
        )
        plan = [(ins["layer"], [tuple(cp) for cp in ins["copies"]]) for ins in reply["installs"]]
        key = hashlib.blake2b(repr((self.decisions, chunk, plan)).encode(), digest_size=8).digest()
        digest = float(int.from_bytes(key, "little") >> 12)  # 52 bits: exact in a float64
        d = float(self.decisions)
        vote = np.array([float(self.forwards), d, digest, -d, -digest, float(self._copy_request())])
        out = np.empty(6, dtype=np.float64)
        self.ready_vote = (
            self.comm_ready.Iallreduce(vote, out, op=MPI.MAX),
            (vote, out),
            reply["installs"],
            chunk,
        )
        note = self._ctl_apply(reply) if self._last.get("ctl") is not None else ""
        if self.ep_rank == 0:
            logger.info(
                f"GenEPLB renewal {self.renewals} @replay {self.replays}: mode {self.mode}, T {self._last['tokens']}"
                f"{'' if self._last.get('install', True) else ' (below LIVE_MIN_T: shares only)'}, "
                f"shares {len(reply['tables'])} layers, gate {len(reply['installs'])}/{reply['candidates']} layers, "
                f"N_rem {self._last['n_rem']}{' (live horizon)' if self.live else ''}, c_gpu_ms {self.c_gpu_ms}"
                f"{note}"
            )
        self.dormant = not reply["installs"] and not self._holds_replicas()
        self._next_window()

    def _next_window(self) -> None:
        gap = DORMANT_PERIOD if self.dormant else RENEWAL_FLOOR - WINDOW
        self.window_start = max(self.replays + 1, self.window_end + gap)
        self.window_end = None

    # ------------------------------------------------------------------ the control channel
    def _ctl_read(self) -> None:
        """Re-read CTL_FILE when its (mtime, size, inode) changed: a missing file is every default, any other error keeps
        the codes in force. open() + fstat, not a bare stat: on NFS an open revalidates the attributes (close-to-open)."""
        try:
            with open(CTL_FILE, "rb") as f:
                st = os.fstat(f.fileno())
                key = (st.st_mtime_ns, st.st_size, st.st_ino)
                if key == self._ctl_key:
                    return
                text = f.read().decode("utf-8", "replace")
        except FileNotFoundError:
            self._ctl_codes, self._ctl_key = CTL_DEFAULT, None
            return
        except OSError:
            return
        self._ctl_codes, bad = parse_ctl(text)
        self._ctl_key = key
        if bad and self.ep_rank == 0:
            logger.warning(f"GenEPLB ctl: ignored {bad} in {CTL_FILE}")

    def _ctl_apply(self, reply: dict) -> str:
        """A decision's control fields take effect: the gate's hysteresis state, the device word and the drop flag (written
        with the renewal's tables, _write_tables), this rank's status file on a change. -> the renewal line's note."""
        last = self._last
        ctl = last["ctl"]
        self._hyst = reply.get("hyst", {})
        if self.t_ctl is not None:
            self._ctl_word_next = CTL_PAD_WORD[ctl["pad"]] | (CTL_IDENTITY_BIT if ctl["identity"] else 0)
            self._drop_next = ctl["pad"] == "drop"
        text = ctl_text(ctl)
        status = f"{text} word={self._ctl_word_next} nan={ctl['nan']}"
        if status != self._ctl_applied:
            if self.ep_rank == 0 and not (self._ctl_applied or "").startswith(text + " "):
                logger.info(
                    f"GenEPLB ctl @renewal {self.renewals} (decision {self.decisions}, replay {self.replays}): {text}, "
                    f"device word {self._ctl_word_next}"
                )
                if self.t_ctl is None and (ctl["pad"] != "spread" or ctl["identity"]):
                    logger.warning("GenEPLB ctl: pad / identity need cost model B's route kernel; mode C ignores them")
                if ctl["copy"] == "ce":
                    logger.warning("GenEPLB ctl: copy=ce is reserved; this build copies with NCCL")
                if ctl["placement"] == "flex" and (ctl["objective"] != "ef" or ctl["pad"] != "drop"):  # (p56ctl)
                    logger.warning(
                        "GenEPLB ctl: placement=flex needs objective=ef and pad=drop; inert, the planner keeps the lpt "
                        "placement"
                    )
                if self.mode != "B" and (ctl["objective"] != "shipped" or ctl["placement"] != "lpt"):  # (p56ctl)
                    logger.warning("GenEPLB ctl: objective / placement need cost model B; mode C ignores them")
            self._ctl_applied = status
            self._ctl_status(status)
        d = reply.get("diag") or {}
        return (
            f" | ctl {text}; T real {last['tokens']} padded {ctl['T_pad']} guard {ctl['guard_t']} {last['t_guard']}"
            f"{'' if last['install'] else ' (no install: ' + last['why'] + ')'}; gate us/step: "
            f"cur-new {d.get('cur_new', 0.0):.2f} admitted {d.get('adm', 0.0):.2f} ({d.get('n_eval', 0)} eval, "
            f"{d.get('n_copy', 0)} pass copy, {d.get('n_margin', 0)} margin-rej, {d.get('n_hyst', 0)} hyst-wait), "
            f"cur-id {d.get('cur_id', 0.0):.2f}, J_cur {d.get('j_cur', 0.0):.1f}; word {self._ctl_word_next}; "
            f"nan {ctl['nan']}"
        )

    def _ctl_status(self, status: str) -> None:
        """<CTL_FILE>.rank<r>: the control this rank applies (rewritten on a change: tmp + rename)."""
        path = f"{CTL_FILE}.rank{self.ep_rank}"
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                f.write(
                    f"rank={self.ep_rank} renewal={self.renewals} decision={self.decisions} replay={self.replays} "
                    f"forwards={self.forwards} {status}\n"
                )
            os.replace(tmp, path)
        except OSError:
            pass

    def _holds_replicas(self) -> bool:
        """Whether any layer has a copy in a spare slot (its layout differs from the home layout)."""
        M, H = self.geo.local_slots, self.geo.homes
        return any(lay[s] >= 0 for lay in self.layouts for s in range(len(lay)) if s % M >= H)

    def _stage_layer(self, li: int, arrays) -> None:
        self.h_count[li], self.h_start[li], self.h_slots[li], self.h_cumw[li] = arrays

    def _write_tables(self) -> None:
        """Every layer's tables in one H2D per table, between two replays on the forward stream."""
        for dst, src in (
            (self.t_count, self.h_count),
            (self.t_start, self.h_start),
            (self.t_slots, self.h_slots),
            (self.t_cumw, self.h_cumw),
        ):
            dst.copy_(maybe_pin_memory(torch.from_numpy(src)), non_blocking=True)
        self.t_wmode.fill_(1)
        if self._ctl_word_next != self._ctl_word:  # the control word switches with the renewal's tables
            self.t_ctl.fill_(self._ctl_word_next)
            self._ctl_word = self._ctl_word_next
        self._drop_words = self._drop_next
        self.pending_writes = False

    def _copy_due(self) -> None:
        """Copy the replicas due before this replay. Every rank copies the same chunk of the plan between the
        same two replays (the agreed step), on the forward stream: the copies never run beside the MoE kernels,
        whose cross-rank barriers would otherwise wait on SMs the copies hold."""
        for ins in self.installs:
            chunks = -(-len(ins["plans"]) // ins["chunk"])
            step = ins["first"] + ins["issued"]
            if ins["issued"] >= chunks or self.forwards < step:
                continue
            if self.forwards > step:
                raise RuntimeError(
                    f"GenEPLB rank {self.ep_rank}: forward {self.forwards} is past the agreed copy step {step}"
                )
            k = ins["issued"]
            plans = ins["plans"][k * ins["chunk"] : (k + 1) * ins["chunk"]]
            ins["events"].append((*self._copy_layers(plans, ins.get("copy", COPY_NCCL)), len(plans)))
            ins["issued"] += 1

    def _copy_layers(self, plans, mode=COPY_NCCL):
        """Copy the plans' replicas from their home slots, on the current stream: (start, done) timing events."""
        if mode == COPY_CE:
            if self._ce is None:
                self._arm_ce()  # collective; only if after_warmup did not run (every rank is at this agreed step)
            if self._ce is not None:
                return self._ce.copy(plans)  # copy-engine pull through the symmetric staging buffer
        import torch.distributed as dist

        M = self.geo.local_slots
        ops, local = [], []
        for p in plans:
            for expert, dst in p["copies"]:
                src = self.geo.home_slot(expert)  # homes never move
                sr, dr = src // M, dst // M
                for _, t in self.slot_tensors[p["layer"]]:
                    if sr == dr == self.ep_rank:
                        local.append((t[dst % M], t[src % M]))
                    elif sr == self.ep_rank:
                        ops.append(dist.P2POp(dist.isend, t[src % M], dr, group=self.copy_group))
                    elif dr == self.ep_rank:
                        ops.append(dist.P2POp(dist.irecv, t[dst % M], sr, group=self.copy_group))
        if (
            ops and not self.connected
        ):  # NCCL connects a pair at its first transfer: not inside the timing
            probes = [
                dist.P2POp(
                    kind,
                    self._probe[0:1] if kind is dist.isend else self._probe[1 + peer : 2 + peer],
                    peer,
                    group=self.copy_group,
                )
                for peer, kind in sorted(
                    {(op.peer, op.op) for op in ops}, key=lambda pk: (pk[0], pk[1] is dist.irecv)
                )
            ]
            for w in dist.batch_isend_irecv(probes):
                w.wait()
        start, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for dst_t, src_t in local:
            dst_t.copy_(src_t, non_blocking=True)
        for w in dist.batch_isend_irecv(ops) if ops else []:
            w.wait()  # the forward stream waits for the copies: the next replay reads after them
        done.record()
        return start, done

    def after_warmup(self) -> None:
        """Once the final KV cache and CUDA graphs are allocated: connect every NCCL peer (NCCL connects a pair at
        its first transfer, which would otherwise stall the first install's steps) and time a startup copy of
        PROBE_LAYERS layers into FREE spare slots, so the first gate prices installs with a measured copy cost.
        Collective on every rank; once."""
        if not self.finalized or self.connected:
            return
        import time

        import torch.distributed as dist
        from mpi4py import MPI

        self.connected = True
        R, r = self.ep_size, self.ep_rank
        torch.cuda.synchronize()
        if COPY_CE in self._copy_armed and self._ce is None:
            self._arm_ce()  # symmetric staging buffer (collective); NCCL on every rank if any rank fails
        t0 = time.perf_counter()
        send = torch.zeros(CONNECT_BYTES, dtype=torch.uint8, device=self.device)
        recv = torch.empty((R, CONNECT_BYTES), dtype=torch.uint8, device=self.device)
        free0 = torch.cuda.mem_get_info(self.device)[0]
        peers = [p for p in range(R) if p != r] if COPY_NCCL in self._copy_armed else []
        ops = [dist.P2POp(dist.isend, send, p, group=self.copy_group) for p in peers]
        ops += [dist.P2POp(dist.irecv, recv[p], p, group=self.copy_group) for p in peers]
        for w in dist.batch_isend_irecv(ops) if ops else []:
            w.wait()
        torch.cuda.synchronize()
        nccl_mb = (free0 - torch.cuda.mem_get_info(self.device)[0]) / 2**20
        connect_s = time.perf_counter() - t0
        del send, recv
        # Probe: every rank receives the first replicas-per-rank home experts of the next rank into its first spare
        # slots (FREE: no rank routes there; a later install overwrites them).
        geo, n = self.geo, max(1, (self.geo.local_slots - self.geo.homes) // 2)
        plans = [
            dict(
                layer=li,
                copies=[
                    (((d + 1) % R) * geo.homes + j, d * geo.local_slots + geo.homes + j)
                    for d in range(R)
                    for j in range(n)
                ],
            )
            for li in range(min(PROBE_LAYERS, self.num_layers))
        ]
        probe = {}
        for mode in self._copy_armed:  # one startup price per armed transport
            for _ in range(2):  # the second, warm copy prices installs (the first pays one-time setup)
                start, done = self._copy_layers(plans, mode)
                done.synchronize()
            probe[mode] = start.elapsed_time(done) / len(plans)
        connect_s, nccl_mb, *probe_max = (
            self.comm_done.allreduce(v, op=MPI.MAX) for v in (connect_s, nccl_mb, *probe.values())
        )
        for mode, ms in zip(probe, probe_max):
            self._copy_hist[mode] = [ms]
        self._set_copy_mode(self._copy_mode)
        if r == 0:
            logger.info(
                f"GenEPLB connected {R - 1} peers in {connect_s:.2f} s (NCCL buffers {nccl_mb:.0f} MB, max over "
                f"ranks); startup copy probe {len(plans)} layers x {n} replicas/rank: c_gpu_ms {self.c_gpu_ms:.3f}"
                + ("" if self._copy_armed == (COPY_NCCL,) else
                   f"; copy transports {', '.join(f'{COPY_MODE_NAMES[m]} {p:.3f}' for m, p in zip(probe, probe_max))} "
                   f"ms/layer, agreed {COPY_MODE_NAMES[self._copy_mode]}; CE {self._ce.describe() if self._ce else 'off'}")
            )

    def _set_copy_mode(self, mode: int) -> None:
        """Make ``mode`` the agreed transport: the median of its measured copies prices the gate."""
        self._copy_mode = mode if mode in self._copy_armed else self._copy_armed[0]
        self.copy_ms_history = self._copy_hist.setdefault(self._copy_mode, [])
        if self.copy_ms_history:
            self.c_gpu_ms = float(np.median(self.copy_ms_history))

    def _copy_request(self) -> int:
        """This rank's transport request for the next vote: with both transports armed, the 'copy=' field of the
        CTL file (re-read on mtime change; no field: the startup default), else the agreed transport."""
        if len(self._copy_armed) < 2 or not COPY_CTL_FILE:
            return self._copy_mode
        try:
            mtime = os.stat(COPY_CTL_FILE).st_mtime
            if mtime != self._copy_ctl[0]:
                from .fastcopy import parse_copy_field

                with open(COPY_CTL_FILE) as f:
                    want = parse_copy_field(f.read())
                self._copy_ctl = (mtime, COPY_MODE_NAMES.index(want) if want else self._copy_default)
        except OSError:
            pass
        return self._copy_ctl[1] if self._copy_ctl[1] in self._copy_armed else self._copy_mode

    def _arm_ce(self) -> None:
        """Copy-engine transport (fastcopy.py): each rank allocates its symmetric staging buffer, then all
        rendezvous on the copy group. Collective; if any rank fails, every rank keeps NCCL copies."""
        from mpi4py import MPI

        ce, err = None, None
        try:
            from . import fastcopy

            geo = self.geo
            ce = fastcopy.CeCopier(
                self.slot_tensors, home_slot=geo.home_slot, local_slots=geo.local_slots, ranks=self.ep_size,
                rank=self.ep_rank, group=self.copy_group, device=self.device,
            )
        except Exception as e:  # any local failure: the ranks agree on NCCL below
            ce, err = None, e
        item = ce.item if ce is not None else -1
        mine = np.array([int(ce is not None), item, -item], dtype=np.int64)
        agreed = np.empty_like(mine)
        self.comm_done.Allreduce(mine, agreed, op=MPI.MIN)
        if agreed[0] and agreed[1] == -agreed[2]:  # every rank allocated, one staging layout
            try:
                ce.connect()  # collective rendezvous: every peer's staging buffer mapped (fabric handles)
            except Exception as e:
                err = e
            if self.comm_done.allreduce(int(err is None), op=MPI.MIN):
                self._ce = ce
                return
        self._ce = None
        self._copy_armed, self._copy_mode, self._copy_default = (COPY_NCCL,), COPY_NCCL, COPY_NCCL
        logger.warning(f"GenEPLB rank {self.ep_rank}: copy-engine transport unavailable ({err!r}): NCCL copies")

    def _poll_installs(self) -> None:
        """Vote once this rank's copies of an install landed; publish once every rank's have. Never blocks: a
        peer may need this rank's next forward before it can reach the matching call."""
        while self.installs:
            ins = self.installs[0]
            if ins["vote"] is None:
                if (
                    ins["issued"] < -(-len(ins["plans"]) // ins["chunk"])
                    or not ins["events"][-1][1].query()
                ):
                    return
                from mpi4py import MPI

                ms = max(s.elapsed_time(d) / max(1, n) for s, d, n in ins["events"])
                ins["send"] = np.array([ms])
                ins["recv"] = np.empty(1, dtype=np.float64)
                ins["vote"] = self.comm_done.Iallreduce(ins["send"], ins["recv"], op=MPI.MAX)
            if not ins["vote"].Test():
                return
            self.installs.pop(0)
            self._publish(ins)

    def _publish(self, ins: dict) -> None:
        for p in ins["plans"]:
            li = p["layer"]
            old, new = self.layouts[li], list(p["layout"])
            self.retired[li] = [s for s in range(len(old)) if old[s] >= 0 and new[s] < 0]
            self.layouts[li] = new
            self._stage_layer(li, p["tables"])
        self.pending_writes = True
        copy_ms = float(ins["recv"][0])
        mode = ins.get("copy", COPY_NCCL)
        self._copy_hist[mode] = (self._copy_hist.get(mode, []) + [copy_ms])[-COPY_HISTORY:]
        self._set_copy_mode(self._copy_mode)  # the gate prices the agreed transport's copies
        self.install_count += 1
        if self.ep_rank == 0:
            copies = sum(len(p["copies"]) for p in ins["plans"])
            logger.info(
                f"GenEPLB install {self.install_count} @replay {self.replays}: {len(ins['plans'])} layers, "
                f"{copies} replicas, copy {copy_ms:.3f} ms/layer{' (ce)' if mode == COPY_CE else ''} (max over ranks), "
                f"gate estimate {self.c_gpu_ms:.3f} "
                f"(median of {len(self.copy_ms_history)})"
            )

    def shutdown(self, local_safe: bool = True) -> bool:
        if self.is_shutdown:
            return True
        self.is_shutdown = True
        worker = getattr(self, "worker", None)
        if worker is not None:
            worker.close()
        return True


def _is_dummy(request) -> bool:
    return any(
        bool(getattr(request, name, False))
        for name in ("is_dummy", "is_dummy_request", "is_attention_dp_dummy", "is_cuda_graph_dummy")
    )
