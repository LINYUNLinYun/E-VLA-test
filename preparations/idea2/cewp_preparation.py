"""Optional pi0.5 fixed-schedule conditioning preparation (immutable weights).

Values use the original CUDA FP32 recurrence and original MLP rounding path.
Only timestep conditioning is cached; images/hidden states/KV are never cached
by prompt identity. Each slot owns its own conditioning tensors.
"""
import torch
import torch.nn.functional as F

from .runtime_graph import DynamicCore


class PreparedScheduleCore(DynamicCore):
    def __init__(self, backend, prepared):
        super().__init__(backend, prepared)
        if not self.model.pi05:
            raise ValueError("This optional conditioning ablation is pi0.5 only")
        from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
        self.conditions = []
        self.schedule_values = []
        with torch.no_grad():
            t = self.initial_time.clone()
            for _ in range(self.num_steps):
                self.schedule_values.append(t.clone())
                expanded = t.expand(prepared.state.shape[0])
                value = create_sinusoidal_pos_embedding(expanded, self.model.action_in_proj.out_features,
                    min_period=4e-3, max_period=4.0, device=t.device).to(dtype=t.dtype)
                value = F.silu(self.model.time_mlp_in(value))
                value = F.silu(self.model.time_mlp_out(value))
                self.conditions.append(value)
                t += self.dt

    def _suffix(self, prepared, actions, timestep):
        condition = self.conditions[self.condition_index]
        self.condition_index += 1
        return self.model.action_in_proj(actions), condition

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        self.condition_index = 0
        result = super().generate(*args, **kwargs)
        assert self.condition_index == self.num_steps
        return result


class PreparedModulationCore(PreparedScheduleCore):
    """Also prepare the original per-layer adaRMS dense(condition) results.

Only used during single-host cold graph construction, before any Pool lease.
Original norm/scale/shift/gate math remains intact. Temporary Python dispatch
replacement is restored even on capture failure; replay uses slot-owned,
read-only tensors and never calls these Python functions.
"""
    def __init__(self, backend, prepared):
        if hasattr(backend, "compiler_originals"):
            raise ValueError("Modulation preparation is an independent eager-Graph ablation")
        super().__init__(backend, prepared)
        expert = self.model.paligemma_with_expert.gemma_expert.model
        self.modulation_modules = [m.dense for m in expert.modules()
                                   if getattr(m, "cond_dim", None) is not None and getattr(m, "dense", None) is not None]
        with torch.no_grad():
            self.modulations = [[dense(condition) for condition in self.conditions]
                                for dense in self.modulation_modules]

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        originals = [(m, m.forward) for m in self.modulation_modules]
        def prepared_forward(index):
            def forward(condition):
                step = self.condition_index-1
                if condition is not self.conditions[step]:
                    raise RuntimeError("Adaptive norm received an unprepared time condition")
                return self.modulations[index][step]
            return forward
        try:
            for index, module in enumerate(self.modulation_modules):
                module.forward = prepared_forward(index)
            return super().generate(*args, **kwargs)
        finally:
            for module, forward in originals:
                module.forward = forward
