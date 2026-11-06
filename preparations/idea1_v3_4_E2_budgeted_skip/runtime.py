"""Apply prepared budgets at Full boundaries without discarding residual history."""
from copy import deepcopy
from pathlib import Path

from .budget import load_budget, prepare_budget


def checked_budget(artifact):
    if isinstance(artifact, dict):
        loaded = load_budget(artifact['artifact_path'])
        if loaded != artifact:
            raise ValueError('Budget dictionary differs from its verified artifact')
        return loaded
    return load_budget(artifact)


def default_config(budget, cost_weighted=False):
    return dict(id='adjustable_mixed_E', condition='adjustable_mixed_E',
        budget=budget['budget_id'], family='mixed', scoring='error',
        cost_weighted=bool(cost_weighted), method='comp_error_budgeted',
        compensation='extrapolate', adjacency_rule='spaced',
        model='pi0', mode='periodic', period=5, full_steps=10, fast_steps=2,
        H=50, R=5, weights=[1/31, 2/31, 4/31, 8/31, 16/31])


class BudgetedPolicy:
    """A mutable budget around the already verified native mixed policy.

    Prepare new profiles off the prediction path, then queue them with
    set_budget. A queued profile takes effect at the next Full, preserving the
    current four-Fast mask, proposal acknowledgement, and two real Full banks.
    """
    def __init__(self, backend, budget_artifact, *, cost_weighted=False,
                 policy_factory=None, artifact_root=None):
        if not isinstance(cost_weighted, bool):
            raise ValueError('cost_weighted must be a boolean')
        self._budget = checked_budget(budget_artifact)
        self._queued_budget = None
        self._cost_weighted = cost_weighted
        self._artifact_root = Path(artifact_root) if artifact_root is not None else (
            Path(self._budget['artifact_path']).parent.parent)
        if policy_factory is None:
            if self._budget['n_layers'] != 17:
                raise ValueError('The native pi0 mixed policy requires seventeen candidate layers')
            from preparations.idea1_v3_4_E2_mixed_cost.policy import Policy
            policy_factory = Policy
        self._policy = policy_factory(backend, {self._budget['budget_id']: self._budget})
        self.configure()

    def __getattr__(self, name):
        return getattr(self._policy, name)

    @property
    def current_budget(self):
        return deepcopy(self._budget)

    @property
    def queued_budget(self):
        return None if self._queued_budget is None else deepcopy(self._queued_budget)

    def _apply(self, budget):
        self._budget = budget
        self._policy.budgets = {budget['budget_id']: budget}
        self._policy.config['budget'] = budget['budget_id']

    def configure(self, config=None):
        # A new episode is itself a Full boundary. The explicit configure call
        # clears policy state; ordinary budget changes never do so.
        next_budget = self._queued_budget if self._queued_budget is not None else self._budget
        overrides = dict(config or {})
        required = dict(period=5, adjacency_rule='spaced', family='mixed',
                        scoring='error', compensation='extrapolate', H=50, R=5,
                        full_steps=10, fast_steps=2, model='pi0', mode='periodic',
                        weights=[1/31, 2/31, 4/31, 8/31, 16/31])
        for key, value in required.items():
            if key in overrides and overrides[key] != value:
                raise ValueError('Budget control does not alter the fixed protocol: ' + key)
        if 'budget' in overrides and overrides['budget'] != next_budget['budget_id']:
            raise ValueError('Use set_budget to select a different verified budget artifact')
        weighted = overrides.get('cost_weighted', self._cost_weighted)
        if not isinstance(weighted, bool):
            raise ValueError('cost_weighted must be a boolean')
        merged = default_config(next_budget, weighted)
        merged.update(overrides)
        previous_budgets = self._policy.budgets
        self._policy.budgets = {next_budget['budget_id']: next_budget}
        try:
            result = self._policy.configure(merged)
        except Exception:
            self._policy.budgets = previous_budgets
            raise
        self._budget = next_budget
        self._queued_budget = None
        self._cost_weighted = weighted
        return result

    def set_budget(self, *, budget_artifact=None, saved_ms=None, saved_fraction=None,
                   tolerance_ms=None, reference_vlm_ms=None):
        if budget_artifact is not None:
            if any(value is not None for value in
                   (saved_ms, saved_fraction, tolerance_ms, reference_vlm_ms)):
                raise ValueError('Choose a prepared artifact or a new budget specification')
            budget = checked_budget(budget_artifact)
        else:
            source = self._budget.get('cost_table_path')
            if source is None:
                raise ValueError('A measured cost table is required to prepare a new budget')
            budget = prepare_budget(source, self._artifact_root,
                saved_ms=saved_ms, saved_fraction=saved_fraction,
                tolerance_ms=self._budget['tau_ms'] if tolerance_ms is None else tolerance_ms,
                reference_vlm_ms=reference_vlm_ms)
        if budget['cost_table_hash'] != self._budget['cost_table_hash']:
            raise ValueError('Changing the hardware calibration requires a new policy instance')
        if budget['n_layers'] != self._budget['n_layers']:
            raise ValueError('Budget layer count differs from the current policy')
        self._queued_budget = budget
        return dict(queued=True, applied_at='next_full', budget_id=budget['budget_id'],
                    target_saved_ms=budget['anchor_saved_ms'])

    def predict(self, *args, **kwargs):
        if self._policy.pending is not None:
            raise RuntimeError('Unacknowledged action')
        changed = False
        previous_budget = self._budget
        previous_round = self._policy.round
        if self._queued_budget is not None and self._policy.round % 5 == 0:
            self._apply(self._queued_budget)
            self._queued_budget = None
            changed = True
        try:
            result = self._policy.predict(*args, **kwargs)
        except Exception:
            if changed and self._policy.round == previous_round and self._policy.pending is None:
                failed_budget = self._budget
                self._apply(previous_budget)
                self._queued_budget = failed_budget
            raise
        result['route'].update(budget_target_saved_ms=self._budget['anchor_saved_ms'],
            budget_target_saved_fraction=self._budget.get('requested_saved_fraction'),
            budget_reference_vlm_ms=self._budget.get('reference_vlm_ms'),
            budget_change_applied=changed, budget_hash=self._budget['budget_hash'])
        return result

    def acknowledge(self, *args, **kwargs):
        return self._policy.acknowledge(*args, **kwargs)

    def reset(self):
        return self.configure()

    def close(self):
        self._policy.selector.clear()
        self._policy.reset()
