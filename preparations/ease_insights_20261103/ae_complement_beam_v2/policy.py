"""Use the established periodic policy with this cohort's frozen masks."""
from preparations.ease_insights_20261103.module_tradeoff_v1.policy import Policy as OriginalPolicy
from .common import validate_condition


class Policy(OriginalPolicy):
    def __init__(self, backend):
        super().__init__(backend, validator=validate_condition)
