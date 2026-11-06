"""Frozen six-mask policy mechanics with the within-suite identity whitelist."""
from preparations.ease_insights_20261103.vlm_local_search_v1.policy import Policy as FrozenPolicy
from .common import validate_condition


class Policy(FrozenPolicy):
    def configure(self, config):
        self.mask = validate_condition(config)
        self.config = dict(config)
        self.reset()
        return {}
