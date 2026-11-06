"""Reuse the frozen policy mechanics, with an isolated mask whitelist."""
from preparations.ease_insights_20261103.vlm.policy import Policy as FrozenPolicy
from .common import validate_condition


class Policy(FrozenPolicy):
    def configure(self, config):
        self.mask = validate_condition(config)
        self.config = dict(config)
        self.reset()
        return {}
