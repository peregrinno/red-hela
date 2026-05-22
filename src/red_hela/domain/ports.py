from typing import Protocol


class FraudScorerPort(Protocol):
    def score(self, payload: dict) -> tuple[bool, float]: ...
