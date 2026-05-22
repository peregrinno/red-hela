from datetime import UTC, datetime

import numpy as np

from red_hela.domain.transaction import FraudScoreRequest

NORMALIZATION = {
    "max_amount": 10000.0,
    "max_installments": 12.0,
    "amount_vs_avg_ratio": 10.0,
    "max_minutes": 1440.0,
    "max_km": 1000.0,
    "max_tx_count_24h": 20.0,
    "max_merchant_avg_amount": 10000.0,
}

MCC_RISK = {
    "5411": 0.15,
    "5812": 0.30,
    "5912": 0.20,
    "5944": 0.45,
    "7801": 0.80,
    "7802": 0.75,
    "7995": 0.85,
    "4511": 0.35,
    "5311": 0.25,
    "5999": 0.50,
}


class TransactionVectorizer:
    def __init__(self) -> None:
        self._normalization = NORMALIZATION
        self._mcc_risk = MCC_RISK

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def vectorize(self, payload: FraudScoreRequest) -> np.ndarray:
        transaction = payload.transaction
        customer = payload.customer
        merchant = payload.merchant
        terminal = payload.terminal
        last_transaction = payload.last_transaction

        requested_at = self._as_utc(transaction.requested_at)
        customer_avg_amount = customer.avg_amount
        transaction_amount = transaction.amount
        amount_vs_avg = 0.0
        if customer_avg_amount > 0.0:
            amount_vs_avg = self._clamp(
                (transaction_amount / customer_avg_amount)
                / self._normalization["amount_vs_avg_ratio"]
            )

        minutes_since_last_tx = -1.0
        km_from_last_tx = -1.0
        if last_transaction is not None:
            last_timestamp = self._as_utc(last_transaction.timestamp)
            delta_minutes = max(
                0.0,
                (requested_at - last_timestamp).total_seconds() / 60.0,
            )
            minutes_since_last_tx = self._clamp(
                delta_minutes / self._normalization["max_minutes"]
            )
            km_from_last_tx = self._clamp(
                last_transaction.km_from_current / self._normalization["max_km"]
            )

        return np.array(
            [
                self._clamp(transaction_amount / self._normalization["max_amount"]),
                self._clamp(
                    transaction.installments / self._normalization["max_installments"]
                ),
                amount_vs_avg,
                requested_at.hour / 23.0,
                requested_at.weekday() / 6.0,
                minutes_since_last_tx,
                km_from_last_tx,
                self._clamp(terminal.km_from_home / self._normalization["max_km"]),
                self._clamp(
                    customer.tx_count_24h / self._normalization["max_tx_count_24h"]
                ),
                1.0 if terminal.is_online else 0.0,
                1.0 if terminal.card_present else 0.0,
                0.0 if merchant.id in customer.known_merchants else 1.0,
                float(self._mcc_risk.get(merchant.mcc, 0.5)),
                self._clamp(
                    merchant.avg_amount / self._normalization["max_merchant_avg_amount"]
                ),
            ],
            dtype=np.float32,
        )
