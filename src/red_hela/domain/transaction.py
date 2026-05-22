from datetime import datetime

import msgspec


class TransactionPayload(msgspec.Struct):
    amount: float
    installments: int
    requested_at: datetime


class CustomerPayload(msgspec.Struct):
    avg_amount: float
    tx_count_24h: int
    known_merchants: list[str]


class MerchantPayload(msgspec.Struct):
    id: str
    mcc: str
    avg_amount: float


class TerminalPayload(msgspec.Struct):
    is_online: bool
    card_present: bool
    km_from_home: float


class LastTransactionPayload(msgspec.Struct):
    timestamp: datetime
    km_from_current: float


class FraudScoreRequest(msgspec.Struct):
    id: str
    transaction: TransactionPayload
    customer: CustomerPayload
    merchant: MerchantPayload
    terminal: TerminalPayload
    last_transaction: LastTransactionPayload | None = None
