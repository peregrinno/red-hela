import msgspec

RESPONSE_READY = msgspec.json.encode({"status": "ready"})

RESPONSE_BODIES: dict[tuple[bool, float], bytes] = {}
for fraud_count in range(6):
    fraud_score = fraud_count / 5.0
    approved = fraud_score < 0.6
    RESPONSE_BODIES[(approved, fraud_score)] = msgspec.json.encode(
        {"approved": approved, "fraud_score": fraud_score}
    )

RESPONSE_FALLBACK = RESPONSE_BODIES[(True, 0.0)]
