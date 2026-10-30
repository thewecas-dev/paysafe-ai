"""Generate synthetic UPI-like transaction records for academic demonstration only."""
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    n = 2500
    amount = np.clip(rng.lognormal(mean=6.4, sigma=1.0, size=n), 10, 150000)
    txns_last_hour = rng.poisson(1.2, n)
    new_payee = rng.binomial(1, 0.22, n)
    new_device = rng.binomial(1, 0.12, n)
    night = rng.binomial(1, 0.18, n)
    failed_attempts = rng.poisson(0.25, n)

    # Synthetic label rule with noise. Not representative of real-world fraud.
    logit = (
        -4.0
        + 0.000025 * amount
        + 0.55 * txns_last_hour
        + 1.1 * new_payee
        + 1.3 * new_device
        + 0.65 * night
        + 0.8 * failed_attempts
    )
    prob = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, prob, n)

    df = pd.DataFrame({
        "transaction_id": [f"TX{i+1:05d}" for i in range(n)],
        "amount": np.round(amount, 2),
        "txns_last_hour": txns_last_hour,
        "new_payee": new_payee,
        "new_device": new_device,
        "night_transaction": night,
        "failed_attempts": failed_attempts,
        "is_fraud": fraud,
    })
    import os
    os.makedirs("data", exist_ok=True)
    out = "data/transactions.csv"
    df.to_csv(out, index=False)
    print(f"Created {out} with {len(df)} synthetic records.")
    print("Important: labels are generated for demonstration, not actual UPI fraud data.")

if __name__ == "__main__":
    main()
