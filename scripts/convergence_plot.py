from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pricer.black_scholes import call_price
from pricer.binomial import binomial_price



BASE = dict(S=100, K=100, T=1, r=0.05, sigma=0.2)


def main():
    steps = np.arange(10, 501)
    tree = np.array([binomial_price(**BASE, steps=n) for n in steps])
    bs = call_price(**BASE)

    even = steps % 2 == 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps[even], tree[even], lw=1, label="Binomial tree, even N")
    ax.plot(steps[~even], tree[~even], lw=1, label="Binomial tree, odd N")
    ax.axhline(bs, color="black", ls="--", lw=1, label=f"Black-Scholes ({bs:.4f})")

    ax.set_xlabel("Number of steps N")
    ax.set_ylabel("Call price")
    ax.set_title("CRR binomial tree converges to Black-Scholes (S = K = 100, T = 1)")
    ax.legend()
    fig.tight_layout()

    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "binomial_convergence.png", dpi=150)


if __name__ == "__main__":
    main()