# riemann_resonance_toy.py
# A disciplined experimental scaffold for prime-gated unreduced-pair dynamics.
# Two observables:
#   (1) flux_bunce: abs(sum(det(prev, curr))) / L
#   (2) LQ_like: average centered sign of (tension mod N^2) over an induced cycle
#
# No floating point required for observables except when printing ratios.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -------------------------
# Prime utilities (sieve)
# -------------------------

def sieve_is_prime(max_n: int) -> List[bool]:
    """Return boolean is_prime array for 0..max_n."""
    if max_n < 1:
        return [False] * (max_n + 1)
    is_p = [True] * (max_n + 1)
    is_p[0] = False
    if max_n >= 1:
        is_p[1] = False
    p = 2
    while p * p <= max_n:
        if is_p[p]:
            step = p
            start = p * p
            for k in range(start, max_n + 1, step):
                is_p[k] = False
        p += 1
    return is_p

# -------------------------
# Core discrete maps
# -------------------------

@dataclass(frozen=True)
class PairState:
    n: int
    d: int

def step_prime_gated(state: PairState, t: int, is_prime: List[bool]) -> PairState:
    """
    Prime-gated update rule (toy):
      - if t is prime: SURGE along diagonal: (n,d)->(n+1,d+1)
      - else: TWIST swap: (n,d)->(d,n)
    """
    n, d = state.n, state.d
    if is_prime[t]:
        return PairState(n + 1, d + 1)
    else:
        return PairState(d, n)

def det2(a: PairState, b: PairState) -> int:
    """2x2 determinant det([a],[b]) = a.n*b.d - b.n*a.d."""
    return a.n * b.d - b.n * a.d

# -------------------------
# Observable 1: Bunce flux
# -------------------------

def flux_bunce(L: int, is_prime: List[bool], state0: PairState = PairState(0, 1)) -> float:
    """
    Compute abs(sum_t det(prev, curr)) / L.
    Uses integers internally; returns float for convenience.
    """
    prev = state0
    bunce = 0
    cur = state0
    for t in range(1, L + 1):
        cur = step_prime_gated(cur, t, is_prime)
        bunce += det2(prev, cur)
        prev = cur
    return abs(bunce) / L

# ---------------------------------------------
# Observable 2: "LQ-like" cycle / mod N^2 sign
# ---------------------------------------------

def centered_residue(x: int, mod: int) -> int:
    """Map x mod mod into symmetric residue in (-mod/2, mod/2]."""
    r = x % mod
    half = mod // 2
    if r > half:
        r -= mod
    return r

def centered_sign(x: int, mod: int) -> int:
    """Sign of centered residue class: +1, 0, -1."""
    r = centered_residue(x, mod)
    if r > 0:
        return 1
    if r < 0:
        return -1
    return 0

def LQ_like(
    N: int,
    is_prime: List[bool],
    max_steps: int = 500000,
    state0: PairState = PairState(0, 1),
    cycle_key: str = "modN"
) -> Tuple[float, int]:
    """
    A framework-aligned toy:
      - evolve prime-gated dynamics
      - compute tension tau_t = det(prev,curr)
      - reduce tau_t mod N^2, take centered sign
      - average centered sign over a detected cycle

    Cycle detection:
      - cycle_key="modN": detect cycle on (n mod N, d mod N)
      - cycle_key="modN2": detect cycle on (n mod N^2, d mod N^2)
      - cycle_key="raw": detect cycle on exact (n,d) (may never cycle; not recommended)

    Returns: (average_sign_over_cycle, cycle_length)
    """
    if N <= 1:
        raise ValueError("N must be >= 2")
    mod = N * N

    seen: Dict[Tuple[int, int], Tuple[int, int]] = {}
    # maps key -> (time index when first seen, cumulative sum at that time)

    def make_key(s: PairState) -> Tuple[int, int]:
        if cycle_key == "modN":
            return (s.n % N, s.d % N)
        if cycle_key == "modN2":
            return (s.n % mod, s.d % mod)
        if cycle_key == "raw":
            return (s.n, s.d)
        raise ValueError("cycle_key must be 'modN', 'modN2', or 'raw'")

    cur = state0
    prev = state0
    ssum = 0  # cumulative sum of centered_sign(tau_t mod N^2)
    t = 0

    # seed
    k0 = make_key(cur)
    seen[k0] = (0, 0)

    while t < max_steps:
        t += 1
        cur = step_prime_gated(cur, t, is_prime)
        tau = det2(prev, cur)
        ssum += centered_sign(tau, mod)
        prev = cur

        k = make_key(cur)
        if k in seen:
            t0, s0 = seen[k]
            cycle_len = t - t0
            cycle_sum = ssum - s0
            avg = cycle_sum / cycle_len
            return avg, cycle_len

        seen[k] = (t, ssum)

    # If no cycle detected within max_steps:
    return (ssum / max(1, t), 0)

# -------------------------
# Experiment harness
# -------------------------

def run_horizon_sequences():
    sequences = [
        ("Original", [101, 401, 1009, 4621, 7477, 19997]),
        ("Small Primes-ish", [11, 23, 47, 97, 197, 397, 797]),
        ("Twin Leaders-ish", [11, 29, 71, 101, 191, 311, 521]),
        ("Fibonacci-ish", [89, 233, 377, 610, 987, 1597, 2584]),
        ("Highly Composite-ish", [120, 360, 840, 1260, 2520, 5040, 10080]),
        ("Powers2-1-ish", [127, 255, 511, 1023, 2047, 4095, 8191]),
    ]

    maxL = max(max(vals) for _, vals in sequences)
    is_prime = sieve_is_prime(maxL)

    print("\n=== Observable 1: bunce flux ===")
    print(f"{'Sequence':<22} {'L':>8} {'flux':>14}")
    print("-" * 48)
    for name, vals in sequences:
        for L in vals:
            f = flux_bunce(L, is_prime)
            print(f"{name:<22} {L:>8} {f:>14.8f}")

def run_LQ_over_primes(prime_horizons: List[int], cycle_key: str = "modN"):
    """
    Treat each prime p as modulus N and compute LQ_like(p).
    This is closer to your documents' 'periodic average mod N^2' idea than bunce/L.
    """
    maxN = max(prime_horizons)
    # Need primes up to max_steps too; choose max_steps based on maxN
    # We'll build prime list up to a reasonable cap for gating time.
    max_steps = min(200000, 40 * maxN)  # heuristic; adjust as needed

    is_prime = sieve_is_prime(max_steps)

    print("\n=== Observable 2: LQ_like(N) over listed N ===")
    print(f"cycle_key = {cycle_key}, max_steps = {max_steps}")
    print(f"{'N':>8} {'LQ_like':>14} {'cycle_len':>12}")
    print("-" * 42)
    for N in prime_horizons:
        avg, clen = LQ_like(N, is_prime, max_steps=max_steps, cycle_key=cycle_key)
        print(f"{N:>8} {avg:>14.8f} {clen:>12}")

if __name__ == "__main__":
    # 1) Horizon experiments (like your React view)
    run_horizon_sequences()

    # 2) Modulus experiments (closer to "L-function-like" cancellation structure)
    test_moduli = [101, 401, 1009, 4621, 7477, 19997]
    run_LQ_over_primes(test_moduli, cycle_key="modN")
    # Try modN2 if you want stricter state-keying:
    # run_LQ_over_primes(test_moduli, cycle_key="modN2")

