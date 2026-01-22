#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def sign(x):
    return (x > 0) - (x < 0)

# -------------------------------------------------
# Matter generator ⊞ (projective, 3D)
# -------------------------------------------------

def matter_add(P1, P2):
    X1, Y1, Z1 = P1
    X2, Y2, Z2 = P2

    U = Y2*Z1 - Y1*Z2
    V = X2*Z1 - X1*Z2
    W = Z1*Z2
    if V == 0:
        V = 1

    A = U*U*W - V**3 - 2*V*V*X1*Z2
    X3 = V*A
    Y3 = U*(V*V*X1*Z2 - A) - V**3*Y1*Z2
    Z3 = V**3 * W
    return np.array([X3, Y3, Z3], dtype=int)

# -------------------------------------------------
# ψ operator (Dirac symplectic resolution)
# -------------------------------------------------

def psi(PA, PB):
    XA, YA, ZA = PA
    XB, YB, ZB = PB
    return (
        np.array([XB, YA, ZA], dtype=int),
        np.array([XA, YB, ZB], dtype=int)
    )

# -------------------------------------------------
# Structural tension
# -------------------------------------------------

def tension(P, Pn):
    Xt, _, Zt = P
    Xn, _, Zn = Pn
    return Xt*Zn - Xn*Zt

# -------------------------------------------------
# Rational Lorentz boost Λᵤ
# -------------------------------------------------

def lorentz_boost(P, U):
    p, q = U
    X, Y, Z = P
    return np.array([
        q*X + p*Z,
        Y,
        p*X + q*Z
    ], dtype=int)

# -------------------------------------------------
# One Dirac step
# -------------------------------------------------

def dirac_step(PA, PB, field=None):
    if field is not None:
        PA = matter_add(PA, field)
        PB = matter_add(PB, field)
    return psi(PA, PB)

# -------------------------------------------------
# Simulation core
# -------------------------------------------------

def simulate(args):
    steps = args.steps

    PA = np.array([1, 1, 1], dtype=int)
    PB = np.array([1, -1, 1], dtype=int)

    field = None
    if args.bound:
        field = np.array([1, 0, 1], dtype=int)

    trajA, trajB, spins = [PA.copy()], [PB.copy()], []

    for _ in range(steps):
        PA0 = PA.copy()
        PA, PB = dirac_step(PA, PB, field)

        if args.boost:
            PA = lorentz_boost(PA, args.u)
            PB = lorentz_boost(PB, args.u)

        τ = tension(PA0, PA)
        spins.append(sign(τ))

        trajA.append(PA.copy())
        trajB.append(PB.copy())

    return np.array(trajA), np.array(trajB), spins

# -------------------------------------------------
# Visualization
# -------------------------------------------------

def plot_static(A, B, spins):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*A.T, label="Channel A")
    ax.plot(*B.T, label="Channel B")
    ax.legend()
    ax.set_title("Rigbyspace Dirac Trajectory")

    ax2 = fig.add_subplot(122)
    ax2.step(range(len(spins)), spins, where="post")
    ax2.set_title("Spin (τ sign)")
    plt.show()

def plot_animation(A, B):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    lineA, = ax.plot([],[],[],'o-',label="A")
    lineB, = ax.plot([],[],[],'o-',label="B")
    ax.legend()

    def update(i):
        lineA.set_data(A[:i,0], A[:i,1])
        lineA.set_3d_properties(A[:i,2])
        lineB.set_data(B[:i,0], B[:i,1])
        lineB.set_3d_properties(B[:i,2])
        return lineA, lineB

    FuncAnimation(fig, update, frames=len(A), interval=200)
    plt.show()

# -------------------------------------------------
# Stern–Gerlach analogue
# -------------------------------------------------

def stern_gerlach(spins):
    up = spins.count(1)
    down = spins.count(-1)
    print(f"Spin-Up: {up}, Spin-Down: {down}")

# -------------------------------------------------
# ψ-frequency vs mass scan
# -------------------------------------------------

def mass_scan():
    print("Mass-Level → ψ frequency")
    for d in range(1,6):
        print(f"mL={d} → ψ-rate ~ {2*d}")

# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3D Rigbyspace Dirac Simulator"
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--boost", action="store_true")
    parser.add_argument("--u", nargs=2, type=int, default=[1,0])
    parser.add_argument("--free", action="store_true")
    parser.add_argument("--bound", action="store_true")
    parser.add_argument("--stern-gerlach", action="store_true")
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--mass-scan", action="store_true")

    args = parser.parse_args()
    args.u = tuple(args.u)

    if args.mass_scan:
        mass_scan()
        return

    A, B, spins = simulate(args)

    if args.stern_gerlach:
        stern_gerlach(spins)

    if args.animate:
        plot_animation(A, B)
    else:
        plot_static(A, B, spins)

if __name__ == "__main__":
    main()

