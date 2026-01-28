"""
Rigbyspace Triadic Engine - Computational Implementation
========================================================

This is a direct implementation of the Rigbyspace discrete algebra engine
as specified in triadic_engine.pdf and triadic_engine2.pdf

Author: Implementation based on D. Veneziano's theoretical framework
Date: January 28, 2026
"""

from dataclasses import dataclass
from typing import Tuple, List, Set, Optional
from enum import Enum
import math

# ============================================================================
# PRIMITIVE ONTOLOGY
# ============================================================================

@dataclass(frozen=True)
class IntegerState:
    """
    An integer state S = (n, d) where n, d ∈ Z \ {0}
    NO reduction, normalization, or equivalence is applied
    """
    n: int
    d: int
    
    def __post_init__(self):
        if self.n == 0 or self.d == 0:
            raise ValueError("Neither n nor d can be zero")
    
    def __repr__(self):
        return f"({self.n}/{self.d})"


@dataclass(frozen=True)
class CoupledPair:
    """
    Coupled pair (S_L, S_U) - the fundamental dynamical object
    Single states never evolve alone
    """
    lower: IntegerState
    upper: IntegerState
    
    def __repr__(self):
        return f"[{self.lower}, {self.upper}]"


# ============================================================================
# FUNDAMENTAL OPERATORS
# ============================================================================

def barycentric_addition(s1: IntegerState, s2: IntegerState) -> IntegerState:
    """
    Barycentric Addition (⊞)
    S1 ⊞ S2 := (n1*d2 + n2*d1, d1*d2)
    """
    n = s1.n * s2.d + s2.n * s1.d
    d = s1.d * s2.d
    return IntegerState(n, d)


def vacuum_step(s1: IntegerState, s2: IntegerState) -> IntegerState:
    """
    Vacuum Step (⊕)
    S1 ⊕ S2 := (n1 + n2, d1 + d2)
    """
    n = s1.n + s2.n
    d = s1.d + s2.d
    return IntegerState(n, d)


def coupled_inversion(pair: CoupledPair) -> CoupledPair:
    """
    Coupled Inversion (ψ)
    ψ((a/b), (c/d)) := ((d/a), (b/c))
    
    ψ⁴ = I (period-4)
    ψ never acts on individual states
    """
    new_lower = IntegerState(pair.upper.d, pair.lower.n)
    new_upper = IntegerState(pair.lower.d, pair.upper.n)
    return CoupledPair(new_lower, new_upper)


# ============================================================================
# RANK AND STRUCTURAL DEPTH
# ============================================================================

def rank(m: int) -> int:
    """
    Rank ρ(m): number of dyadic growth layers
    Defined as: 2^ρ(m) ≤ |m| < 2^(ρ(m)+1)
    Essentially floor(log2(|m|))
    """
    if m == 0:
        return 0
    return int(math.floor(math.log2(abs(m))))


def state_rank(s: IntegerState) -> int:
    """
    Rank of a state is the rank of its denominator
    (This encodes the accumulated structural depth)
    """
    return rank(abs(s.d))


# ============================================================================
# PRIME-GATED DYNAMICS
# ============================================================================

def is_prime(n: int) -> bool:
    """
    Primality test (deterministic)
    Primes label irreducible operational gating events
    """
    n = abs(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def prime_gate_test(pair: CoupledPair) -> bool:
    """
    Prime gating admissibility test
    Returns True if the state passes prime gating
    """
    # Check if any component triggers prime gating
    # This is a simplified version - full implementation needs context
    components = [pair.lower.n, pair.lower.d, pair.upper.n, pair.upper.d]
    
    # For now, we check if any prime appears
    # Full engine would have more sophisticated gating rules
    has_prime = any(is_prime(abs(c)) for c in components)
    
    return has_prime


# ============================================================================
# TEMPORAL STRUCTURE
# ============================================================================

class Phase(Enum):
    """Triadic phases: Emission, Memory, Return"""
    EMISSION = "E"
    MEMORY = "M"
    RETURN = "R"


class TemporalStructure:
    """
    Temporal structure: 11 microticks per generation, 3 generations per cycle
    Fundamental cycle length T = 33
    """
    MICROTICKS_PER_GENERATION = 11
    GENERATIONS_PER_CYCLE = 3
    CYCLE_LENGTH = MICROTICKS_PER_GENERATION * GENERATIONS_PER_CYCLE  # 33
    
    def __init__(self):
        self.tick = 0
    
    def current_microtick(self) -> int:
        """Current microtick within generation (1-11)"""
        return (self.tick % self.MICROTICKS_PER_GENERATION) + 1
    
    def current_generation(self) -> int:
        """Current generation within cycle (0-2)"""
        return (self.tick // self.MICROTICKS_PER_GENERATION) % self.GENERATIONS_PER_CYCLE
    
    def current_phase(self) -> Phase:
        """Current triadic phase"""
        gen = self.current_generation()
        if gen == 0:
            return Phase.EMISSION
        elif gen == 1:
            return Phase.MEMORY
        else:
            return Phase.RETURN
    
    def advance(self):
        """Advance one tick"""
        self.tick += 1
    
    def at_cycle_boundary(self) -> bool:
        """Check if we're at an Ω (cycle boundary) event"""
        return self.tick % self.CYCLE_LENGTH == 0


# ============================================================================
# DEVIATION AND STABILIZATION
# ============================================================================

def deviation(pair: CoupledPair) -> int:
    """
    Deviation Δ := |n_U * d_L - n_L * d_U|
    Measures distance from stabilization
    """
    return abs(pair.upper.n * pair.lower.d - pair.lower.n * pair.upper.d)


def is_stabilized(pair: CoupledPair) -> bool:
    """
    A coupled state is stabilized if Δ = 1
    This defines particle emergence
    """
    return deviation(pair) == 1


# ============================================================================
# HISTORY TRACKING
# ============================================================================

class OperationType(Enum):
    BARYCENTRIC = "⊞"
    VACUUM = "⊕"
    INVERSION = "ψ"


@dataclass
class HistoryStep:
    """Single step in evolution history"""
    operation: OperationType
    state_before: CoupledPair
    state_after: CoupledPair
    tick: int
    phase: Phase


class History:
    """
    Complete evolution history from seed to stabilization
    """
    def __init__(self, seed: CoupledPair):
        self.seed = seed
        self.steps: List[HistoryStep] = []
        self.current_state = seed
        self.stabilized = False
        self.rank_profile: List[int] = []
    
    def add_step(self, operation: OperationType, new_state: CoupledPair, 
                 tick: int, phase: Phase):
        """Record evolution step"""
        step = HistoryStep(operation, self.current_state, new_state, tick, phase)
        self.steps.append(step)
        self.current_state = new_state
        
        # Track rank profile
        rank_val = max(state_rank(new_state.lower), state_rank(new_state.upper))
        self.rank_profile.append(rank_val)
        
        # Check stabilization
        if is_stabilized(new_state):
            self.stabilized = True
    
    def max_rank(self) -> int:
        """Maximum rank achieved in this history"""
        if not self.rank_profile:
            return 0
        return max(self.rank_profile)
    
    def __repr__(self):
        status = "STABILIZED" if self.stabilized else "EVOLVING"
        return f"History[{status}, {len(self.steps)} steps, max_rank={self.max_rank()}]"


# ============================================================================
# RESOLUTION EQUIVALENCE
# ============================================================================

def psi_equivalent(h1: History, h2: History) -> bool:
    """
    Two histories are ψ-equivalent if they differ only by 
    ψ-phase permutations within a ψ⁴ orbit
    """
    # Simplified version - full implementation needs orbit tracking
    # For now, check if they have same number of ψ operations mod 4
    psi_count1 = sum(1 for s in h1.steps if s.operation == OperationType.INVERSION)
    psi_count2 = sum(1 for s in h2.steps if s.operation == OperationType.INVERSION)
    
    return (psi_count1 % 4) == (psi_count2 % 4)


def resolution_equivalent(h1: History, h2: History) -> bool:
    """
    Two stabilized histories are resolution-equivalent if:
    1. Both stabilize at Δ = 1
    2. They are ψ-equivalent
    3. They share the same rank profile (max rank)
    4. They stabilize in equivalent triadic positions
    """
    if not (h1.stabilized and h2.stabilized):
        return False
    
    if not is_stabilized(h1.current_state) or not is_stabilized(h2.current_state):
        return False
    
    if not psi_equivalent(h1, h2):
        return False
    
    if h1.max_rank() != h2.max_rank():
        return False
    
    # Simplified triadic position check
    # Full implementation needs more sophisticated equivalence
    
    return True


# ============================================================================
# ENGINE: EVOLUTION AND STABILIZATION SEARCH
# ============================================================================

class RigbyspaceEngine:
    """
    The complete Rigbyspace Engine
    Searches for stabilized histories at specified rank
    """
    
    def __init__(self, max_rank: int = 1, max_steps: int = 1000):
        self.max_rank = max_rank
        self.max_steps = max_steps
        self.temporal = TemporalStructure()
        self.histories: List[History] = []
    
    def evolve_pair(self, pair: CoupledPair, history: History) -> Optional[CoupledPair]:
        """
        Evolve a coupled pair by one step
        Returns new state or None if evolution should terminate
        """
        # Check rank constraint
        current_rank = max(state_rank(pair.lower), state_rank(pair.upper))
        if current_rank > self.max_rank:
            return None
        
        # Check stabilization
        if is_stabilized(pair):
            return pair  # Already stabilized
        
        # Try vacuum step first (mediant-like growth)
        try:
            new_state = CoupledPair(
                vacuum_step(pair.lower, pair.upper),
                pair.upper
            )
            
            # Check if this creates rank growth
            new_rank = max(state_rank(new_state.lower), state_rank(new_state.upper))
            
            # If prime gating fails or rank exceeded, try inversion
            if new_rank > self.max_rank or not prime_gate_test(new_state):
                # Apply ψ-resolution
                new_state = coupled_inversion(pair)
                history.add_step(OperationType.INVERSION, new_state, 
                               self.temporal.tick, self.temporal.current_phase())
            else:
                history.add_step(OperationType.VACUUM, new_state,
                               self.temporal.tick, self.temporal.current_phase())
            
            return new_state
            
        except:
            # If vacuum step fails, try inversion
            new_state = coupled_inversion(pair)
            history.add_step(OperationType.INVERSION, new_state,
                           self.temporal.tick, self.temporal.current_phase())
            return new_state
    
    def search_from_seed(self, seed: CoupledPair) -> History:
        """
        Evolve from a seed until stabilization or max steps
        """
        history = History(seed)
        current_pair = seed
        
        for step in range(self.max_steps):
            self.temporal.advance()
            
            # Evolve
            new_pair = self.evolve_pair(current_pair, history)
            
            if new_pair is None:
                # Rank exceeded, terminate
                break
            
            if is_stabilized(new_pair):
                # Stabilized!
                history.stabilized = True
                break
            
            current_pair = new_pair
        
        return history
    
    def enumerate_rank_1(self, num_seeds: int = 100) -> dict:
        """
        Enumerate resolution classes at Rank-1
        Search multiple seeds and classify by resolution equivalence
        """
        print(f"\n{'='*70}")
        print(f"ENUMERATING RANK-1 RESOLUTION CLASSES")
        print(f"{'='*70}\n")
        
        resolution_classes = {}
        stabilized_count = 0
        
        # Try various seeds
        seeds = []
        
        # Simple seeds
        for i in range(1, 10):
            for j in range(1, 10):
                seeds.append(CoupledPair(
                    IntegerState(1, i),
                    IntegerState(1, j)
                ))
        
        # Prime-based seeds
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        for p1 in primes[:5]:
            for p2 in primes[:5]:
                seeds.append(CoupledPair(
                    IntegerState(1, p1),
                    IntegerState(1, p2)
                ))
        
        # Asymmetric seeds
        for i in range(1, 8):
            seeds.append(CoupledPair(
                IntegerState(i, i+1),
                IntegerState(i+1, i+2)
            ))
        
        print(f"Testing {len(seeds)} different seeds...\n")
        
        for idx, seed in enumerate(seeds):
            if idx % 20 == 0:
                print(f"Progress: {idx}/{len(seeds)} seeds tested...")
            
            history = self.search_from_seed(seed)
            
            if history.stabilized and history.max_rank() <= self.max_rank:
                stabilized_count += 1
                
                # Classify by deviation pattern and rank
                key = (deviation(history.current_state), history.max_rank())
                
                if key not in resolution_classes:
                    resolution_classes[key] = []
                
                resolution_classes[key].append(history)
        
        print(f"\n{'='*70}")
        print(f"RESULTS:")
        print(f"  Total seeds tested: {len(seeds)}")
        print(f"  Stabilized histories: {stabilized_count}")
        print(f"  Resolution classes found: {len(resolution_classes)}")
        print(f"{'='*70}\n")
        
        return resolution_classes


# ============================================================================
# ANALYSIS AND DISPLAY
# ============================================================================

def analyze_resolution_classes(classes: dict):
    """
    Analyze resolution classes and compare to theoretical predictions
    """
    print("\nRESOLUTION CLASS ANALYSIS:")
    print("-" * 70)
    
    total_states = 0
    class_sizes = []
    
    for key, histories in sorted(classes.items()):
        deviation, rank = key
        count = len(histories)
        total_states += count
        class_sizes.append(count)
        
        print(f"  Class (Δ={deviation}, rank={rank}): {count} histories")
        
        # Show sample
        if histories:
            sample = histories[0]
            print(f"    Sample: {sample.seed} -> {sample.current_state}")
            print(f"    Steps: {len(sample.steps)}")
    
    print("-" * 70)
    print(f"  TOTAL STATES: {total_states}")
    print("-" * 70)
    
    print("\nTHEORETICAL PREDICTION (Rank-1):")
    print("  Resolution Classes: {64, 72, 128}")
    print("  States: 16 + 12 + 16 = 44 fermionic states")
    print("-" * 70)
    
    print("\nNOTE:")
    print("  This is a simplified implementation.")
    print("  Full enumeration requires:")
    print("    - Complete prime gating rules")
    print("    - Full ψ-equivalence orbits")
    print("    - Proper triadic position tracking")
    print("    - Internal degree of freedom counting")
    print("    - Global symmetry quotients")
    print("=" * 70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run the Rigbyspace Engine and enumerate Rank-1 particles
    """
    print("\n" + "="*70)
    print("RIGBYSPACE TRIADIC ENGINE - COMPUTATIONAL IMPLEMENTATION")
    print("="*70)
    print("\nBased on theoretical framework by D. Veneziano")
    print("Implementation: Direct from triadic_engine.pdf specifications")
    print("\n" + "="*70)
    
    # Test basic operators
    print("\nTEST 1: Basic Operators")
    print("-" * 70)
    s1 = IntegerState(1, 2)
    s2 = IntegerState(3, 4)
    print(f"  s1 = {s1}")
    print(f"  s2 = {s2}")
    print(f"  s1 ⊞ s2 = {barycentric_addition(s1, s2)}")
    print(f"  s1 ⊕ s2 = {vacuum_step(s1, s2)}")
    
    pair = CoupledPair(s1, s2)
    print(f"  pair = {pair}")
    print(f"  ψ(pair) = {coupled_inversion(pair)}")
    print(f"  ψ²(pair) = {coupled_inversion(coupled_inversion(pair))}")
    print(f"  ψ⁴(pair) = {coupled_inversion(coupled_inversion(coupled_inversion(coupled_inversion(pair))))}")
    print(f"  Deviation(pair) = {deviation(pair)}")
    print(f"  Stabilized? {is_stabilized(pair)}")
    
    # Test stabilization search
    print("\n\nTEST 2: Stabilization Search")
    print("-" * 70)
    engine = RigbyspaceEngine(max_rank=1, max_steps=100)
    
    test_seed = CoupledPair(IntegerState(1, 1), IntegerState(1, 2))
    print(f"  Seed: {test_seed}")
    
    history = engine.search_from_seed(test_seed)
    print(f"  Result: {history}")
    print(f"  Final state: {history.current_state}")
    print(f"  Deviation: {deviation(history.current_state)}")
    
    # Enumerate Rank-1 resolution classes
    print("\n\nTEST 3: Rank-1 Enumeration")
    classes = engine.enumerate_rank_1(num_seeds=100)
    
    analyze_resolution_classes(classes)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNEXT STEPS:")
    print("  1. Implement full prime gating rules")
    print("  2. Add proper resolution equivalence checking")
    print("  3. Count internal degrees of freedom")
    print("  4. Apply global symmetry quotients")
    print("  5. Verify 44 fermion count at Rank-1")
    print("  6. Extend to Rank-2 (88 bosons)")
    print("  7. Verify discrete running (Ar+1 = 4×Ar)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
