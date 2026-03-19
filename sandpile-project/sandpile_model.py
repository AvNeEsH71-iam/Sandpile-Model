"""
sandpile_model.py
=================
Bak-Tang-Wiesenfeld (BTW) Sandpile Model for Foam Collapse Simulation.

Think of this like a pile of sand where every time you add one grain,
it might cause a small slide, or sometimes a massive avalanche.
This exact same math describes how bubbles burst in foams!

Author: Avneesh Singh (MS23249)
Course: IDC621 - Module 1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from enum import Enum


class BoundaryCondition(Enum):
    """How grains behave at the edges of the grid."""
    OPEN = "open"         # Grains fall off the edge (energy is lost, like a real sandpile)
    CLOSED = "closed"     # Grains bounce back from the edge
    PERIODIC = "periodic" # The grid wraps around like a donut (toroidal topology)


@dataclass
class TopplingEvent:
    """Records what happened at one single topple."""
    position: Tuple[int, int]
    grains_dropped: int
    neighbors_affected: int
    energy_released: float


@dataclass
class Avalanche:
    """Everything that happened during one full avalanche."""
    size: int                        # Total number of topplings in this avalanche
    duration: int                    # How many time steps it lasted
    area: int                        # Number of unique sites that toppled
    energy_dissipated: float         # Total energy released
    start_position: Tuple[int, int]  # Where the grain that triggered it was dropped
    events: List[TopplingEvent] = field(default_factory=list)


@dataclass
class SandpileStats:
    """Running statistics across the whole simulation."""
    total_avalanches: int = 0
    total_grains_added: int = 0
    total_energy_dissipated: float = 0.0

    avalanche_sizes: List[int] = field(default_factory=list)
    avalanche_durations: List[int] = field(default_factory=list)
    avalanche_energies: List[float] = field(default_factory=list)

    # Fitted power-law exponent tau (filled in by the analyzer)
    size_distribution_exponent: Optional[float] = None

    toppling_counts: Optional[np.ndarray] = None


class SandpileModel:
    """
    The BTW Sandpile Model.

    Imagine a grid of cells. Each cell holds some number of sand grains.
    When a cell has 4 or more grains, it "topples": it passes one grain
    to each of its 4 neighbors and loses 4 grains total.
    If a neighbor now has 4+ grains, it topples too. This chain reaction
    is an avalanche.

    In foam physics:
      - Each cell = a small region of bubble film
      - Grains      = local stress / pressure
      - Toppling    = a film rupture event
      - Avalanche   = a cascade of bubble bursts

    Parameters
    ----------
    shape : tuple
        Size of the grid, e.g. (50, 50) means a 50x50 grid.
    critical_threshold : int
        How many grains a cell needs before it topples. Default is 4
        (the standard choice for a 2D square grid, one per neighbor).
    boundary_condition : BoundaryCondition
        What happens at the edges.
    grain_energy : float
        Energy released each time one grain is transferred.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (50, 50),
        critical_threshold: int = 4,
        boundary_condition: BoundaryCondition = BoundaryCondition.OPEN,
        grain_energy: float = 1.0
    ):
        self.shape = shape
        self.critical_threshold = critical_threshold
        self.boundary_condition = boundary_condition
        self.grain_energy = grain_energy

        # Main grid: how many grains are at each cell
        self.lattice = np.zeros(shape, dtype=np.int32)

        # How many times has each cell toppled overall?
        self.toppling_counts = np.zeros(shape, dtype=np.int32)

        self.stats = SandpileStats()
        self._avalanche_events: List[TopplingEvent] = []

        # Pre-compute neighbor lists so we do not recompute during simulation
        self._neighbors = self._compute_neighbors()

    def _compute_neighbors(self) -> np.ndarray:
        """
        For every cell, figure out which cells are its neighbors.
        Pre-computing this once speeds things up a lot during the simulation.
        """
        rows, cols = self.shape
        neighbors = np.empty((rows, cols), dtype=object)

        for i in range(rows):
            for j in range(cols):
                nb = []

                # Up
                if i > 0:
                    nb.append((i - 1, j))
                elif self.boundary_condition == BoundaryCondition.PERIODIC:
                    nb.append((rows - 1, j))

                # Down
                if i < rows - 1:
                    nb.append((i + 1, j))
                elif self.boundary_condition == BoundaryCondition.PERIODIC:
                    nb.append((0, j))

                # Left
                if j > 0:
                    nb.append((i, j - 1))
                elif self.boundary_condition == BoundaryCondition.PERIODIC:
                    nb.append((i, cols - 1))

                # Right
                if j < cols - 1:
                    nb.append((i, j + 1))
                elif self.boundary_condition == BoundaryCondition.PERIODIC:
                    nb.append((i, 0))

                neighbors[i, j] = nb

        return neighbors

    def add_grain(self, position: Optional[Tuple[int, int]] = None) -> Optional[Avalanche]:
        """
        Drop one grain onto the grid.

        If you do not specify where, it lands at a random cell.
        If this causes a cell to reach the critical threshold, an avalanche starts.

        Parameters
        ----------
        position : (row, col), optional
            Where to drop the grain. Random if not given.

        Returns
        -------
        Avalanche or None
            Returns the avalanche details if one happened, otherwise None.
        """
        if position is None:
            position = (
                np.random.randint(0, self.shape[0]),
                np.random.randint(0, self.shape[1])
            )

        self.lattice[position] += 1
        self.stats.total_grains_added += 1

        if self._should_topple(position):
            return self._relax(position)

        return None

    def _should_topple(self, position: Tuple[int, int]) -> bool:
        """Is this cell at or above the critical threshold?"""
        return bool(self.lattice[position] >= self.critical_threshold)

    def _relax(self, start_position: Tuple[int, int]) -> Avalanche:
        """
        Let the grid settle by processing all topplings until everything is stable.

        Uses a parallel update rule: every currently unstable cell topples
        at the same time in each time step (one "generation").
        """
        self._avalanche_events = []
        unstable = self._find_unstable_sites()

        duration = 0
        total_energy = 0.0

        while unstable:
            duration += 1
            next_unstable: set = set()

            for pos in unstable:
                event = self._topple(pos)
                self._avalanche_events.append(event)
                total_energy += event.energy_released

                # Any neighbor that is now unstable joins the next wave
                for nb in self._neighbors[pos]:
                    if self._should_topple(nb):
                        next_unstable.add(nb)

            unstable = next_unstable

        unique_sites = {e.position for e in self._avalanche_events}

        avalanche = Avalanche(
            size=len(self._avalanche_events),
            duration=duration,
            area=len(unique_sites),
            energy_dissipated=total_energy,
            start_position=start_position,
            events=self._avalanche_events.copy()
        )

        # Record in global stats
        self.stats.total_avalanches += 1
        self.stats.total_energy_dissipated += total_energy
        self.stats.avalanche_sizes.append(avalanche.size)
        self.stats.avalanche_durations.append(avalanche.duration)
        self.stats.avalanche_energies.append(avalanche.energy_dissipated)

        self._avalanche_events = []
        return avalanche

    def _find_unstable_sites(self) -> set:
        """Return all cells that are currently at or above the threshold."""
        rows, cols = self.shape
        return {
            (i, j)
            for i in range(rows)
            for j in range(cols)
            if self.lattice[i, j] >= self.critical_threshold
        }

    def _topple(self, position: Tuple[int, int]) -> TopplingEvent:
        """
        Topple one cell.

        The cell loses `critical_threshold` grains.
        Each neighbor gains exactly 1 grain.
        For open boundaries, grains that would go off the edge are just lost
        (this is the dissipation that keeps the system from exploding).

        Bug fix from original: energy calculation now correctly counts
        only the grains actually transferred to neighbors, not ghost grains.
        """
        self.lattice[position] -= self.critical_threshold
        self.toppling_counts[position] += 1

        neighbors = self._neighbors[position]
        grains_transferred = 0

        for nb in neighbors:
            self.lattice[nb] += 1
            grains_transferred += 1

        # For open boundaries, some grains are lost (dissipated at the edges)
        # The number of lost grains = 4 minus actual neighbors
        grains_lost = self.critical_threshold - grains_transferred

        # Energy released = grains actually moved (transferred + dissipated)
        energy_released = self.critical_threshold * self.grain_energy

        return TopplingEvent(
            position=position,
            grains_dropped=grains_transferred,
            neighbors_affected=len(neighbors),
            energy_released=energy_released
        )

    def drive_to_critical_state(self, n_grains: int) -> List[Avalanche]:
        """
        Keep adding grains one at a time until the system reaches its critical state.

        In a fresh grid, the first thousands of grains just pile up quietly.
        Eventually the grid gets crowded enough that avalanches start happening
        regularly. That is the self-organized critical state.

        Parameters
        ----------
        n_grains : int
            How many grains to add total.

        Returns
        -------
        list of Avalanche
            Every avalanche that occurred.
        """
        avalanches = []
        for _ in range(n_grains):
            result = self.add_grain()
            if result is not None:
                avalanches.append(result)
        return avalanches

    def get_stable_state(self) -> np.ndarray:
        """Return a snapshot of the current grid."""
        return self.lattice.copy()

    def compute_energy_density(self) -> np.ndarray:
        """
        Map the grain count to energy density.
        In the foam context this represents local stress in bubble films.
        """
        return self.lattice.astype(float) * self.grain_energy

    def reset(self):
        """Wipe everything and start fresh."""
        self.lattice = np.zeros(self.shape, dtype=np.int32)
        self.toppling_counts = np.zeros(self.shape, dtype=np.int32)
        self.stats = SandpileStats()


class FoamSandpileAnalyzer:
    """
    Post-simulation analysis tools.

    After running the sandpile, use this class to dig into the statistics:
    power-law fits, energy rates, correlation functions, and more.

    Parameters
    ----------
    model : SandpileModel
        A model that has already been run.
    """

    def __init__(self, model: SandpileModel):
        self.model = model
        self.stats = model.stats

    def compute_size_distribution(self, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bin the avalanche sizes to build a probability distribution.

        Uses logarithmic bins so that the power-law behavior shows up
        as a straight line on a log-log plot.

        Returns
        -------
        bin_centers : np.ndarray
        probabilities : np.ndarray
        """
        sizes = np.array(self.stats.avalanche_sizes)
        if len(sizes) == 0:
            return np.array([]), np.array([])

        log_bins = np.logspace(
            np.log10(max(1, sizes.min())),
            np.log10(sizes.max() + 1),
            bins + 1
        )

        hist, bin_edges = np.histogram(sizes, bins=log_bins)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

        total = hist.sum()
        if total > 0:
            probabilities = hist / total / np.diff(bin_edges)
        else:
            probabilities = np.zeros_like(bin_centers)

        return bin_centers, probabilities

    def fit_power_law_exponent(
        self,
        min_size: int = 10,
        method: str = "mle"
    ) -> Optional[float]:
        """
        Estimate the power-law exponent tau for P(s) ~ s^(-tau).

        The theoretical value for the 2D BTW sandpile is around 1.2 to 1.5.

        Parameters
        ----------
        min_size : int
            Only use avalanches larger than this (small ones are too noisy).
        method : str
            "mle" for maximum likelihood (more accurate),
            "ols" for ordinary least squares on the log-log plot.

        Returns
        -------
        tau : float or None
        """
        sizes = np.array(self.stats.avalanche_sizes)
        sizes = sizes[sizes >= min_size]

        if len(sizes) < 10:
            return None

        if method == "mle":
            # Hill estimator (MLE for power law)
            x_min = float(min_size)
            tau = 1.0 + len(sizes) / np.sum(np.log(sizes / x_min))
        elif method == "ols":
            unique, counts = np.unique(sizes, return_counts=True)
            # Need at least 2 unique sizes for a linear fit
            if len(unique) < 2:
                return None
            coeffs = np.polyfit(np.log(unique), np.log(counts), 1)
            tau = -coeffs[0]
        else:
            raise ValueError(f"Unknown fitting method: {method}. Use 'mle' or 'ols'.")

        self.stats.size_distribution_exponent = tau
        return tau

    def compute_energy_dissipation_rate(self) -> float:
        """
        Average energy released per grain added.

        In foam terms: how much surface energy is released each time
        you squeeze the foam a tiny bit more.
        """
        if self.stats.total_grains_added == 0:
            return 0.0
        return self.stats.total_energy_dissipated / self.stats.total_grains_added

    def compute_temporal_correlation(self, max_lag: int = 100) -> np.ndarray:
        """
        Check whether large avalanches tend to bunch together in time.

        If the correlation drops to zero quickly, avalanches are independent.
        If it stays high, big events tend to follow other big events.

        Parameters
        ----------
        max_lag : int
            How many steps back to check.

        Returns
        -------
        correlations : np.ndarray of length max_lag
        """
        sizes = np.array(self.stats.avalanche_sizes, dtype=float)
        n = len(sizes)

        if n < 2:
            return np.array([])

        max_lag = min(max_lag, n // 2)
        if max_lag < 1:
            return np.array([1.0])

        mean = sizes.mean()
        var = sizes.var()

        if var == 0:
            return np.ones(max_lag)

        correlations = np.zeros(max_lag)
        correlations[0] = 1.0
        for lag in range(1, max_lag):
            c = np.sum((sizes[:-lag] - mean) * (sizes[lag:] - mean))
            correlations[lag] = c / ((n - lag) * var)

        return correlations

    def compute_size_duration_scaling(self) -> Optional[float]:
        """
        Fit D ~ S^gamma, the scaling between avalanche duration and size.

        For self-organized critical systems this should follow a power law.
        Returns the exponent gamma, or None if there is not enough data.
        """
        sizes = np.array(self.stats.avalanche_sizes)
        durations = np.array(self.stats.avalanche_durations)

        mask = (sizes > 1) & (durations > 0)
        if mask.sum() < 10:
            return None

        s = sizes[mask].astype(float)
        d = durations[mask].astype(float)

        coeffs = np.polyfit(np.log(s), np.log(d), 1)
        return float(coeffs[0])

    def get_summary(self) -> dict:
        """One-stop shop: run all analyses and return a dictionary of results."""
        tau = self.fit_power_law_exponent()
        gamma = self.compute_size_duration_scaling()

        sizes = self.stats.avalanche_sizes
        durations = self.stats.avalanche_durations

        return {
            "total_avalanches": self.stats.total_avalanches,
            "total_grains_added": self.stats.total_grains_added,
            "total_energy_dissipated": self.stats.total_energy_dissipated,
            "mean_avalanche_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_avalanche_size": int(max(sizes)) if sizes else 0,
            "mean_avalanche_duration": float(np.mean(durations)) if durations else 0.0,
            "max_avalanche_duration": int(max(durations)) if durations else 0,
            "energy_dissipation_rate": self.compute_energy_dissipation_rate(),
            "power_law_exponent_tau": tau,
            "size_duration_exponent_gamma": gamma,
            "lattice_shape": list(self.model.shape),
            "critical_threshold": self.model.critical_threshold,
            "boundary_condition": self.model.boundary_condition.value,
        }


def run_simulation(
    shape: Tuple[int, int] = (50, 50),
    n_grains: int = 10000,
    critical_threshold: int = 4,
    boundary_condition: str = "open",
    grain_energy: float = 1.0,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[SandpileModel, "FoamSandpileAnalyzer", dict]:
    """
    Run a complete sandpile simulation from scratch.

    Parameters
    ----------
    shape : tuple
        Grid size, e.g. (50, 50).
    n_grains : int
        Total grains to drop.
    critical_threshold : int
        Toppling threshold (4 is standard for 2D).
    boundary_condition : str
        "open", "closed", or "periodic".
    grain_energy : float
        Energy per grain moved.
    seed : int, optional
        Set this for reproducible results.
    verbose : bool
        Print progress updates.

    Returns
    -------
    model, analyzer, summary
    """
    if seed is not None:
        np.random.seed(seed)

    bc = BoundaryCondition(boundary_condition)

    if verbose:
        print(f"Starting simulation on {shape[0]}x{shape[1]} grid...")
        print(f"Dropping {n_grains:,} grains with threshold={critical_threshold}, "
              f"boundary={boundary_condition}")

    model = SandpileModel(
        shape=shape,
        critical_threshold=critical_threshold,
        boundary_condition=bc,
        grain_energy=grain_energy
    )

    model.drive_to_critical_state(n_grains)

    analyzer = FoamSandpileAnalyzer(model)
    summary = analyzer.get_summary()

    if verbose:
        print("\nDone! Results:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    return model, analyzer, summary


if __name__ == "__main__":
    print("=" * 60)
    print("Sandpile Model  (Foam Collapse Simulation)")
    print("=" * 60)

    model, analyzer, summary = run_simulation(
        shape=(50, 50),
        n_grains=50000,
        boundary_condition="open",
        seed=42
    )
