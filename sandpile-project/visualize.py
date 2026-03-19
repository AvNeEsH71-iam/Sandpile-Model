"""
visualize.py
============
Generate all figures for the sandpile foam collapse simulation.

Run this after the main simulation to get publication-quality plots.
All figures are saved to the figures/ folder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os

from sandpile_model import SandpileModel, FoamSandpileAnalyzer, run_simulation


# Make sure the output folder exists
os.makedirs("figures", exist_ok=True)

# Global style settings (clean, publication-ready look)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

BLUE  = "#2E86AB"
RED   = "#E84855"
GREEN = "#3BB273"
GOLD  = "#F4A261"


def plot_lattice_state(model: SandpileModel, save: bool = True) -> plt.Figure:
    """
    Heatmap of the final grain count at every cell.

    Cells near the threshold (value = 3) are shown in red/dark.
    Empty or near-empty cells are shown in yellow/light.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        model.lattice,
        cmap="YlOrRd",
        vmin=0,
        vmax=model.critical_threshold - 1,
        interpolation="nearest",
        origin="upper"
    )
    cbar = fig.colorbar(im, ax=ax, label="Grains per site")
    ax.set_title(
        f"Final Sandpile Lattice State\n(Critical threshold = {model.critical_threshold})"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Highlight sites that are one grain away from toppling
    near_critical = np.ma.masked_less(model.lattice, model.critical_threshold - 1)
    ax.contour(near_critical, levels=[model.critical_threshold - 1.5],
               colors="black", linewidths=0.4, alpha=0.4)

    fig.tight_layout()
    if save:
        fig.savefig("figures/01_lattice_state.png")
    return fig


def plot_toppling_heatmap(model: SandpileModel, save: bool = True) -> plt.Figure:
    """
    Show which parts of the grid toppled the most during the whole simulation.

    The color scale is logarithmic because the differences are huge.
    You will notice the center topples far more than the edges (for open boundaries).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    tc = model.toppling_counts.astype(float)
    tc[tc == 0] = 0.1  # avoid log(0)

    im = ax.imshow(
        tc,
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=0.5, vmax=tc.max()),
        interpolation="nearest",
        origin="upper"
    )
    cbar = fig.colorbar(im, ax=ax, label="Number of topplings (log scale)")
    ax.set_title(
        f"Toppling Activity Map\n(Total: {model.toppling_counts.sum():,} topplings)"
    )
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    fig.tight_layout()
    if save:
        fig.savefig("figures/02_toppling_heatmap.png")
    return fig


def plot_avalanche_distribution(analyzer: FoamSandpileAnalyzer, save: bool = True) -> plt.Figure:
    """
    Plot the avalanche size distribution on both a regular histogram
    and a log-log plot to show the power-law (straight line on log-log = power law).
    """
    sizes = np.array(analyzer.stats.avalanche_sizes)
    tau = analyzer.fit_power_law_exponent()
    bin_centers, probs = analyzer.compute_size_distribution(bins=25)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    tau_str = f"{tau:.4f}" if tau is not None else "N/A"
    fig.suptitle(
        f"Avalanche Size Distribution  (tau = {tau_str})",
        fontsize=14, fontweight="bold"
    )

    # Left: raw histogram
    log_bins = np.logspace(np.log10(max(1, sizes.min())), np.log10(sizes.max() + 1), 30)
    ax1.hist(sizes, bins=log_bins, color=BLUE, edgecolor="white", alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Avalanche Size (number of topplings)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Avalanche Size Histogram")

    # Right: probability density + power-law fit
    valid = probs > 0
    ax2.scatter(bin_centers[valid], probs[valid],
                color=BLUE, s=30, alpha=0.8, label="Simulation data", zorder=3)

    if tau is not None and len(bin_centers[valid]) > 0:
        s_range = np.logspace(
            np.log10(bin_centers[valid].min()),
            np.log10(bin_centers[valid].max()),
            200
        )
        # Normalize the power-law curve to the data
        ref_idx = len(bin_centers[valid]) // 4
        ref_x = bin_centers[valid][ref_idx]
        ref_y = probs[valid][ref_idx]
        scale = ref_y * (ref_x ** tau)
        ax2.plot(s_range, scale * s_range ** (-tau),
                 color=RED, linestyle="--", linewidth=2,
                 label=f"Power law fit (tau = {tau:.3f})", zorder=2)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Avalanche Size")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Size Distribution (Log-Log Scale)")
    ax2.legend()

    fig.tight_layout()
    if save:
        fig.savefig("figures/03_avalanche_distribution.png")
    return fig


def plot_energy_dissipation(analyzer: FoamSandpileAnalyzer, save: bool = True) -> plt.Figure:
    """
    Show how energy was released over the course of the simulation.

    The cumulative energy should grow roughly linearly once the system
    hits its critical state (constant energy release per grain added).
    """
    energies = np.array(analyzer.stats.avalanche_energies)
    cumulative = np.cumsum(energies)
    rate = analyzer.compute_energy_dissipation_rate()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Energy Dissipation Analysis\nRate: {rate:.4f} per grain added",
        fontsize=14, fontweight="bold"
    )

    # Left: cumulative energy
    ax1.plot(cumulative / 1e6, color=BLUE, linewidth=1.5)
    # Add a trend line showing constant rate
    x_trend = np.array([0, len(cumulative)])
    mean_per_av = cumulative[-1] / len(cumulative) if len(cumulative) > 0 else 0
    ax1.plot(x_trend, mean_per_av * x_trend / 1e6,
             color=RED, linestyle="--", linewidth=1.5,
             label=f"Rate: {mean_per_av:.1f} per avalanche")
    ax1.set_xlabel("Avalanche Number")
    ax1.set_ylabel("Cumulative Energy (x10^6)")
    ax1.set_title("Cumulative Energy Dissipation")
    ax1.legend()

    # Right: energy per avalanche histogram
    log_bins = np.logspace(np.log10(max(1, energies.min())),
                           np.log10(energies.max() + 1), 30)
    ax2.hist(energies, bins=log_bins, color=GOLD, edgecolor="white", alpha=0.9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Energy Dissipated per Avalanche")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Energy Distribution per Avalanche")

    fig.tight_layout()
    if save:
        fig.savefig("figures/04_energy_dissipation.png")
    return fig


def plot_temporal_correlation(analyzer: FoamSandpileAnalyzer, save: bool = True) -> plt.Figure:
    """
    Does a big avalanche predict the next one will also be big?

    If yes, the correlation stays high for many lags.
    For a truly critical system, it drops fast to zero (avalanches are independent).
    """
    corr = analyzer.compute_temporal_correlation(max_lag=100)
    if len(corr) == 0:
        print("Not enough data for temporal correlation plot.")
        return None

    noise_threshold = 2.0 / np.sqrt(len(analyzer.stats.avalanche_sizes))

    fig, ax = plt.subplots(figsize=(9, 5))

    lags = np.arange(len(corr))
    ax.plot(lags, corr, color=RED, linewidth=1.2, marker="o", markersize=3, alpha=0.8)
    ax.axhline(noise_threshold, color="gray", linestyle=":",
               linewidth=1.2, label=f"Noise threshold (~{noise_threshold:.3f})")
    ax.axhline(-noise_threshold, color="gray", linestyle=":", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Time Lag (avalanches)")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("Temporal Correlation of Avalanche Sizes")
    ax.legend()
    ax.set_ylim(-0.15, 1.05)

    fig.tight_layout()
    if save:
        fig.savefig("figures/05_temporal_correlation.png")
    return fig


def plot_size_duration_scaling(analyzer: FoamSandpileAnalyzer, save: bool = True) -> plt.Figure:
    """
    Show how long avalanches take versus how big they are.

    For SOC systems this follows a power law: D ~ S^gamma.
    On a log-log plot this appears as a straight line.
    """
    sizes = np.array(analyzer.stats.avalanche_sizes)
    durations = np.array(analyzer.stats.avalanche_durations)
    gamma = analyzer.compute_size_duration_scaling()

    mask = (sizes > 1) & (durations > 0)
    s = sizes[mask]
    d = durations[mask]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(s, d, color=GREEN, s=4, alpha=0.3, label="Avalanche events")

    if gamma is not None and len(s) > 0:
        s_range = np.logspace(np.log10(s.min()), np.log10(s.max()), 200)
        # Fit the scale factor
        log_s = np.log(s.astype(float))
        log_d = np.log(d.astype(float))
        intercept = np.mean(log_d) - gamma * np.mean(log_s)
        ax.plot(s_range, np.exp(intercept) * s_range ** gamma,
                color=RED, linestyle="--", linewidth=2,
                label=f"Scaling: D ~ S^{gamma:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Avalanche Size (topplings)")
    ax.set_ylabel("Duration (generations)")
    ax.set_title("Avalanche Size vs Duration Scaling")
    ax.legend()

    fig.tight_layout()
    if save:
        fig.savefig("figures/06_size_duration_scaling.png")
    return fig


def plot_summary_dashboard(model: SandpileModel, analyzer: FoamSandpileAnalyzer,
                            save: bool = True) -> plt.Figure:
    """
    A single figure with six panels summarizing the whole simulation.
    Good for a quick overview or a report cover figure.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax_lattice   = fig.add_subplot(gs[0, 0])
    ax_toppling  = fig.add_subplot(gs[0, 1])
    ax_dist      = fig.add_subplot(gs[0, 2])
    ax_energy    = fig.add_subplot(gs[1, 0])
    ax_corr      = fig.add_subplot(gs[1, 1])
    ax_scaling   = fig.add_subplot(gs[1, 2])

    # 1. Lattice state
    im1 = ax_lattice.imshow(model.lattice, cmap="YlOrRd",
                             vmin=0, vmax=model.critical_threshold - 1,
                             interpolation="nearest")
    fig.colorbar(im1, ax=ax_lattice, label="Grains")
    ax_lattice.set_title("Final Lattice State")
    ax_lattice.set_xlabel("Column")
    ax_lattice.set_ylabel("Row")

    # 2. Toppling heatmap
    tc = model.toppling_counts.astype(float)
    tc[tc == 0] = 0.1
    im2 = ax_toppling.imshow(tc, cmap="viridis",
                              norm=mcolors.LogNorm(vmin=0.5, vmax=tc.max()),
                              interpolation="nearest")
    fig.colorbar(im2, ax=ax_toppling, label="Topplings (log)")
    ax_toppling.set_title("Toppling Activity")
    ax_toppling.set_xlabel("Column")
    ax_toppling.set_ylabel("Row")

    # 3. Size distribution
    sizes = np.array(analyzer.stats.avalanche_sizes)
    tau = analyzer.fit_power_law_exponent()
    bin_centers, probs = analyzer.compute_size_distribution(bins=20)
    valid = probs > 0
    ax_dist.scatter(bin_centers[valid], probs[valid], color=BLUE, s=20, alpha=0.8)
    if tau is not None and valid.sum() > 0:
        ref_idx = len(bin_centers[valid]) // 4
        ref_x = bin_centers[valid][ref_idx]
        ref_y = probs[valid][ref_idx]
        scale = ref_y * (ref_x ** tau)
        s_range = np.logspace(np.log10(bin_centers[valid].min()),
                              np.log10(bin_centers[valid].max()), 200)
        ax_dist.plot(s_range, scale * s_range ** (-tau),
                     color=RED, linestyle="--", linewidth=1.8,
                     label=f"tau = {tau:.3f}")
    ax_dist.set_xscale("log")
    ax_dist.set_yscale("log")
    ax_dist.set_title("Size Distribution")
    ax_dist.set_xlabel("Avalanche Size")
    ax_dist.set_ylabel("Probability Density")
    if tau is not None:
        ax_dist.legend(fontsize=9)

    # 4. Cumulative energy
    energies = np.array(analyzer.stats.avalanche_energies)
    cumulative = np.cumsum(energies)
    ax_energy.plot(cumulative / 1e6, color=GOLD, linewidth=1.2)
    ax_energy.set_xlabel("Avalanche Number")
    ax_energy.set_ylabel("Cumulative Energy (x10^6)")
    ax_energy.set_title("Energy Dissipation")

    # 5. Temporal correlation
    corr = analyzer.compute_temporal_correlation(max_lag=100)
    if len(corr) > 0:
        ax_corr.plot(np.arange(len(corr)), corr, color=RED, linewidth=1.0,
                     marker="o", markersize=2, alpha=0.7)
        ax_corr.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_corr.set_xlabel("Time Lag")
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_title("Temporal Correlation")

    # 6. Size-duration scaling
    durations = np.array(analyzer.stats.avalanche_durations)
    gamma = analyzer.compute_size_duration_scaling()
    mask = (sizes > 1) & (durations > 0)
    if mask.sum() > 0:
        ax_scaling.scatter(sizes[mask], durations[mask],
                           color=GREEN, s=2, alpha=0.25)
        if gamma is not None:
            s_range = np.logspace(np.log10(sizes[mask].min()),
                                  np.log10(sizes[mask].max()), 200)
            log_s = np.log(sizes[mask].astype(float))
            log_d = np.log(durations[mask].astype(float))
            intercept = np.mean(log_d) - gamma * np.mean(log_s)
            ax_scaling.plot(s_range, np.exp(intercept) * s_range ** gamma,
                            color=RED, linestyle="--", linewidth=2,
                            label=f"gamma = {gamma:.2f}")
            ax_scaling.legend(fontsize=9)
    ax_scaling.set_xscale("log")
    ax_scaling.set_yscale("log")
    ax_scaling.set_xlabel("Avalanche Size")
    ax_scaling.set_ylabel("Duration")
    ax_scaling.set_title("Size-Duration Scaling")

    summary = analyzer.get_summary()
    fig.suptitle(
        f"Sandpile Foam Collapse Simulation  |  {summary['lattice_shape'][0]}x{summary['lattice_shape'][1]} grid  "
        f"|  {summary['total_grains_added']:,} grains  |  {summary['total_avalanches']:,} avalanches",
        fontsize=13, fontweight="bold", y=1.01
    )

    if save:
        fig.savefig("figures/00_summary_dashboard.png")
    return fig


def generate_all_figures(model: SandpileModel, analyzer: FoamSandpileAnalyzer):
    """Generate and save every figure in one call."""
    print("Generating figures...")
    plot_lattice_state(model)
    print("  [1/6] Lattice state saved.")
    plot_toppling_heatmap(model)
    print("  [2/6] Toppling heatmap saved.")
    plot_avalanche_distribution(analyzer)
    print("  [3/6] Avalanche distribution saved.")
    plot_energy_dissipation(analyzer)
    print("  [4/6] Energy dissipation saved.")
    plot_temporal_correlation(analyzer)
    print("  [5/6] Temporal correlation saved.")
    plot_size_duration_scaling(analyzer)
    print("  [6/6] Size-duration scaling saved.")
    plot_summary_dashboard(model, analyzer)
    print("  Dashboard saved.")
    print("\nAll figures saved to the figures/ folder!")


if __name__ == "__main__":
    print("Running simulation for visualization (seed=42, 50k grains)...")
    model, analyzer, summary = run_simulation(
        shape=(50, 50),
        n_grains=50000,
        boundary_condition="open",
        seed=42
    )
    generate_all_figures(model, analyzer)
    plt.show()
