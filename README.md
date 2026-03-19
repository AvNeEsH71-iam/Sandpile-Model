# Sandpile Model for Foam Collapse

**Course:** IDC621 | Module 1
**Author:** Avneesh Singh (MS23249)

---

## What is this project?


This project simulates that exact kind of behavior using something called the **Bak-Tang-Wiesenfeld (BTW) Sandpile Model**. It is one of the most famous models in physics because it naturally produces a phenomenon called **Self-Organized Criticality (SOC)**, where a system "finds" a critical state all on its own, without any fine-tuning.

The connection to foam is neat: in our model, every cell in a grid holds some "grains" (think of them as local stress in a bubble film). When a cell gets too many grains, it "topples" and passes them to its neighbors. This can trigger a chain reaction, just like how one bubble popping can cause nearby bubbles to pop too.

---

## Key Results (from the paper)

| What we measured | Value |
|---|---|
| Grid size | 50 x 50 |
| Grains dropped | 50,000 |
| Avalanches triggered | 20,106 |
| Biggest avalanche | 8,427 topplings |
| Mean avalanche size | 212.5 topplings |
| Power-law exponent (tau) | 1.41 |
| Total energy dissipated | 17,090,104 |
| Energy rate per grain | 341.8 |

The power-law exponent tau = 1.41 sits right in the theoretical range of 1.2 to 1.5 for the 2D BTW model. This is the signature that tells us the system is genuinely in a self-organized critical state.

---

## How does it work?

Here is the whole idea in plain language:

1. Start with an empty grid (50 rows x 50 columns = 2500 cells).
2. Drop one grain at a random cell.
3. If that cell now has 4 or more grains, it **topples**: it loses 4 grains and its four neighbors (up, down, left, right) each gain 1.
4. If any neighbor now also has 4+ grains, they topple too. This keeps going until everything is stable again.
5. The whole chain of topplings is one **avalanche**.
6. Repeat from step 2, fifty thousand times.

At first, nothing much happens. But after enough grains pile up, the grid reaches a critical state where even a single extra grain can sometimes set off a massive cascade, almost like a system that is always living on the edge.

The boundary is **open**: grains that fall off the edge are lost. This is important because it keeps the total energy from growing forever.

---

## Files in this repository

```
sandpile-project/
|
|-- sandpile_model.py      The core model: grid, toppling logic, avalanche tracking
|-- visualize.py           All plotting functions (saves figures to figures/)
|-- run_example.py         One-click script: run simulation + generate all figures
|-- requirements.txt       Python packages you need
|-- figures/               All generated plots live here
|   |-- 00_summary_dashboard.png
|   |-- 01_lattice_state.png
|   |-- 02_toppling_heatmap.png
|   |-- 03_avalanche_distribution.png
|   |-- 04_energy_dissipation.png
|   |-- 05_temporal_correlation.png
|   |-- 06_size_duration_scaling.png
|-- README.md              This file
```

---

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/sandpile-foam-collapse.git
cd sandpile-foam-collapse
```

### 2. Install dependencies

Only two packages are needed: numpy and matplotlib.

```bash
pip install -r requirements.txt
```

### 3. Run the simulation

```bash
python run_example.py
```

This drops 50,000 grains on a 50x50 grid, prints all the statistics, and saves all six figures to the `figures/` folder.

### 4. Use it in your own code

```python
from sandpile_model import run_simulation
from visualize import generate_all_figures

# Run with your own settings
model, analyzer, summary = run_simulation(
    shape=(100, 100),   # bigger grid
    n_grains=200000,    # more grains
    boundary_condition="open",
    seed=7              # set seed for reproducibility
)

print(summary)

# Save all the plots
generate_all_figures(model, analyzer)
```

---

## The figures explained

### 00 Summary Dashboard
A six-panel overview of everything at once. Good for a quick look at the whole simulation.

### 01 Lattice State
The final state of the grid. Dark red = cells with lots of grains (near the critical threshold). Light yellow = nearly empty cells. The grid should look random but structured, which is the hallmark of SOC.

### 02 Toppling Heatmap (log scale)
Which cells toppled the most during the simulation? For open boundaries, the center topples way more than the edges, because grains keep flowing inward from all directions.

### 03 Avalanche Size Distribution
The most important plot. On a log-log scale, a power law looks like a straight line. Our data follows this beautifully, confirming SOC. The fitted exponent tau = 1.41.

### 04 Energy Dissipation
Cumulative energy released over time. It grows nearly linearly once the system hits its critical state, meaning the energy release rate becomes constant.

### 05 Temporal Correlation
Do large avalanches bunch together in time? The correlation drops to near zero after just one lag, which means each avalanche is essentially independent from the previous one. This is expected for SOC.

### 06 Size-Duration Scaling
Bigger avalanches take longer. The relationship follows D ~ S^0.65, another power law. This is a universal feature of critical systems.

---

## The physics: Self-Organized Criticality

SOC was introduced by Bak, Tang, and Wiesenfeld in 1987. The idea is that many complex systems, from earthquakes to forest fires to stock markets to foams, naturally evolve toward a critical state where events of all sizes happen, following a power-law distribution.

The power law P(s) ~ s^(-tau) means there is no "typical" avalanche size. Most are tiny, but occasionally a massive one sweeps the whole system. You cannot predict when the big one will happen, just that it will, eventually.

In foam terms:
- The grid cells are small patches of bubble film
- Grains represent local stress or pressure
- The critical threshold (4 grains) is the stress a film can hold before it ruptures
- Topplings are bubble bursts
- Avalanches are cascading burst events, like foam suddenly collapsing

The fact that real foams show power-law distributions of burst events suggests foam collapse might genuinely be a self-organized critical phenomenon.

---

## Bug fixes from original code

A couple of small issues were cleaned up from the original submission:

1. **Energy calculation**: The original `_topple` method separately counted "grains lost" for boundary cells and added them to energy, but since those grains were never transferred, this double-counted a bit. Fixed: energy per topple is now simply `critical_threshold * grain_energy`, representing all energy released by one topple regardless of boundary position.

2. **OLS fit edge case**: The `fit_power_law_exponent` with `method="ols"` would crash if all avalanche sizes were identical (only one unique value). Added a guard check.

3. **F-string bug**: The summary title in `plot_avalanche_distribution` had a malformed f-string that printed the conditional expression as text instead of evaluating it. Fixed.

---

## Parameters you can play with

| Parameter | Default | What it does |
|---|---|---|
| `shape` | (50, 50) | Grid dimensions. Bigger = slower but richer statistics |
| `n_grains` | 50000 | More grains = more avalanches = better statistics |
| `critical_threshold` | 4 | How many grains before a cell topples |
| `boundary_condition` | "open" | "open" loses grains at edges; "periodic" wraps around |
| `grain_energy` | 1.0 | Energy per grain transferred (scale factor only) |
| `seed` | None | Set for reproducible results |

---

## References

1. Bak, P., Tang, C., and Wiesenfeld, K. (1987). *Self-organized criticality: An explanation of 1/f noise.* Physical Review Letters, 59(4), 381.
2. Dhar, D. (1990). *Self-organized critical state of sandpile automaton models.* Physical Review Letters, 64(14), 1613.
3. Held, G. A., et al. (1990). *Experimental study of critical-mass fluctuations in an evolving sandpile.* Physical Review Letters, 65(9), 1120.
