"""
run_example.py
==============
A simple script you can run to see the sandpile model in action.

Just run:  python run_example.py

It will simulate, print results, and save all the figures to the figures/ folder.
"""

import matplotlib
matplotlib.use("Agg")  # No interactive window needed; just save files

from sandpile_model import run_simulation
from visualize import generate_all_figures


def main():
    print("=" * 60)
    print("  Sandpile Model for Foam Collapse (BTW Model)")
    print("  Course: IDC621 | Module 1 | Avneesh Singh MS23249")
    print("=" * 60)

    # Standard run matching the report
    model, analyzer, summary = run_simulation(
        shape=(50, 50),
        n_grains=50000,
        critical_threshold=4,
        boundary_condition="open",
        grain_energy=1.0,
        seed=42,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("  Generating figures...")
    print("=" * 60)
    generate_all_figures(model, analyzer)

    print("\n" + "=" * 60)
    print("  DONE!")
    print("  All figures saved to the figures/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
