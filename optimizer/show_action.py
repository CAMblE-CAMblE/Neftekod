from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

BASE_STATE_PATH = ROOT / "optimizer" / "examples" / "base_state.csv"
SELECTED_PATH = ROOT / "optimizer" / "logs" / "selected_scenario.csv"
BOUNDS_PATH = ROOT / "optimizer" / "examples" / "controllable_bounds.csv"


def main():
    base = pd.read_csv(BASE_STATE_PATH).iloc[0]
    selected = pd.read_csv(SELECTED_PATH).iloc[0]
    bounds = pd.read_csv(BOUNDS_PATH)

    print("\n=== OPTIMIZER ACTION ===\n")

    changes = []

    for parameter in bounds["parameter"]:
        old_value = float(base[parameter])
        new_value = float(selected[parameter])

        if old_value != new_value:
            changes.append(
                {
                    "parameter": parameter,
                    "current": old_value,
                    "recommended": new_value,
                    "delta": new_value - old_value,
                }
            )

    if not changes:
        print("Оптимизатор не предлагает изменять режим.")
        return

    for change in changes:
        print(
            f"{change['parameter']}: "
            f"{change['current']:.3f} → "
            f"{change['recommended']:.3f} "
            f"(Δ {change['delta']:+.3f})"
        )

    print("\n=== EXPECTED RESULT ===")
    print(
        f"Sulfur: "
        f"{selected['predicted_sulfur_mg_kg']:.3f} mg/kg"
    )
    print(
        f"Energy proxy: "
        f"{selected['energy_proxy_eu_h']:.3f} EU/h"
    )


if __name__ == "__main__":
    main()