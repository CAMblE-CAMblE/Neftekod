from pathlib import Path
import itertools
import pandas as pd


N_STEPS = 3
STEP_FRACTION = 0.05


def generate_scenarios(
    base_state_path: str | Path,
    bounds_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:

    base_state = pd.read_csv(base_state_path).iloc[0].to_dict()
    bounds = pd.read_csv(bounds_path)

    parameters = []
    parameter_values = []

    for _, row in bounds.iterrows():
        parameter = row["parameter"]
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])

        if parameter not in base_state:
            raise ValueError(
                f"Parameter '{parameter}' not found in base_state.csv"
            )

        current = float(base_state[parameter])

        if not lower <= current <= upper:
            raise ValueError(
                f"{parameter}: current value {current} "
                f"is outside [{lower}, {upper}]"
            )

        # Шаг = 10% полного допустимого диапазона
        step = (upper - lower) * STEP_FRACTION

        if step == 0:
            values = [current]
        else:
            values = [
                current + i * step
                for i in range(-N_STEPS, N_STEPS + 1)
            ]

            # Не выходим за допустимые границы
            values = [
                max(lower, min(upper, value))
                for value in values
            ]

            # Убираем дубликаты после ограничения границами
            values = sorted(set(values))

        parameters.append(parameter)
        parameter_values.append(values)

    scenarios = []

    for candidate_id, combination in enumerate(
        itertools.product(*parameter_values)
    ):
        scenario = dict(base_state)

        for parameter, value in zip(parameters, combination):
            scenario[parameter] = value

        scenario["candidate_id"] = candidate_id
        scenarios.append(scenario)

    result = pd.DataFrame(scenarios)

    # candidate_id первым
    columns = ["candidate_id"] + [
        c for c in result.columns if c != "candidate_id"
    ]
    result = result[columns]

    if output_path is not None:
        result.to_csv(output_path, index=False)

    return result


if __name__ == "__main__":
    ROOT = Path.cwd()

    scenarios = generate_scenarios(
        base_state_path=ROOT / "examples/base_state.csv",
        bounds_path=ROOT / "examples/controllable_bounds.csv",
        output_path=ROOT / "logs/scenarios.csv",
    )

    print(f"Generated {len(scenarios)} scenarios")