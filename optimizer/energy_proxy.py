from pathlib import Path
import pandas as pd

# Настройки энергетического proxy
BASE_ENERGY = 100.0  # EU/h

TEMPERATURE_COEFFICIENT = 0.03
PRESSURE_COEFFICIENT = 0.02

FLOW_COEFFICIENT = 0.50
FLOW_EXPONENT = 2.0


# Определение типа параметра
def get_parameter_type(parameter):
    """
    Определяет физический тип параметра по его названию.

    Возвращает:
        temperature
        pressure
        flow
        unknown
    """

    name = parameter.lower()

    temperature_keywords = [
        "_t",
        "temp",
        "temperature",
    ]

    pressure_keywords = [
        "_p",
        "press",
        "pressure",
    ]

    flow_keywords = [
        "_f",
        "flow",
        "feed",
    ]

    if any(keyword in name for keyword in temperature_keywords):
        return "temperature"

    if any(keyword in name for keyword in pressure_keywords):
        return "pressure"

    if any(keyword in name for keyword in flow_keywords):
        return "flow"

    return "unknown"


# Расчет энергетического proxy
def calculate_energy_proxy(base_state, scenario, parameters):
    """
    Рассчитывает условное энергопотребление сценария.

    Единицы:
        EU/h — условные энергетические единицы в час.

    Базовый режим:
        100 EU/h
    """

    energy = BASE_ENERGY

    contributions = {}

    for parameter in parameters:

        if parameter not in base_state:
            raise ValueError(
                f"Parameter '{parameter}' not found in base state"
            )

        base_value = float(base_state[parameter])
        scenario_value = float(scenario[parameter])

        if base_value == 0:
            continue

        parameter_type = get_parameter_type(parameter)

        relative_change = (
            scenario_value - base_value
        ) / abs(base_value)

        contribution = 0.0

        # Температура
        if parameter_type == "temperature":

            # Учитываем только дополнительный нагрев
            if relative_change > 0:
                contribution = (
                    TEMPERATURE_COEFFICIENT
                    * relative_change
                    * BASE_ENERGY
                )

        # Давление
        elif parameter_type == "pressure":

            # Повышение давления требует дополнительных затрат
            if relative_change > 0:
                contribution = (
                    PRESSURE_COEFFICIENT
                    * relative_change
                    * BASE_ENERGY
                )

        # Расход
        elif parameter_type == "flow":

            # Расход влияет нелинейно
            flow_ratio = scenario_value / base_value

            contribution = (
                FLOW_COEFFICIENT
                * (flow_ratio ** FLOW_EXPONENT - 1)
                * BASE_ENERGY
            )

        # Иной параметр
        else:
            # Пока неизвестные параметры не влияют
            # на энергетический proxy.
            contribution = 0.0

        energy += contribution

        contributions[parameter] = contribution

    return energy, contributions


# Обработка всех сценариев
def calculate_scenarios_energy(
    base_state_path,
    scenarios_path,
    bounds_path,
    output_path,
):
    base_state = pd.read_csv(base_state_path).iloc[0].to_dict()
    scenarios = pd.read_csv(scenarios_path)
    bounds = pd.read_csv(bounds_path)

    if "candidate_id" not in scenarios.columns:
        raise ValueError(
            "scenarios.csv must contain 'candidate_id'"
        )

    if "parameter" not in bounds.columns:
        raise ValueError(
            "controllable_bounds.csv must contain 'parameter'"
        )

    parameters = bounds["parameter"].tolist()

    results = []

    for _, scenario in scenarios.iterrows():

        energy, contributions = calculate_energy_proxy(
            base_state=base_state,
            scenario=scenario,
            parameters=parameters,
        )

        row = {
            "candidate_id": scenario["candidate_id"],
            "energy_proxy_eu_h": energy,
        }

        # Добавляем вклад каждого параметра
        for parameter, contribution in contributions.items():
            row[f"energy_{parameter}"] = contribution

        results.append(row)

    result = pd.DataFrame(results)

    result.to_csv(output_path, index=False)

    return result


# Запуск
if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent.parent

    result = calculate_scenarios_energy(
        base_state_path=(
            ROOT / "optimizer" / "examples" / "base_state.csv"
        ),
        scenarios_path=(
            ROOT / "optimizer" / "logs" / "scenarios.csv"
        ),
        bounds_path=(
            ROOT / "optimizer" / "examples" / "controllable_bounds.csv"
        ),
        output_path=(
            ROOT / "optimizer" / "logs" / "energy_results.csv"
        ),
    )

    print(f"Calculated energy for {len(result)} scenarios")

    print(
        f"Energy range: "
        f"{result['energy_proxy_eu_h'].min():.2f} - "
        f"{result['energy_proxy_eu_h'].max():.2f} EU/h"
    )