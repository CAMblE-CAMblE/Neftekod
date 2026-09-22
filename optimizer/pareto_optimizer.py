from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Пути
ROOT = Path(__file__).resolve().parent.parent
QUALITY_RESULTS_PATH = (ROOT / "optimizer" / "logs" / "quality_results.csv")
ENERGY_RESULTS_PATH = (ROOT / "optimizer" / "logs" / "energy_results.csv")
OUTPUT_DIR = ROOT / "optimizer" / "logs"
PARETO_RESULTS_PATH = (OUTPUT_DIR / "pareto_results.csv")
SELECTED_SCENARIO_PATH = (OUTPUT_DIR / "selected_scenario.csv")
PARETO_PLOT_PATH = (OUTPUT_DIR / "pareto_front.png")


# Pareto front
def calculate_pareto_front(df):
    """
    Быстрый расчёт Pareto-front для двух критериев:

    Минимизируем:
    - predicted_sulfur_mg_kg
    - energy_proxy_eu_h

    Алгоритм:
    1. Оставляем только quality-feasible сценарии.
    2. Сортируем их по energy_proxy_eu_h.
    3. Идём по отсортированным значениям энергии.
    4. Кандидат является Pareto-optimal, если его
       predicted_sulfur_mg_kg меньше минимального значения,
       встреченного ранее.

    Сложность:
    O(N log N) из-за сортировки.

    Внутри основного алгоритма не используются:
    - iterrows()
    - DataFrame.drop()
    - двойной цикл по кандидатам.
    """

    result = df.copy()

    result["pareto_optimal"] = False


    # Только допустимые по качеству сценарии
    feasible = result.loc[
        result["quality_feasible"]
    ].copy()

    if feasible.empty:
        return result

    # Убираем дубликаты по двум оптимизируемым критериям
    #
    # Если два сценария имеют абсолютно одинаковые
    # sulfur + energy, для Pareto они эквивалентны.
    feasible = feasible.drop_duplicates(
        subset=[
            "predicted_sulfur_mg_kg",
            "energy_proxy_eu_h",
        ]
    )

    # Получаем NumPy-массивы

    energy = feasible["energy_proxy_eu_h"].to_numpy(
        dtype=float
    )

    sulfur = feasible["predicted_sulfur_mg_kg"].to_numpy(
        dtype=float
    )

    indices = feasible.index.to_numpy()

    # Сортировка по энергии:
    # сначала самые дешёвые сценарии.
    #
    # mergesort стабильный, что даёт детерминированное
    # поведение при одинаковой энергии.

    order = np.argsort(
        energy,
        kind="mergesort",
    )

    sorted_energy = energy[order]
    sorted_sulfur = sulfur[order]
    sorted_indices = indices[order]

    # Для каждой позиции определяем:
    # является ли текущая сера новым минимумом.
    #
    # cumulative minimum предыдущих значений:
    # [9.8, 9.5, 9.7, 9.2]
    #
    # становится:
    # [9.8, 9.5, 9.5, 9.2]
    #
    # Текущая точка Pareto, если её sulfur
    # строго меньше предыдущего минимума.


    cumulative_min = np.minimum.accumulate(
        sorted_sulfur
    )

    previous_min = np.empty_like(
        cumulative_min
    )

    previous_min[0] = np.inf

    if len(previous_min) > 1:
        previous_min[1:] = cumulative_min[:-1]

    pareto_mask = (
        sorted_sulfur < previous_min
    )

    pareto_indices = sorted_indices[
        pareto_mask
    ]

    # Помечаем Pareto-кандидатов
    result.loc[
        pareto_indices,
        "pareto_optimal"
    ] = True

    return result


# Выбор одного сценария

def select_scenario(df):
    """
    Выбирает один сценарий с Pareto-фронта.

    Используется расстояние до идеальной точки:

    - минимальная predicted_sulfur_mg_kg
    - минимальная energy_proxy_eu_h

    Оба критерия нормализуются в [0, 1].
    """

    pareto = df.loc[
        df["pareto_optimal"]
    ].copy()

    if pareto.empty:
        raise RuntimeError(
            "Pareto front is empty. "
            "No feasible scenario available."
        )

    # Если Pareto состоит из одного сценария
    if len(pareto) == 1:

        selected = pareto.iloc[[0]].copy()

        selected["distance_to_ideal"] = 0.0

        return selected

    # NumPy-массивы
    sulfur = pareto[
        "predicted_sulfur_mg_kg"
    ].to_numpy(dtype=float)

    energy = pareto[
        "energy_proxy_eu_h"
    ].to_numpy(dtype=float)

    # Нормализация серы
    sulfur_min = np.min(sulfur)
    sulfur_max = np.max(sulfur)

    if sulfur_max == sulfur_min:

        sulfur_normalized = np.zeros_like(
            sulfur
        )

    else:

        sulfur_normalized = (
            (sulfur - sulfur_min)
            / (sulfur_max - sulfur_min)
        )

    # Нормализация энергии
    energy_min = np.min(energy)
    energy_max = np.max(energy)

    if energy_max == energy_min:

        energy_normalized = np.zeros_like(
            energy
        )

    else:

        energy_normalized = (
            (energy - energy_min)
            / (energy_max - energy_min)
        )

    # Расстояние до идеальной точки (0, 0)

    distance = np.sqrt(
        sulfur_normalized ** 2
        + energy_normalized ** 2
    )

    pareto["sulfur_normalized"] = (
        sulfur_normalized
    )

    pareto["energy_normalized"] = (
        energy_normalized
    )

    pareto["distance_to_ideal"] = distance

    # Выбираем минимальное расстояние
    selected_position = np.argmin(
        distance
    )

    selected = pareto.iloc[
        [selected_position]
    ].copy()

    return selected


# Визуализация
def plot_pareto_front(df):
    """
    Строит график Pareto-front.

    Отображаются только Pareto-optimal сценарии.
    """

    pareto = df.loc[
        df["pareto_optimal"]
    ].copy()

    if pareto.empty:
        raise RuntimeError(
            "Cannot plot Pareto front: front is empty."
        )

    pareto = pareto.sort_values(
        "energy_proxy_eu_h"
    )

    plt.figure(
        figsize=(10, 6)
    )

    # Pareto-точки
    plt.scatter(
        pareto["energy_proxy_eu_h"],
        pareto["predicted_sulfur_mg_kg"],
        s=70,
        label="Pareto front",
    )

    # Соединяем точки
    if len(pareto) > 1:

        plt.plot(
            pareto["energy_proxy_eu_h"],
            pareto["predicted_sulfur_mg_kg"],
            linestyle="--",
        )

    # Ограничение по сере
    plt.axhline(
        y=10,
        linestyle=":",
        label="Sulfur limit (10 mg/kg)",
    )

    plt.xlabel(
        "Energy proxy, EU/h"
    )

    plt.ylabel(
        "Predicted sulfur, mg/kg"
    )

    plt.title(
        "Pareto front"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        PARETO_PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# Main
def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Загрузка результатов
    quality = pd.read_csv(
        QUALITY_RESULTS_PATH
    )

    energy = pd.read_csv(
        ENERGY_RESULTS_PATH
    )

    # Проверка candidate_id
    if "candidate_id" not in quality.columns:
        raise ValueError(
            "quality_results.csv must contain candidate_id"
        )

    if "candidate_id" not in energy.columns:
        raise ValueError(
            "energy_results.csv must contain candidate_id"
        )

    # Merge
    result = quality.merge(
        energy,
        on="candidate_id",
        how="inner",
        suffixes=("", "_energy"),
    )

    if result.empty:
        raise RuntimeError(
            "No candidates after merging "
            "quality and energy results."
        )

    # Определяем quality_feasible
    if "quality_feasible" not in result.columns:

        if "limit_exceeded" not in result.columns:
            raise ValueError(
                "Quality results must contain "
                "quality_feasible or limit_exceeded."
            )

        result["quality_feasible"] = (
            ~result["limit_exceeded"].astype(bool)
        )

    # Проверяем наличие допустимых сценариев
    feasible_count = int(
        result["quality_feasible"].sum()
    )

    if feasible_count == 0:
        raise RuntimeError(
            "No quality-feasible scenarios available. "
            "Pareto optimization cannot be performed."
        )

    # Проверяем обязательные поля
    required_columns = [
        "candidate_id",
        "predicted_sulfur_mg_kg",
        "energy_proxy_eu_h",
    ]

    missing = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Pareto front
    result = calculate_pareto_front(
        result
    )

    pareto = result.loc[
        result["pareto_optimal"]
    ].copy()

    if pareto.empty:
        raise RuntimeError(
            "Pareto front is empty."
        )

    # Выбор сценария
    selected = select_scenario(
        result
    )

    # Сохраняем результаты
    # В файл попадают только допустимые сценарии.


    result_to_save = result.loc[
        result["quality_feasible"]
    ].copy()

    result_to_save.to_csv(
        PARETO_RESULTS_PATH,
        index=False,
    )

    selected.to_csv(
        SELECTED_SCENARIO_PATH,
        index=False,
    )


    # График

    plot_pareto_front(
        result
    )


    # Console output
    print("PARETO OPTIMIZER")
    print(f"Total candidates:      {len(result)}")
    print(f"Quality-feasible:      {feasible_count}")
    print(f"Pareto-optimal:        {len(pareto)}")
    print()
    print("Selected scenario:")
    print(selected[
            [
                "candidate_id",
                "predicted_sulfur_mg_kg",
                "energy_proxy_eu_h",
                "distance_to_ideal",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"Pareto results:    {PARETO_RESULTS_PATH}")
    print(f"Selected scenario: {SELECTED_SCENARIO_PATH}")
    print(f"Pareto plot:       {PARETO_PLOT_PATH}")

if __name__ == "__main__":
    main()