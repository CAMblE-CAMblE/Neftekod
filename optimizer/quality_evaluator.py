from pathlib import Path
import sys
import pandas as pd


# Пути проекта
ROOT = Path(__file__).resolve().parent.parent

QUALITY_AGENT_PACKAGE = ROOT / "quality_agent_package"

BASE_STATE_PATH = ROOT / "optimizer" / "examples" / "base_state.csv"
SCENARIOS_PATH = ROOT / "optimizer" / "logs" / "scenarios.csv"
BOUNDS_PATH = ROOT / "optimizer" / "examples" / "controllable_bounds.csv"

MODEL_DIR = QUALITY_AGENT_PACKAGE / "model"

OUTPUT_PATH = ROOT / "optimizer" / "logs" / "quality_results.csv"


# Импорт Quality Agent
sys.path.insert(0, str(QUALITY_AGENT_PACKAGE))
from quality_agent.inference import load_bundle, evaluate_candidates


# Оценка сценариев
def evaluate_scenarios(
    model_dir,
    base_state_path,
    scenarios_path,
    bounds_path,
    output_path,
):
    # Загружаем данные
    base_state = pd.read_csv(base_state_path)
    scenarios = pd.read_csv(scenarios_path)
    bounds = pd.read_csv(bounds_path)

    # Проверяем необходимые колонки
    if "candidate_id" not in scenarios.columns:
        raise ValueError(
            "scenarios.csv must contain 'candidate_id'"
        )

    if "parameter" not in bounds.columns:
        raise ValueError(
            "controllable_bounds.csv must contain 'parameter'"
        )

    # Параметры, которыми управляет оптимизатор
    controllable_parameters = bounds["parameter"].tolist()

    # Проверяем, что они есть в scenarios.csv
    missing = [
        parameter
        for parameter in controllable_parameters
        if parameter not in scenarios.columns
    ]

    if missing:
        raise ValueError(
            f"Scenarios are missing controllable parameters: {missing}"
        )

    # Quality Agent должен получить только:
    # candidate_id + изменяемые параметры
    candidates = scenarios[
        ["candidate_id"] + controllable_parameters
    ].copy()

    # Загружаем модель
    bundle = load_bundle(model_dir)

    # Получаем прогноз качества
    result = evaluate_candidates(
        bundle=bundle,
        base_state=base_state,
        candidates=candidates,
    )

    # Жесткое ограничение по сере
    result["quality_feasible"] = ~result["limit_exceeded"]

    # Сохраняем результаты
    result.to_csv(output_path, index=False)

    return result


# Запуск
if __name__ == "__main__":

    result = evaluate_scenarios(
        model_dir=MODEL_DIR,
        base_state_path=BASE_STATE_PATH,
        scenarios_path=SCENARIOS_PATH,
        bounds_path=BOUNDS_PATH,
        output_path=OUTPUT_PATH,
    )

    print(f"Evaluated scenarios: {len(result)}")

    print(
        f"Quality-feasible: "
        f"{result['quality_feasible'].sum()}/{len(result)}"
    )

    print(f"Results saved to: {OUTPUT_PATH}")