# Комплект агента качества для оптимизатора

Обученный агент качества `first_quality_current` для
диагностической оценки вариантов режима. Модель работает в режиме `current`:
прогноз относится к текущему состоянию, а не к будущему горизонту.

Сценарная сетка исследует реакцию регрессии на заданные входы. В конфиге сохранено
`inference.allow_scenario_assessment: false`.

## Состав файлов

- `artifacts/quality_agent/first_quality_current/model.cbm` - CatBoost-модель.
- `artifacts/quality_agent/first_quality_current/config.json` - фактически использованный конфиг.
- `artifacts/quality_agent/first_quality_current/feature_names.json` - точный порядок признаков.
- `artifacts/quality_agent/first_quality_current/metadata.json` - run_id, split-периоды, версии зависимостей.
- `artifacts/quality_agent/first_quality_current/metrics.json` - метрики.
- `artifacts/quality_agent/first_quality_current/feature_importance.csv` - важности признаков.
- `src/quality_agent/` и `quality_formulas.py` - инференс, подготовка признаков и формулы ВАК.
- `examples/quality_agent/first_quality_current/sample_states.csv` - маленький пример реальных состояний.
- `examples/quality_agent/first_quality_current/model_input_preview.csv` - фактические входы модели с целью.
- `examples/quality_agent/first_quality_current/feature_descriptions.csv` - описание признаков.
- `examples/quality_agent/first_quality_current/single_prediction.csv` - результат одиночного примера.
- `examples/quality_agent/first_quality_current/grid_predictions.csv` - результат диагностической сетки.
- `scripts/predict_quality.py` - одиночный/пакетный прогноз по подготовленным состояниям.
- `scripts/predict_quality_grid.py` - диагностическая сетка T6/F9/P13.
- `requirements-quality-agent.txt` - минимальные зависимости для запуска агента.


## Установка

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install -r requirements-quality-agent.txt
```

Если окружение проекта уже установлено, достаточно:

```powershell
py -m pip install -r requirements.txt
```

## Один прогноз

```powershell
py scripts/predict_quality.py `
  --model-dir artifacts/quality_agent/first_quality_current `
  --input examples/quality_agent/first_quality_current/sample_states.csv `
  --output examples/quality_agent/first_quality_current/single_prediction.csv
```

Результат появляется в `examples/quality_agent/first_quality_current/single_prediction.csv`.

## Диагностическая сетка

Параметры сценария:

- `--input-sulfur` - входящая лабораторная сера, мг/кг.
- `--t6-values` - T6, температура на входе реактора, °C.
- `--f9-values` - F9, массовый расход сырья, т/ч.
- `--p13-values` - P13, давление на входе реактора, МПа.

Диапазоны примера являются опциональными, условными `sample_states.csv`. Технологические ограничения нужно подключать
по подтвержденному источнику ограничений.

```powershell
py scripts/predict_quality_grid.py `
  --model-dir artifacts/quality_agent/first_quality_current `
  --input examples/quality_agent/first_quality_current/sample_states.csv `
  --output examples/quality_agent/first_quality_current/grid_predictions.csv `
  --state-index 0 `
  --input-sulfur 8325.999975 `
  --t6-values 352,354,356 `
  --f9-values 150,155,160 `
  --p13-values 3.70,3.82,3.95
```

CSV содержит:

`candidate_id, source_state_time, input_sulfur_mg_kg, T6, F9, P13,
predicted_sulfur_mg_kg, status, status_reason, assessment_scope`.

Перед каждым предсказанием зависимые ВАК пересчитываются штатной подготовкой
`prepare_features(...)`, порядок признаков берется из `feature_names.json`.

## API

```python
import pandas as pd

from quality_agent.data import read_table
from quality_agent.inference import load_bundle, predict_frame, predict_diagnostic_grid

bundle = load_bundle("artifacts/quality_agent/first_quality_current")
states = read_table("examples/quality_agent/first_quality_current/sample_states.csv")

single = predict_frame(states.head(1), bundle)

grid = predict_diagnostic_grid(
    bundle,
    states.iloc[0],
    scenario_input_sulfur_mg_kg=8325.999975,
    t6_values=[352, 354, 356],
    f9_values=[150, 155, 160],
    p13_values=[3.70, 3.82, 3.95],
)
```

## Зафиксированные результаты

- CatBoost MAE на test относительно Q21: `1.4406 мг/кг`, примерно `1.44 мг/кг`.
- Проверка test относительно `211` лабораторных анализов выходной серы:
  MAE `1.8150 мг/кг`, примерно `1.81 мг/кг`.
- Эти ошибки описывают историческую проверку. 

## Важные допущения

- Цель: выходящая сера `Q21`, мг/кг.
- `Q20` и `Q21` исключены из `X`.
- Подтвержденные ошибочные значения `Q21=307` исключены из цели, исходное
  измерение сохранено для трассировки.
- Данные разделены по времени: train до `2025-07-08 19:00:00`, validation до
  `2026-01-21 21:30:00`, test с `2026-01-21 21:40:00`.
- Лабораторные результаты учитываются только по времени доступности
  `available_at`.
- Входящая лабораторная сера доступна примерно в половине строк.
- Неизвестные назначения и единицы части технологических тегов отмечены в
  `feature_descriptions.csv` как требующие уточнения.

## Что должен проверить оптимизатор

- Подтвержденные технологические ограничения T6/F9/P13 и остальных связанных
  параметров.
- Достоверность сценарной чувствительности модели при изменении режима, а не
  только историческую ошибку.
- Область применимости: не выходят ли кандидаты за распределение обучающих
  данных и физически допустимые режимы.
- Согласование единиц F9 и P13 по промышленному справочнику тегов.
- Правила обработки пропусков ЛИМС и максимально допустимый возраст входящей
  серы для онлайн-работы.
