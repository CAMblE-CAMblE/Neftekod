# Автономный комплект агента качества

Комплект считает серу на выходе гидроочистки по текущей модели `first_quality_current`.
Модель работает в режиме `current`: прогноз относится к тому же времени, что и `state_time`.
Единица целевой серы и прогноза: `мг/кг`. Единицы исходных сигналов нужно сохранять такими же, как в обучающих выгрузках.

## Установка

Из корня `quality_agent_package`:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

## Быстрый запуск 

Готовое состояние уже лежит в `data/base_state.csv`.

```powershell
py -m quality_agent.cli predict `
  --base-state data/base_state.csv `
  --output outputs/prediction.csv
```

Результат: `outputs/prediction.csv`.

## Оценка candidates.csv

`candidates.csv` содержит `candidate_id` и любые изменяемые исходные колонки. Значения являются новыми абсолютными значениями, не приращениями.

Пример:

```csv
candidate_id,hdt_T6,hdt_F9,avt_F45
c01,360.0,230.0,15.5
c02,362.0,235.0,17.0
```

Запуск:

```powershell
py -m quality_agent.cli evaluate-candidates `
  --base-state data/base_state.csv `
  --candidates candidates.csv `
  --output outputs/candidate_predictions.csv
```

ВАК гидроочистки и АВТ пересчитываются автоматически. Технологические диапазоны и собственный список разрешенных параметров проверяет внешний оптимизатор. В пакете нет жесткого allow-list 

Схема проверяется: неизвестные колонки, цели, служебные поля, расчетные ВАК, пустые переопределения и дубли `candidate_id` отклоняются.

## Сборка нового base_state.csv

Скопируй `config/paths.example.yaml` и пропиши свои пути. Относительные пути разрешаются относительно YAML-файла, абсолютные Windows-пути тоже поддерживаются.

Обязательные источники для сборки из сырья:

* `242000_tags.csv`;
* `avt_tags.csv`;
* `ЛИМСы 01.01.2023 - н.в_ (2).xlsx`.

`pak_path` по умолчанию `null`: для текущего инференса серы ПАК не нужен. Исходный Excel формул нужен только для сверки, при выполнении используются формулы в коде.

```powershell
py -m quality_agent.cli build-base-state `
  --paths-config config/paths.my.yaml `
  --state-time "2025-07-16 06:30:00" `
  --output outputs/base_state.csv `
  --report outputs/base_state_report.json
```

Если точного времени нет, команда завершится понятной ошибкой. Явно разрешить ближайшее более раннее время можно так:

```powershell
py -m quality_agent.cli build-base-state `
  --paths-config config/paths.my.yaml `
  --state-time "2025-07-16 06:31:00" `
  --allow-previous `
  --output outputs/base_state.csv `
  --report outputs/base_state_report.json
```

Для повторной сборки из готового parquet задай в YAML `prepared_path` и оставь пути к сырью как есть или пустыми. В этом режиме большие CSV/XLSX не читаются.

`base_state_report.json` содержит запрошенное и фактически выбранное время, режим выбора, отсутствующие входы и возраст лабораторных анализов.

## Python API

Модель можно загрузить один раз и переиспользовать:

```python
from pathlib import Path
import pandas as pd

from quality_agent import evaluate_candidates, load_bundle
from quality_agent.data import read_table

root = Path("quality_agent_package")
bundle = load_bundle(root / "model")
base_state = read_table(root / "data" / "base_state.csv")

candidates = pd.DataFrame(
    {"candidate_id": ["a", "b"], "hdt_T6": [360.0, 362.0]},
    index=[10, 20],
)
result = evaluate_candidates(bundle, base_state, candidates)
```

`candidate_id` возвращается прямо в результате и сохраняет порядок входных кандидатов; соединять по индексу pandas не нужно.

## Поля результатов

`prediction.csv`:

* `state_time`, `prediction_time`;
* `predicted_sulfur_mg_kg`;
* `sulfur_limit_mg_kg`, `limit_exceeded`;
* `missing_or_stale_inputs`;
* `model_version`;
* `interval_status`, `violation_probability_status`.

`candidate_predictions.csv` дополнительно содержит `candidate_id`, переданные колонки переопределения, `source_state_time`, `status`, `status_reason`, `assessment_scope`.

`status=ok` означает успешный расчет. Интервалы и вероятность нарушения не оцениваются: это исследовательская сценарная оценка реакции регрессии, а не производственная рекомендация.

## Текущие метрики

Модель: CatBoost `first_quality_current`, режим `current`.

* Test MAE: `1.5454 мг/кг`;
* Test RMSE: `2.3325 мг/кг`;
* Проверка test против выходного ЛИМС: `211` анализов, MAE `1.8460 мг/кг`, RMSE `2.8681 мг/кг`;
* Median baseline test MAE: `3.2364 мг/кг`.

Полные артефакты: `model/metrics.json`, `model/feature_importance.csv`, `model/metadata.json`.
