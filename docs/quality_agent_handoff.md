# Комплект агента качества АВТ + гидроочистка

Актуальный запуск модели: `artifacts/quality_agent/first_quality_current`.

Модель работает в режиме `current`: прогноз относится к текущему состоянию установки. Отдельная задержка процесса в модели не моделируется. Оценка измененных режимов имеет исследовательский статус: это реакция регрессии на подготовленный вход, а не доказанный физический эффект и не производственная рекомендация.

## Что передать

Минимальный комплект для обычного инференса:

- `artifacts/quality_agent/first_quality_current/model.cbm`
- `artifacts/quality_agent/first_quality_current/config.json`
- `artifacts/quality_agent/first_quality_current/feature_names.json`
- `artifacts/quality_agent/first_quality_current/metrics.json`
- `artifacts/quality_agent/first_quality_current/feature_importance.csv`
- `artifacts/quality_agent/first_quality_current/metadata.json`
- `artifacts/quality_agent/first_quality_current/baseline.json`
- `examples/quality_agent/first_quality_current/base_state.csv`
- `src/quality_agent/`
- `src/quality_agent/orchestrator_adapter.py`, если нужен ответ в контракте оркестратора
- `quality_formulas.py`
- `avt6_formulas.py`
- `requirements-quality-agent.txt`
- `docs/quality_agent_handoff.md`

Для пересборки обучающего датасета дополнительно нужны:

- `scripts/build_quality_dataset.py`
- `configs/quality_agent.yaml`
- `lims_parser.py`
- `pak_parser.py`
- исходные файлы из `configs/quality_agent.yaml`:
  - `E:/ITMO/Хакатон_Нефтекод/Нефтекод_2.0/data/242000_tags.csv`
  - `E:/ITMO/Хакатон_Нефтекод/Нефтекод_2.0/data/avt_tags.csv`
  - `E:/ITMO/Хакатон_Нефтекод/Нефтекод_2.0/Выгрузка ПАК 01.01.2023 - н.в_.xlsx`
  - `E:/ITMO/Хакатон_Нефтекод/Нефтекод_2.0/ЛИМСы 01.01.2023 - н.в_ (2).xlsx`

Команда текущей сборки датасета:

```powershell
py scripts/build_quality_dataset.py `
  --config configs/quality_agent.yaml `
  --output data/processed/quality_dataset.parquet `
  --output-lims data/processed/quality_output_lims.parquet `
  --report data/processed/quality_dataset_report.json
```

Для обычного пакетного прогноза полные исходные датасеты не нужны: достаточно модели, кода и входной таблицы состояний.

## Установка

```powershell
py -m pip install -r requirements-quality-agent.txt
```

Если используется общий проектный environment, зависимости качества также входят в `requirements.txt`.

## Базовое состояние

Файл `examples/quality_agent/first_quality_current/base_state.csv` содержит одну реальную строку validation-периода: `2025-07-16 06:30:00`.

Строка выбрана по полноте и качеству данных: есть сигналы гидроочистки и АВТ, `avt_match_status=exact`, есть входящая сера, D15 и 95%.T с возрастом `20.5` часа, stale-флаги ложные, подтвержденного rejected-измерения цели нет. Участок выбран как относительно спокойный по локальной изменчивости основных сигналов, без подбора под прогноз или чувствительность.

Целевые поля, выходящая сера ПАК/ЛИМС и заранее сохраненный прогноз в `base_state.csv` не входят. Для сравнения по исходному датасету в этот момент: измеренная `target_sulfur_mg_kg = 9.014893`, базовый прогноз текущей модели `8.575980 мг/кг`.

## Пакетный инференс

Имена сигналов используют префиксы `hdt_` для гидроочистки и `avt_` для АВТ. Расчетные показатели ВАК напрямую менять не нужно: они пересчитываются внутри `prepare_features(...)`, которую вызывает штатный `predict_frame(...)`.

Список разрешенных регулируемых параметров и технологические диапазоны задает оптимизатор. В модели нет отдельного технологического allow-list: она принимает признаки из `feature_names.json`. Для интеграционного примера ниже меняется `hdt_T6`; остальные исходные измерения копируются из базовой строки.

```python
from pathlib import Path
import sys

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from quality_agent.data import read_table
from quality_agent.features import prepare_features
from quality_agent.inference import load_bundle, predict_frame

MODEL_DIR = ROOT / "artifacts/quality_agent/first_quality_current"
BASE_STATE = ROOT / "examples/quality_agent/first_quality_current/base_state.csv"

bundle = load_bundle(MODEL_DIR)
base = read_table(BASE_STATE).iloc[0].to_dict()

candidates = []
for candidate_id, hdt_t6 in enumerate([base["hdt_T6"], base["hdt_T6"] + 2.0]):
    row = dict(base)
    row["candidate_id"] = candidate_id
    row["hdt_T6"] = hdt_t6
    candidates.append(row)

candidate_frame = pd.DataFrame(candidates)

# Необязательная проверка интеграции: значение дошло до X, ВАК пересчитались.
X, _, _ = prepare_features(candidate_frame, bundle.config, feature_names=bundle.feature_names)
assert X.loc[1, "hdt_T6"] == candidate_frame.loc[1, "hdt_T6"]
assert X.loc[0, "hdt_T50"] != X.loc[1, "hdt_T50"]

prediction = predict_frame(candidate_frame, bundle)
result = candidate_frame[["candidate_id"]].join(prediction)
print(result[["candidate_id", "state_time", "predicted_sulfur_mg_kg", "limit_exceeded"]])
```

Фактический ответ `predict_frame(...)`:

- `state_time`
- `prediction_time`
- `predicted_sulfur_mg_kg` в `мг/кг`
- `sulfur_limit_mg_kg`
- `limit_exceeded`
- `missing_or_stale_inputs`
- `model_version`
- `assumptions`
- `interval_status`
- `violation_probability_status`

Связь результата с сеткой оптимизатора делается по позиции строк или через внешний `candidate_id`, как в примере выше.

## Метрики текущей модели

Целевая величина: сера на выходе гидроочистки, `мг/кг`.

- CatBoost test MAE: `1.5454 мг/кг`
- CatBoost test RMSE: `2.3325 мг/кг`
- Проверка test против выходного ЛИМС: `211` анализов, MAE `1.8460 мг/кг`, RMSE `2.8681 мг/кг`
- Baseline median test MAE: `3.2364 мг/кг`

## Ограничения

- Модель и `config.json` должны загружаться из одного каталога запуска.
- Порядок признаков берется из `feature_names.json`.
- Входящая сера `% масс.` при сборке датасета переводится в `мг/кг`.
- ЛИМС присоединяется backward join по времени доступности; при отсутствии `available_at` используется допущение `sample_time + 4 часа`.
- Максимальный возраст входящего анализа сейчас `240` часов; настройка помечена как экспериментальная.
- Подтвержденное ошибочное значение `hdt_Q21 = 307` исключается из целевой переменной при подготовке данных.
