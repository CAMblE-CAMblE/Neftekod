# Optimizer README

## 1. Назначение

Optimizer - модуль, который получает текущее состояние технологического процесса, генерирует допустимые варианты изменения управляемых параметров, оценивает каждый вариант по качеству и условным энергозатратам, после чего выбирает компромиссный режим.

Текущий pipeline:

```text
Текущее состояние процесса
        ↓
Scenario Generator
        ↓
Набор сценариев
        ↓
Quality Evaluator
        ↓
Прогноз качества
        ↓
Energy Proxy
        ↓
Оценка энергозатрат
        ↓
Pareto Optimizer
        ↓
Pareto front
        ↓
Selected scenario
        ↓
Управляющее воздействие
```

---

# 2. Структура

```text
optimizer/
│
├── README.md
├── scenario_generator.py
├── quality_evaluator.py
├── energy_proxy.py
├── pareto_optimizer.py
├── show_action.py
│
├── examples/
│   ├── base_state.csv
│   └── controllable_bounds.csv
│
├── logs/
│   ├── scenarios.csv
│   ├── quality_results.csv
│   ├── energy_results.csv
│   ├── pareto_results.csv
│   ├── selected_scenario.csv
│   └── pareto_front.png
│
└── docs/
    ├── Scenario_Generator_Handoff.md
    ├── Quality_Evaluator_Handoff.md
    ├── Energy_Proxy_Handoff.md
    └── Pareto_Optimizer_Handoff.md
```

Отдельно:

```text
quality_agent_package/
```

содержит модель и код Quality Agent.

Optimizer использует его через:

```python
from quality_agent.inference import load_bundle, evaluate_candidates
```

`quality_evaluator.py` добавляет `quality_agent_package` в `sys.path` и вызывает Quality Agent напрямую.

---

# 3. Что является входом Optimizer

Основные входные данные:

### `base_state.csv`

Текущее состояние процесса.

Сейчас используется первая строка:

```python
pd.read_csv(base_state_path).iloc[0]
```

Это означает, что в реальной системе этот файл должен формироваться автоматически из актуального состояния процесса.

---

### `controllable_bounds.csv`

Список параметров, которыми разрешено управлять:

```text
parameter,lower_bound,upper_bound
```

Например:

```text
avt_T6,180,360.0
avt_F12,300,450.0
hdt_W4,0,5
hdt_T12,150,200
hdt_F25,10000,20000
```

Именно этот файл определяет:

* какие параметры может менять Optimizer;
* допустимый диапазон каждого параметра.

Исторический min/max сам по себе не должен использоваться как технологический диапазон.

---

# 4. Полный запуск

Логически pipeline запускается в таком порядке:

```text
1. scenario_generator.py
2. quality_evaluator.py
3. energy_proxy.py
4. pareto_optimizer.py
5. show_action.py
```

---

# 5. Шаг 1 - генерация сценариев

Запускается:

```bash
python optimizer/scenario_generator.py
```

Скрипт:

1. читает `base_state.csv`;
2. читает `controllable_bounds.csv`;
3. для каждого управляемого параметра строит значения вокруг текущего режима;
4. комбинирует значения всех параметров;
5. создаёт `candidate_id`;
6. сохраняет:

```text
optimizer/logs/scenarios.csv
```

Сейчас:

```python
N_STEPS = 3
STEP_FRACTION = 0.05
```

То есть для каждого параметра создаётся до 7 значений:

```text
current - 3 step
current - 2 step
current - 1 step
current
current + 1 step
current + 2 step
current + 3 step
```

Размер шага:

```text
step = 5% от полного допустимого диапазона
```

Значения ограничиваются `lower_bound` / `upper_bound`.
При `N` управляемых параметров максимальное количество сценариев:

```text
7^N
```

Например:

```text
3 параметра → 343 сценария
4 параметра → 2401 сценарий
5 параметров → 16807 сценариев
```

Поэтому при масштабировании количество сценариев станет одним из основных ограничений.

---

# 6. Шаг 2 - оценка качества

Запуск:

```bash
python optimizer/quality_evaluator.py
```

Скрипт:

1. загружает текущее состояние;
2. загружает сценарии;
3. получает список управляемых параметров из `controllable_bounds.csv`;
4. передаёт Quality Agent только:

```text
candidate_id
управляемые параметры
```

То есть полный `scenarios.csv` напрямую в модель не передаётся.

Quality Agent рассчитывает прогноз качества для каждого сценария. Затем добавляется:

```text
quality_feasible
```

где:

```python
quality_feasible = ~limit_exceeded
```

То есть нарушение жёсткого ограничения по сере делает сценарий недопустимым.

Результат:

```text
optimizer/logs/quality_results.csv
```

---

# 7. Шаг 3 - Energy Proxy

Запуск:

```bash
python optimizer/energy_proxy.py
```

Energy Proxy не использует реальное потребление энергии.

Это текущая оценка в у.е..

Базовое значение:

```text
100 EU/h
```

Для температуры:

```text
рост температуры → увеличение proxy
```

Для давления:

```text
рост давления → увеличение proxy
```

Для расхода используется нелинейная зависимость:

```text
flow_ratio²
```

Текущие коэффициенты:

```text
BASE_ENERGY = 100 EU/h

TEMPERATURE_COEFFICIENT = 0.03
PRESSURE_COEFFICIENT = 0.02

FLOW_COEFFICIENT = 0.50
FLOW_EXPONENT = 2.0
```

Результат:

```text
optimizer/logs/energy_results.csv
```

Помимо общего proxy там сохраняется вклад каждого управляемого параметра.

---

# 8. Шаг 4 - Pareto Optimizer

Запуск:

```bash
python optimizer/pareto_optimizer.py
```

Он объединяет:

```text
quality_results.csv
        +
energy_results.csv
```

по:

```text
candidate_id
```

## Жёсткое ограничение

В Pareto-анализе рассматриваются quality-feasible сценарии.

Критерии Pareto:

```text
1. predicted_sulfur_mg_kg → минимизировать
2. energy_proxy_eu_h      → минимизировать
```

Сценарий доминируется, если существует другой сценарий, который:

* не хуже по сере;
* не хуже по энергии;
* строго лучше хотя бы по одному критерию.

---


# 9. Как выбирается selected_scenario

После построения Pareto front Optimizer должен выбрать один сценарий.

Сейчас используется метод расстояния до идеальной точки.

Идеальная точка:

```text
минимальная сера
минимальная энергия
```

Оба показателя нормализуются:

```text
0 = лучший результат на текущем Pareto front
1 = худший
```

Затем:

```text
distance =
sqrt(
    sulfur_normalized²
    +
    energy_normalized²
)
```

Выбирается сценарий с минимальным расстоянием.

Таким образом, текущая политика:

```text
качество и энергия имеют одинаковый вес
```

Это пока MVP-политика выбора, а не утверждённая технологическая стратегия.

---

# 10. Результаты Pareto

После запуска появляются:

### `pareto_results.csv`

Результаты Pareto-анализа с флагом:

```text
pareto_optimal
```

### `selected_scenario.csv`

Один выбранный сценарий.

### `pareto_front.png`

График Pareto front.

На графике отображаются только Pareto-сценарии, без остальных кандидатов.

---

# 11. Шаг 5 - получение управляющего воздействия

Запуск:

```bash
python optimizer/show_action.py
```

Скрипт сравнивает:

```text
base_state.csv
        VS
selected_scenario.csv
```

и берёт только параметры из `controllable_bounds.csv`.

Если значение изменилось, выводится:

```text
parameter:
    current → recommended
    Δ
```

Например:

```text
=== OPTIMIZER ACTION ===

hdt_T6: 348.000 → 351.000 (Δ +3.000)
hdt_P13: 31.500 → 32.000 (Δ +0.500)
```

То есть именно здесь результат Optimizer превращается в человеко-читаемое управляющее воздействие.

Также выводится ожидаемый результат:

```text
Sulfur
Energy proxy
```

---



# 15. Что должен делать Reliability Agent

Перед запуском Optimizer Reliability Agent должен проверить:

```text
данные существуют?
        ↓
данные свежие?
        ↓
нет пропусков?
        ↓
нет очевидных аномалий?
        ↓
все необходимые параметры доступны?
```

Если состояние плохое:

```text
Optimizer НЕ запускается
```

Например:

```text
Telemetry stale
→ no optimization

Missing hdt_T6
→ no optimization

Invalid process state
→ no optimization
```

Optimizer не должен самостоятельно "догадываться" о недостающем технологическом состоянии.

---

# 16. Что передавать Optimizer

Вместо файла:

```text
base_state.csv
```

в MAS должен приходить строка аналогичного вида

Optimizer должен использовать актуальное состояние как `base_state`.

---

# 17. Текущий запуск

После исправления пути в `scenario_generator.py` полный локальный запуск:

```bash
python optimizer/scenario_generator.py

python optimizer/quality_evaluator.py

python optimizer/energy_proxy.py

python optimizer/pareto_optimizer.py

python optimizer/show_action.py
```

Итоговая команда пользователя должна выглядеть примерно так:

```text
Optimizer
    ↓
"Увеличить hdt_T6 с 348 до 351
 и hdt_P13 с 31.5 до 32.0"

Ожидаемая сера: 8.42 mg/kg
Energy proxy: 103.2 EU/h
```

Именно этот последний результат уже можно передавать в Supervisor/Decision Agent как предлагаемое управляющее воздействие.
