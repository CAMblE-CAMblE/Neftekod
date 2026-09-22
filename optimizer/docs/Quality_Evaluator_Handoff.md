# Quality Evaluator - Handoff

## Назначение

`quality_evaluator.py` оценивает сгенерированные оптимизатором сценарии через Quality Agent.

Сейчас используется только один критерий качества:

**сера в готовом дизельном топливе ≤ 10 мг/кг.**

---

## Структура

```text
project/
├── optimizer/
│   ├── quality_evaluator.py
│   ├── scenario_generator.py
│   ├── examples/
│   │   ├── base_state.csv
│   │   └── controllable_bounds.csv
│   └── logs/
│       ├── scenarios.csv
│       └── quality_results.csv
│
└── quality_agent_package/
    └── quality_agent/
        └── inference.py
```

Quality Agent импортируется как пакет:

```python
sys.path.insert(0, str(ROOT / "quality_agent_package"))

from quality_agent.inference import load_bundle, evaluate_candidates
```

---

## Входные данные

### `base_state.csv`

Одно текущее состояние установки.

Используется Quality Agent как базовое состояние, относительно которого оцениваются изменения параметров.

### `scenarios.csv`

Результат `scenario_generator.py`.

Содержит:

```text
candidate_id
state_time
...
управляемые параметры
...
```

В `quality_evaluator.py` из него выбираются только:

```text
candidate_id
+ параметры из controllable_bounds.csv
```

Это необходимо, поскольку `evaluate_candidates()` воспринимает все переданные колонки как overrides.

### `controllable_bounds.csv`

Определяет параметры, которыми управляет оптимизатор:

```text
parameter,lower_bound,upper_bound
```

`quality_evaluator.py` использует колонку `parameter` для определения списка overrides.

---

## Работа

Для каждого `candidate_id` Quality Agent получает:

```text
base_state
+
измененные управляемые параметры
```

После этого вызывается:

```python
evaluate_candidates(
    bundle=bundle,
    base_state=base_state,
    candidates=candidates,
)
```

Quality Agent автоматически пересчитывает необходимые производные признаки.

---

## Результат

Создается:

```text
optimizer/logs/quality_results.csv
```

В нем сохраняются результаты Quality Agent, включая:

```text
candidate_id
predicted_sulfur_mg_kg
sulfur_limit_mg_kg
limit_exceeded
status
status_reason
...
```

Дополнительно `quality_evaluator.py` создает:

```text
quality_feasible
```

Правило:

```python
quality_feasible = ~limit_exceeded
```

То есть:

```text
predicted_sulfur_mg_kg <= sulfur_limit_mg_kg
→ quality_feasible = True

predicted_sulfur_mg_kg > sulfur_limit_mg_kg
→ quality_feasible = False
```

---

## Проверка текущего запуска

Текущая цепочка успешно отработала:

```text
25 сценариев
25 успешно рассчитаны
25 проходят ограничение по сере
```

Следующий этап должен использовать только:

```python
quality_results[
    quality_results["quality_feasible"]
]
```

для дальнейшей оптимизации.

---

## Важное ограничение

`quality_evaluator.py` **не проверяет технологические диапазоны**.

Они должны быть обеспечены:

1. `scenario_generator.py`;
2. reliability agent;
3. `controllable_bounds.csv`.

Quality Agent отвечает только за расчет прогнозируемого качества.

---

## Следующий этап

После Quality Evaluator необходимо добавить:

```text
quality_results.csv
        ↓
отбор quality_feasible
        ↓
расчет energy/cost proxy
        ↓
Pareto front
        ↓
правило выбора одного сценария
        ↓
финальная рекомендация
```

Quality constraint остается **жестким ограничением**, а не частью компенсируемого экономического score.
