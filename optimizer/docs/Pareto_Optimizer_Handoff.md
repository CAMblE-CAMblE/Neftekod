# Pareto Optimizer - Handoff

## Назначение

`pareto_optimizer.py` получает результаты Quality Agent и Energy Proxy, отбрасывает сценарии, нарушающие ограничения по качеству, строит Pareto-front и выбирает один компромиссный сценарий для дальнейшей рекомендации.

---

## Входы

### 1. `optimizer/logs/quality_results.csv`

Результат Quality Agent.

Используемые поля:

* `candidate_id`
* `predicted_sulfur_mg_kg`
* `quality_feasible`
* `limit_exceeded`

`quality_feasible = True` означает, что сценарий прошёл жёсткое ограничение по качеству.

### 2. `optimizer/logs/energy_results.csv`

Результат Energy Proxy.

Используемые поля:

* `candidate_id`
* `energy_proxy_eu_h`

Energy Proxy — условная оценка энергозатрат в `EU/h`, а не реальные измеренные энергозатраты.

---

## Логика

### Шаг 1. Объединение

Quality и Energy результаты объединяются по:

```text
candidate_id
```

### Шаг 2. Фильтрация

В Pareto-оптимизацию допускаются **только quality-feasible сценарии**.

Сценарии с нарушением жёсткого ограничения качества исключаются полностью.

Таким образом:

```text
quality violation → scenario rejected
```

Экономия энергии не может компенсировать нарушение качества.

### Шаг 3. Pareto front

Оптимизируются два критерия, оба минимизируются:

1. `predicted_sulfur_mg_kg`
2. `energy_proxy_eu_h`

Сценарий считается доминируемым, если существует другой сценарий, который:

* имеет не больше серы;
* имеет не больше энергозатрат;
* и строго лучше хотя бы по одному из двух критериев.

Недоминируемые сценарии формируют Pareto front.

---

## Выбор `selected_scenario`

После построения Pareto front выбирается один сценарий.

Для каждого Pareto-сценария:

1. Сера нормализуется относительно Pareto front.
2. Энергия нормализуется относительно Pareto front.
3. Рассчитывается расстояние до идеальной точки `(0, 0)`:

```text
distance =
    sqrt(
        sulfur_normalized²
        + energy_normalized²
    )
```

Выбирается сценарий с минимальным расстоянием.

Таким образом, текущий выбор — это **равновесный компромисс между качеством и энергией**.

Важно: сейчас сера и энергия имеют одинаковый вес при выборе `selected_scenario`.

---

## Выходы

### `optimizer/logs/pareto_results.csv`

Содержит **только допустимые по качеству сценарии**.

Основные поля:

```text
candidate_id
predicted_sulfur_mg_kg
energy_proxy_eu_h
quality_feasible
pareto_optimal
```

`pareto_optimal=True` — сценарий находится на Pareto front.

### `optimizer/logs/selected_scenario.csv`

Содержит выбранный компромиссный сценарий:

```text
candidate_id
predicted_sulfur_mg_kg
energy_proxy_eu_h
distance_to_ideal
...
```

### `optimizer/logs/pareto_front.png`

Визуализация содержит **только Pareto front**:

* X → `energy_proxy_eu_h`
* Y → `predicted_sulfur_mg_kg`
* точки Pareto-сценариев;
* линия Pareto front;
* горизонтальная линия жёсткого ограничения `10 mg/kg`.

Остальные кандидаты на графике не отображаются.

---

## Текущая архитектура

```text
scenario_generator
        ↓
    scenarios
        ↓
  Quality Agent ──────→ quality_results.csv
        ↓
   Energy Proxy ──────→ energy_results.csv
        ↓
   Pareto Optimizer
        ↓
 ┌───────────────┐
 │ quality filter│
 └───────┬───────┘
         ↓
   Pareto front
         ↓
 selected_scenario
```

---

## Важные ограничения

* Quality constraint является жёстким.
* Недопустимые сценарии не участвуют в Pareto-анализе.
* Energy Proxy — MVP-оценка, а не фактическое потребление энергии.
* Pareto front не определяет единственно правильный режим — он показывает набор недоминируемых компромиссов.
* Текущий `selected_scenario` использует равный вклад качества и энергии после нормализации.
* Технологические диапазоны должны быть проверены **до Pareto-оптимизации** отдельным механизмом optimizer.

## Следующий возможный этап

После Pareto можно добавить более технологически осмысленную политику выбора:

```text
1. Hard constraints
        ↓
2. Quality feasibility
        ↓
3. Pareto front
        ↓
4. Приоритет качества / допустимый запас
        ↓
5. Минимизация энергии
        ↓
6. selected_scenario
```

Это позволит явно реализовать принцип ТЗ: **качество и безопасность выше экономической эффективности**.
