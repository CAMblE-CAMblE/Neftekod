# Агент надёжности

Оценивает тяжесть режима и риск для оборудования/катализатора. Отдаёт JSON
в формате контракта `reliability` (см. `orchestrator/contracts.py`,
`ReliabilityAssessment`)

## Запуск

```bash
pip install -r requirements.txt
# положить телеметрию в data/ (242000_tags.csv, avt_tags.csv) или симлинк
python run_reliability.py "2024-11-25 22:50:00"
python run_reliability.py            # демо-моменты
```
Первый запуск строит parquet-кэш телеметрии в `.cache/` (дальше быстро).

## Что возвращает (контракт)

```json
{
  "risk_index": 0.35,
  "risk_class": "low|medium|high",
  "risk_factors": ["текст для оператора"],
  "regime_allowed": true,
  "optimization_constraints": {"T11": [350.0, 384.0]},
  "assumptions": ["коридоры из истории, не из паспорта"]
}
```
`regime_allowed=false` → оркестратор отказывает без рассмотрения сценариев.
`optimization_constraints` → коридоры, по которым проверяются сценарии оптимизатора.

## Логика (3 блока)

- **A. `state_detector.py`** — состояние установки: `steady_normal / shutdown /
  startup / transient`. Останов/пуск/переход → `regime_allowed=false`.
  Порог по температуре реактора и расходу (в данных ~2.9–4% точек — останов).
- **B. `severity.py`** — `risk_index` из прокси: близость T реактора к верхнему
  рабочему перцентилю, тренд деградации катализатора (рост WABT по кампании),
  перепад давления реактора, соотношение газ/сырьё, скорость дрейфа.
- **C. `constraints.py`** — коридоры на управляемые теги из робастных перцентилей
  только по steady-периодам; при high risk сужаются к медиане.

## Конфиг

`config/config.yaml` — теги, пороги, веса факторов, окна, список управляемых тегов.
Всё, что выведено из истории (а не паспорта), уходит в `assumptions`.

