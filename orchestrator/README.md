# Скелет оркестратора

Он запрашивает оценки трех агентов, проверяет жесткие ограничения кодом
и обращается к локальной LLM через Ollama только для выбора между уже
допустимыми вариантами и формулировки объяснения оператору.

## Структура

```
orchestrator/
  contracts.py          # входы/выходы агентов, финальная рекомендация
  json_io.py            # парсинг JSON от агентов -> contracts, сериализация ответа
  hard_constraints.py   # жесткие проверки — НУЖНО РЕДАЧИТЬ
  llm_client.py         # клиент Ollama (+ FakeLLMClient для тестов без Ollama)
  prompts.py            # системный/пользовательский промпт
  orchestrator.py       # склейка: проверки -> (если есть допустимые) -> LLM -> Recommendation
sample_data/            # примеры JSON, имитирующие реальный вывод агентов в едином файле разные случаи
  normal_cycle.json
  quality_risk_cycle.json
  stale_data_cycle.json
demo.py                 # запуск полного цикла на 3 сценариях
```

## Формат входных данных

Оркестратор принимает **JSON**
Схема одного пакета — в `orchestrator/json_io.py`
(`parse_agent_inputs`), примеры — в `sample_data/`.

```python
from orchestrator.json_io import load_agent_inputs
from orchestrator.orchestrator import Orchestrator
from orchestrator.llm_client import OllamaClient

inputs = load_agent_inputs("sample_data/quality_risk_cycle.json")
rec = Orchestrator(OllamaClient()).run_cycle(inputs)
```

Если агенты будут присылать JSON не одним пакетом, а по отдельности
(`state.json`, `quality.json`, `reliability.json`, `optimization.json`) —
используйте `load_and_combine`, она сама прочитает все 4 файла и соберёт
их в нужную структуру:

```python
from orchestrator.json_io import load_and_combine
from orchestrator.orchestrator import Orchestrator
from orchestrator.llm_client import OllamaClient

inputs = load_and_combine(
    state_path="state.json",
    quality_path="quality.json",
    reliability_path="reliability.json",
    optimization_path="optimization.json",
)
rec = Orchestrator(OllamaClient()).run_cycle(inputs)
```

Менять сам оркестратор не нужно.
## Запуск

```bash
pip install -r requirements.txt    # сейчас пустой
ollama pull llama3.2               # если модель еще не загружена
python demo.py                     # реальный Ollama, http://localhost:11434
python demo.py --model qwen2.5:7b  # другая модель
python demo.py --fake              # без Ollama — проверить логику на заглушке LLM
```