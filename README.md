# Neftekod — мультиагентная система управления гидроочисткой

Оркестратор + три агента (качество, надёжность, оптимизация) + LLM-объяснение.
Два способа запуска — см. ниже: старый демо (JSON-заглушки) и новый рабочий
пайплайн (реальные агенты).

## Структура репозитория

```
orchestrator/           оркестратор: контракты, жёсткие проверки, LLM-клиент, промпты
quality/                агент качества: формулы ВАК, парсеры ЛИМС/ПАК, интерфейс (app/)
quality_agent_package/  упакованный агент качества (модель CatBoost + инференс)
reliability/            агент надёжности
optimizer/              генератор сценариев, оценка качества/энергии, Pareto-отбор
sample_data/            примеры JSON для старого demo.py
demo.py                 старый способ запуска (см. "Демо на заглушках" ниже)
run_pipeline.py         новый способ запуска — реальный сквозной пайплайн
requirements.txt        общие зависимости для всего репозитория
```

## Что нужно установить

1. **Python 3.10+** — используется современный синтаксис типов.
2. **Ollama** (https://ollama.com) — LLM работает локально:
   ```
   ollama pull llama3.2
   ```
3. Зависимости:
   ```bash
   python3 -m venv venv
   # Windows: venv\Scripts\Activate.ps1
   # Linux/macOS: source venv/bin/activate
   pip install -r requirements.txt
   ```
4. Для `run_pipeline.py` дополнительно нужна папка `data/` в корне
   (`242000_tags.csv`, `avt_tags.csv` — не в git из-за размера) 
   подготовленный датасет `data/processed/quality_dataset.parquet`
   и файл `quality_agent_package/model/model.cbm` (веса CatBoost).

## Демо на заглушках (`demo.py`)

Проверяет только оркестратор — читает готовые JSON из `sample_data/*.json`
(сценарии со сценариями от "агентов", ничего реально не вызывается).
Полезно для быстрой проверки, что жёсткие проверки/LLM-часть не сломаны.

```bash
python demo.py --fake   # без Ollama, проверка логики
python demo.py          # с реальной Ollama
```

## Рабочий пайплайн (`run_pipeline.py`)

Реальный сквозной цикл: агент надёжности → сетка сценариев → агент качества
(CatBoost) → энергозатраты → агент оптимизации → LLM формулирует объяснение.

```bash
python run_pipeline.py
```

По ходу печатает прогресс `[1/6]`...`[6/6]`. На последнем шаге (LLM) может
занимать несколько минут на CPU — таймаут в `orchestrator/llm_client.py`
сейчас 300 секунд.

## UI для пайплайна

Streamlit-экран берет исторические состояния из `data.prepared_path` в
`quality/configs/quality_agent.yaml`. По умолчанию это
`data/processed/quality_dataset.parquet`; при необходимости путь можно
переопределить переменной окружения `NEFTEKOD_QUALITY_DATASET`.

```bash
streamlit run quality/app/simulator_app.py
```

# Запуск в Docker 

## Вручную 

1. Необходимо подготовить директорию /opt/data: поместить в нее файлы `242000_tags.csv`, `avt_tags.csv`, `data/processed/quality_dataset.parquet`

2. Выполнить загрузку образа ollama и web-llm

```
docker pull ollama/ollama
docker pull vanchello/deep_thinkers:latest
```

3. Выполнить запуск контейнера ollama

```
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama --network host ollama/ollama
docker exec -it ollama ollama pull llama3
```

4. Выполнить запуск контейнера llm-web

```
docker run -d -v config-quality:/pipeline/quality_agent_package/config -v config-reliability:/pipeline/reliability/config \
-p 8501:8501 --name llm-web --network host vanchello/deep_thinkers
```

5. Открыть веб-интерфейс по адресу http://localhost:8501

## С помощью docker-compose 

1. Проверить что на ВМ установлен docker-compose-v2

2. Подготовить директорию /opt/data: поместить в нее файлы `242000_tags.csv`, `avt_tags.csv`, `data/processed/quality_dataset.parquet`

3. Выполнить в директории с файлом docker-compose.yml

```
docker compose up -d 
```

4. Дождаться старта всех контейнеров и окончания работы ollama-pull-init

5. Открыть веб-интерфейс по адресу http://localhost:8501