# MovieLens Analysis

Анализ датасетов MovieLens на Spark DataFrame API (задание 1) и коллаборативная фильтрация по схожести объектов (задание 2).

Датасеты:

- `ml-latest-small` — небольшой набор
- `ml-latest` — полный набор

Порядок выполнения: **сначала small, потом full**.

## Варианты заданий

### Задание 1. Анализ датасета — Вариант 1

Жанры: **Animation**, **Romance**, **Documentary**.

Один фильм может иметь несколько жанров — для подсчёта используется `explode` по полю `genres`.

Скрипты выводят:

1. таблицу жанров и количества фильмов;
2. топ-10 фильмов с наибольшим числом рейтингов для каждого жанра варианта;
3. топ-10 фильмов с наименьшим числом рейтингов при `rating_count > 10`;
4. топ-10 фильмов с наибольшим средним рейтингом при `rating_count > 10`;
5. топ-10 фильмов с наименьшим средним рейтингом при `rating_count > 10`.

### Задание 2. Коллаборативная фильтрация — Вариант 2

Фильтрация **по схожести объектов** (item-based CF).

Скрипты считают:

1. разбиение `train_init` (80%) / `test` (20%);
2. средний рейтинг в `train_init`;
3. baseline RMSE (все предсказания = среднее);
4. RMSE item-based CF (схожесть на `train_init`, оценка на `test`).

## Структура проекта

```text
film-dataset-analysis/
├── data/
│   ├── ml-latest-small/     # небольшой датасет (не в git)
│   └── ml-latest/           # полный датасет (не в git)
├── reports/
│   ├── ml-latest-small/     # готовые отчёты для small
│   └── ml-latest/           # готовые отчёты для full
├── src/
│   ├── ml_latest_small/
│   │   ├── task1_movielens_spark_dataframe.py      # задание 1, Spark
│   │   ├── task1_movielens_report.py               # задание 1, pandas → markdown
│   │   ├── task2_item_based_cf_spark_dataframe.py  # задание 2, Spark
│   │   └── task2_item_based_cf_pandas.py           # задание 2, pandas → markdown
│   └── ml_latest/
│       ├── task1_movielens_latest_spark_dataframe.py
│       ├── task1_movielens_latest_report.py
│       ├── task2_movielens_latest_spark_dataframe.py
│       └── task2_movielens_latest_item_cf.py
├── requirements.txt
└── README.md
```

## Подготовка окружения

### 1. Клонировать репозиторий и перейти в папку

```bash
cd film-dataset-analysis
```

### 2. Скачать датасеты

Источник: https://grouplens.org/datasets/movielens/

Нужны архивы `ml-latest-small.zip` и `ml-latest.zip`. Распакуйте их в `data/`:

```text
data/ml-latest-small/
├── movies.csv
├── ratings.csv
└── ...

data/ml-latest/
├── movies.csv
├── ratings.csv
└── ...
```

Подробнее — в [data/README.md](data/README.md).

### 3. Установить Python-зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Установить Java (для Spark-скриптов)

Spark требует JDK 8, 11 или 17.

macOS (Homebrew):

```bash
brew install openjdk@17
```

Проверка:

```bash
java -version
```

## Запуск

Все скрипты читают данные из `data/` относительно корня проекта — пути настраивать не нужно.

Рекомендуемый порядок: **small → full**, **задание 1 → задание 2**.

### Задание 1 — Spark (основной вариант для сдачи)

**ml-latest-small:**

```bash
spark-submit src/ml_latest_small/task1_movielens_spark_dataframe.py
```

**ml-latest:**

```bash
spark-submit src/ml_latest/task1_movielens_latest_spark_dataframe.py
```

Результат выводится в консоль.

### Задание 1 — pandas (генерация markdown-отчёта)

```bash
python3 src/ml_latest_small/task1_movielens_report.py
python3 src/ml_latest/task1_movielens_latest_report.py
```

Отчёты сохраняются в:

- `reports/ml-latest-small/task1_movielens_results.md`
- `reports/ml-latest/task1_movielens_latest_results.md`

### Задание 2 — Spark

**ml-latest-small** (~1–5 мин):

```bash
spark-submit src/ml_latest_small/task2_item_based_cf_spark_dataframe.py
```

**ml-latest** (тяжёлый запуск, нужно много RAM):

```bash
spark-submit src/ml_latest/task2_movielens_latest_spark_dataframe.py
```

В консоли появятся `Mean rating in train_init`, `Baseline RMSE on test` и `Item-based CF RMSE on test`.

### Задание 2 — pandas (без Java, с сохранением отчёта)

```bash
python3 src/ml_latest_small/task2_item_based_cf_pandas.py
python3 src/ml_latest/task2_movielens_latest_item_cf.py
```

Отчёты:

- `reports/ml-latest-small/task2_item_based_cf_results.md`
- `reports/ml-latest/task2_movielens_latest_results.md`

### Если `spark-submit` не найден

```bash
python3 -m pip show pyspark
```

Путь к `spark-submit` обычно лежит в `site-packages/pyspark/bin/`. Пример:

```bash
SPARK_SUBMIT=$(python3 -c "import pyspark, os; print(os.path.join(os.path.dirname(pyspark.__file__), 'bin/spark-submit'))")
"$SPARK_SUBMIT" src/ml_latest_small/task1_movielens_spark_dataframe.py
```

## Какой скрипт для чего

| Задача | Датасет | Spark (консоль) | Pandas (markdown) |
|--------|---------|-----------------|-------------------|
| Задание 1 | small | `src/ml_latest_small/task1_movielens_spark_dataframe.py` | `src/ml_latest_small/task1_movielens_report.py` |
| Задание 1 | full | `src/ml_latest/task1_movielens_latest_spark_dataframe.py` | `src/ml_latest/task1_movielens_latest_report.py` |
| Задание 2 | small | `src/ml_latest_small/task2_item_based_cf_spark_dataframe.py` | `src/ml_latest_small/task2_item_based_cf_pandas.py` |
| Задание 2 | full | `src/ml_latest/task2_movielens_latest_spark_dataframe.py` | `src/ml_latest/task2_movielens_latest_item_cf.py` |



## Где настраиваются пути в скриптах

Во всех файлах `src/**/*.py` пути заданы так:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "ml-latest-small"   # или ml-latest
```

Для pandas-скриптов с отчётами:

```python
OUTPUT_PATH = PROJECT_ROOT / "reports" / "ml-latest-small" / "task1_movielens_results.md"
```

Если датасеты лежат в другом месте, измените только строку с `DATA_PATH` в нужном скрипте.

## Ожидаемые результаты (small)

Из уже посчитанных отчётов `reports/ml-latest-small/`:

| Метрика | Значение |
|---------|----------|
| Средний рейтинг в train_init | 3.503229 |
| Baseline RMSE | 1.044674 |
| Item-based CF RMSE | 0.977583 |

Числа фильмов по жанрам (small): Animation — 611, Romance — 1596, Documentary — 440.

## Возможные проблемы

| Проблема | Решение |
|----------|---------|
| `Unable to locate a Java Runtime` | Установите JDK (см. раздел выше) |
| `Path does not exist: data/...` | Скачайте и распакуйте датасеты в `data/` |
| Долгий запуск `ml-latest` task 2 | Это нормально; для отчёта можно использовать pandas-версию |
| Нехватка памяти на full dataset | Увеличьте RAM или запускайте pandas-скрипт задания 2 |
