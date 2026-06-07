# Data

В этой папке должны находиться локальные датасеты MovieLens.

Сами CSV-файлы не добавляются в GitHub, потому что датасет `ml-latest` слишком большой.

## Скачать датасеты

MovieLens datasets:

https://grouplens.org/datasets/movielens/

Нужны два датасета:

- `ml-latest-small`
- `ml-latest`

## Ожидаемая структура

```text
data/
├── README.md
├── .gitkeep
├── ml-latest-small/
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   ├── README.txt
│   └── tags.csv
│
└── ml-latest/
    ├── genome-scores.csv
    ├── genome-tags.csv
    ├── links.csv
    ├── movies.csv
    ├── ratings.csv
    ├── README.txt
    └── tags.csv