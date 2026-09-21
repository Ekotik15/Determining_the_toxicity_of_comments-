# Определение токсичности комментариев

Воспроизводимый учебный проект по бинарной классификации англоязычных комментариев
из Civil Comments. Итоговый протокол использует независимый test split, выбирает
гиперпараметры и порог только на validation и сохраняет все артефакты эксперимента.

## Постановка задачи

Для текста комментария требуется предсказать метку:

- `0` — нетоксичный комментарий;
- `1` — токсичный комментарий.

Используется `mteb/toxic_conversations_50k`: 50 000 train- и 50 000 test-примеров.
Метки получены из Jigsaw Unintended Bias in Toxicity Classification; исходная
непрерывная оценка токсичности бинаризована порогом `target >= 0.5`. Положительный
класс составляет около 8%, поэтому основная метрика — F1 токсичного класса, а
дополнительные — macro-F1, PR-AUC, ROC-AUC, precision, recall и accuracy.

## Эксперимент

Сравниваются пять подходов:

1. `DummyClassifier(strategy="prior")`;
2. word TF-IDF + Logistic Regression;
3. word TF-IDF + калиброванный Linear SVM;
4. word+character TF-IDF + тот же калиброванный Linear SVM;
5. fine-tuned `distilbert-base-uncased` со взвешенной cross-entropy.

Исходный train очищается от пустых строк, дубликатов и пересечений с test, после
чего делится на train/validation стратифицированно в пропорции 80/20. Модель и
порог выбираются по F1 токсичного класса на validation. Test используется только
для финальной оценки выбранной конфигурации.

## Быстрый запуск

Требуется Python 3.10–3.12.

```bash
python -m pip install -r requirements-dev.txt
python scripts/run_classical.py --sample-size 2000 --max-features 5000 \
  --artifacts-dir artifacts-smoke
```

Полный классический эксперимент:

```bash
python scripts/run_classical.py
```

Точная команда опубликованного Transformer-эксперимента:

```bash
python -m pip install -r requirements-transformer.txt
python scripts/run_transformer.py \
  --epochs 1 \
  --max-length 128 \
  --train-batch-size 16 \
  --eval-batch-size 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --random-state 42
```

Опубликованный Transformer-запуск выполнен на CPU; точное время не
регистрировалось. Для повторного обучения рекомендуется CUDA GPU. Скрипт
автоматически использует CUDA и mixed precision, если они доступны.

Данные автоматически загружаются с Hugging Face и кэшируются в `data/`.
Источник закреплён на revision
`ab3aee44e1d13d6938102bef4c6af909e6a9799d` ветки `refs/convert/parquet`.
Перед чтением проверяются SHA-256:

- train: `d27d62973dfc05d1931c5cde813a65792c960cae4e1de914b7650f2c8d00376a`;
- test: `0da0742843c8c94406779a18d6a9c575e4a592e7e650fee52593e9cf5c9475cd`.

Закреплённая conversion revision содержит 50 000 строк в каждом исходном split.
Текущая metadata карточки Hugging Face описывает canonical test split из 2 048
строк; это другой вариант публикации набора. Отчёт и результаты относятся только
к зафиксированным выше Parquet-файлам.

## Артефакты

После запуска в `artifacts/` создаются:

- `metrics.csv` — метрики всех моделей на validation и test;
- `dataset_summary.json` — размеры и баланс классов;
- `model_metadata.json` — выбранная модель, порог и seed;
- `transformer_metadata.json` — backbone, гиперпараметры, устройство и порог;
- `toxic_classifier.joblib` — модель для инференса;
- `test_predictions.parquet` — test-предсказания;
- `transformer_test_predictions.parquet` — test-предсказания DistilBERT;
- `transformer_model/` — fine-tuned модель и tokenizer;
- `confusion_matrix.png` и `precision_recall_curves.png` — графики.

После полного эксперимента дополнительный анализ запускается командой:

```bash
python scripts/analyze_predictions.py
python scripts/analyze_predictions.py \
  --predictions artifacts/transformer_test_predictions.parquet \
  --metadata artifacts/transformer_metadata.json
```

Он создаёт `bootstrap_ci.csv`, `threshold_comparison.csv`,
`error_analysis.csv` и `error_analysis_summary.json`. В отчёте эти артефакты
используются для доверительных интервалов, сравнения порогов и анализа ошибок.

Модель и предсказания не добавляются в Git из-за размера, но воспроизводятся одной
командой.

## Приложение

После полного обучения:

```bash
streamlit run streamlit_app.py
```

Streamlit загружает сохранённую классическую SVM и её validation-порог.
Transformer запускается отдельно, поскольку его веса не хранятся в Git.

## Отчёт и ноутбуки

- `report/final_report.pdf` — расширенный готовый отчёт с численным Related Work,
  bootstrap-интервалами, сравнением порогов и анализом ошибок;
- `report/main.tex` и `report/lit.bib` — исходники в структуре шаблона курса;
- `report/report.md` — исходник локальной сборки PDF без LaTeX;
- `notebooks/toxic_comments_classification.ipynb` — чистый учебный ноутбук.

Пересобрать PDF после нового эксперимента:

```bash
python scripts/build_report.py
```

## Проверки

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

Или через Make:

```bash
make test
make lint
make smoke
make transformer
make report
```

## Чек-лист сдачи

1. Загрузить содержимое проекта в публичный или доступный преподавателю GitHub-репозиторий.
2. При необходимости открыть `report/main.tex` в Overleaf и проверить персональные данные.
3. Приложить `report/final_report.pdf`.
4. В поле «Решение» вставить ссылку на GitHub-репозиторий; PDF приложить отдельным
   файлом в форме сдачи.

Отчёт повторяет обязательную структуру шаблона курса: Abstract, Introduction,
Team, Related Work, Dataset, Proposed Approach, Experimental Protocol, Results,
Discussion and Limitations, Conclusion и References.

## Структура

```text
src/toxic_comments/       загрузка данных, метрики и эксперимент
scripts/                  CLI обучения и сборка PDF
tests/                    unit-тесты
notebooks/                чистый учебный ноутбук без сохранённых outputs
report/                   текст и PDF отчёта
streamlit_app.py          демонстрационное приложение
```

## Ограничения

Проект содержит прямое сравнение с DistilBERT на тех же split, но не заявляет
state of the art: опубликованные работы используют другие подвыборки,
многометочную постановку или специальные fairness-метрики, поэтому их числа
нельзя напрямую сравнивать с данным протоколом. Модели обучены на англоязычных
комментариях 2015–2017 годов и могут ошибочно связывать упоминания идентичностей
с токсичностью. Для реального применения необходимы анализ подгрупп, мониторинг
сдвига данных и ручная проверка спорных решений.

## Источники данных

- https://huggingface.co/datasets/mteb/toxic_conversations_50k
- https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification
