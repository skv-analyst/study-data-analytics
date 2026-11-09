# 4. `SeasonalMovingAverageModel` — сезонное скользящее среднее

<!--
Источник — ../notebooks/04-seasonal-moving-average-model.ipynb (эксперименты и сохранённые результаты).
Структура: теория → практика с вопросами и раскрывающимися ответами → итоги.
Иллюстрации извлечены из сохранённых выводов исходного ноутбука.
-->

`SeasonalMovingAverageModel` строит прогноз как среднее нескольких значений ряда, взятых не подряд, а с
постоянным шагом `seasonality` назад во времени. `window` определяет, сколько таких значений усреднять,
а `seasonality` — на каком расстоянии друг от друга они находятся.

Например, для дневных данных и `seasonality=7` модель сопоставляет понедельник с предыдущими понедельниками,
вторник — с предыдущими вторниками и так далее. В этой главе разберём, почему важно попасть в реальный период
ряда, как `seasonality` связана с горизонтом прогноза и как подобрать оба параметра на двух реальных юнитах.

Исполняемый код, полные графики и сохранённые результаты экспериментов — в
[исходном ноутбуке](../notebooks/04-seasonal-moving-average-model.ipynb).

## 1. Теория

### 1.1 Идея сезонного скользящего среднего

Прогноз — среднее арифметическое `window` значений ряда, взятых с шагом `seasonality`:

\[
\hat{y}_{T+h} = \frac{1}{w}\sum_{i=1}^{w} z_{T+h-i\cdot s},
\]

где:

- \(T\) — последняя дата обучающей выборки;
- \(h\) — шаг прогноза;
- \(w\) — размер окна (`window`), то есть сколько значений усредняем;
- \(s\) — сезонность (`seasonality`), то есть расстояние между используемыми точками;
- \(z_t\) — фактическое значение, если дата находится внутри истории, или ранее рассчитанный прогноз,
  если дата уже лежит за границей обучающей выборки.

Модель усредняет не соседние точки, а значения одной и той же фазы цикла. Например, при `seasonality=7` и
`window=3` прогноз на понедельник строится по трём предыдущим понедельникам. Для дневных данных это позволяет
учитывать недельный ритм, не смешивая понедельники с пятницами или выходными.

### 1.2 Параметры и реализация

В исходнике `etna/models/seasonal_ma.py` модель устроена следующим образом:

```python
class SeasonalMovingAverageModel(NonPredictionIntervalContextRequiredAbstractModel):
    def __init__(self, window: int = 5, seasonality: int = 7):
        self.window = window
        self.seasonality = seasonality

    @property
    def context_size(self) -> int:
        return self.window * self.seasonality
```

**Параметры и свойства `SeasonalMovingAverageModel`:**

| Параметр / свойство | Значение |
|---|---|
| `window` | Сколько значений усреднять; целое число, по умолчанию `5`. |
| `seasonality` | Шаг между значениями, используемыми для прогноза; целое число, по умолчанию `7`. |
| `context_size` | `window * seasonality` — столько исторических точек нужно передать модели как контекст. |

`window` и `seasonality` здесь можно настраивать независимо. Это объединяет две предыдущие модели:

- `NaiveModel(lag=L)` — частный случай `SeasonalMovingAverageModel(window=1, seasonality=L)`;
- `MovingAverageModel(window=W)` — частный случай `SeasonalMovingAverageModel(window=W, seasonality=1)`.

У `NaiveModel` размер окна был жёстко равен `1`, а `lag` задавал расстояние до предыдущего значения. У
`MovingAverageModel` расстояние было жёстко равно `1`, а `window` задавал число соседних значений. В сезонной
модели оба параметра свободны: можно усреднять несколько прошлых циклов и при этом брать точки с любым
постоянным шагом.

Типичный порядок обучения и прогноза:

```python
from etna.models import SeasonalMovingAverageModel

HORIZON = 14

model = SeasonalMovingAverageModel(window=3, seasonality=7)
model.fit(train_ts)

future = train_ts.make_future(
    future_steps=HORIZON,
    tail_steps=model.context_size,
)

forecast_ts = model.forecast(
    ts=future,
    prediction_size=HORIZON,
)
```

`tail_steps=model.context_size` добавляет к будущему датасету необходимую историю. Например, при
`window=3` и `seasonality=7` модели нужны `21` точка: три значения, разделённые недельным шагом.

### 1.3 Выбор `seasonality` и рекурсия

**`seasonality` нужно соотносить с реальным периодом данных.** Для дневного ряда с недельным ритмом естественный
кандидат — `7`: понедельник сравнивается с понедельниками, вторник — со вторниками. Если поставить `3`, модель
будет систематически брать другую фазу недели: например, для понедельника — значение пятницы. Это не просто
добавляет случайный шум, а создаёт систематическое смещение прогноза.

**Кратные значения тоже могут попадать в ту же фазу цикла.** Если период равен семи дням, то `seasonality=14`
также обращается к тому же дню недели, только двумя неделями раньше. Поэтому `7` и `14` — осмысленные
кандидаты для дневного ряда с недельной структурой; какой из них лучше, нужно проверять на отложенном периоде.

**Золотое правило: `seasonality ≥ horizon`.** Оно переносится из `NaiveModel`, где аналогичное условие
записывалось как `lag ≥ horizon`. Если горизонт длиннее сезонного шага, начиная с шага
`seasonality + 1` модель может обратиться к собственным предыдущим прогнозам, а не только к фактам.

В частности, при `seasonality=7` и горизонте `14` вторая неделя прогноза уже использует значения первой недели.
При `window=1` это приводит к точному повторению первой недели во второй. Увеличение `window` меняет состав
усредняемых значений, но не отменяет сам факт рекурсии.

Условие `seasonality ≥ horizon` защищает от рекурсивного использования прогнозов на выбранном горизонте,
но само по себе не гарантирует точность: сезонность всё равно должна соответствовать структуре ряда.

## 2. Практика

### Задание 1. Синтетика: `seasonality` должна совпадать с периодом ряда

Начнём с синтетического дневного ряда с периодом `7` дней и без шума. В нём недельный паттерн повторяется
точно, поэтому можно изолированно проверить, что произойдёт при правильном и неправильном `seasonality`.

```python
from etna.datasets import TSDataset, generate_periodic_df
from etna.models import SeasonalMovingAverageModel
from etna.metrics import SMAPE
from etna.analysis import plot_forecast

df = generate_periodic_df(
    periods=42,
    start_time="2026-09-07",
    scale=67,
    period=7,
    n_segments=1,
    freq="D",
    add_noise=False,
    sigma=0,
    random_seed=42,
)

ts = TSDataset(df, freq="D")
ts.plot()
```

```{image} img/04-synthetic-weekly-pattern.png
:alt: Синтетический дневной ряд с повторяющимся недельным паттерном
```

Для сравнения используем один и тот же горизонт `7` дней и `window=1`. Меняется только `seasonality`.

```python
def get_forecast(data, horizon, window, seasonality, n_train_samples=None, is_log=True):
    train_ts, test_ts = data.train_test_split(test_size=horizon)
    model = SeasonalMovingAverageModel(window=window, seasonality=seasonality)
    model.fit(train_ts)

    future_ts = train_ts.make_future(
        future_steps=horizon,
        tail_steps=model.context_size,
    )
    forecast_ts = model.forecast(
        ts=future_ts,
        prediction_size=horizon,
    )

    if is_log:
        plot_forecast(
            forecast_ts=forecast_ts,
            train_ts=train_ts,
            test_ts=test_ts,
            n_train_samples=n_train_samples,
        )

    return train_ts, test_ts, forecast_ts
```

Сначала зададим правильный недельный шаг:

```python
train_ts, test_ts, forecast_ts = get_forecast(
    data=ts,
    horizon=7,
    window=1,
    seasonality=7,
)

SMAPE()(y_true=test_ts, y_pred=forecast_ts)
```

```{image} img/04-synthetic-seasonality-7.png
:alt: Точный прогноз синтетического недельного ряда при seasonality равном 7
```

`SMAPE = 0`: модель каждый день прогнозирует по значению того же дня недели из предыдущего цикла.

Теперь изменим только сезонный шаг:

```python
train_ts, test_ts, forecast_ts = get_forecast(
    data=ts,
    horizon=7,
    window=1,
    seasonality=3,
)

SMAPE()(y_true=test_ts, y_pred=forecast_ts)
```

```{image} img/04-synthetic-seasonality-3.png
:alt: Неточный прогноз синтетического недельного ряда при seasonality равном 3
```

Результаты:

| `seasonality` | SMAPE |
|---:|---:|
| 7 | **0.00** |
| 3 | 96.89 |

**Вопрос:** почему `seasonality=3` даёт настолько большую ошибку, а не просто немного ухудшает прогноз?

:::{dropdown} Ответ
Модель берёт значения не той фазы недельного цикла. Например, для прогноза на понедельник при `seasonality=3`
она обращается к пятнице. Для вторника — к субботе и так далее.

Поскольку реальный ряд повторяет недельный паттерн, ошибка получается систематической: модель последовательно
сравнивает разные дни недели. При `seasonality=7` она, наоборот, использует предыдущие понедельники для
понедельника, предыдущие вторники для вторника и т. д.
:::

### Задание 1.1. Золотое правило: `seasonality ≥ horizon`

Оставим правильный недельный шаг `seasonality=7`, но увеличим горизонт до `14` дней. Это нарушение условия
`seasonality ≥ horizon`: вторая неделя прогноза уже должна обращаться к значениям первой недели.

Проверим два случая — без шума и с шумом:

```python
# Без шума
train_ts, test_ts, forecast_ts = get_forecast(
    data=ts,
    horizon=14,
    window=1,
    seasonality=7,
)

forecast_values = forecast_ts[:, :, "target"].values.flatten()
week_1, week_2 = forecast_values[:7], forecast_values[7:]

print("Неделя 1:", np.round(week_1, 2))
print("Неделя 2:", np.round(week_2, 2))
print("Недели совпадают:", np.allclose(week_1, week_2))
print("SMAPE:", SMAPE()(y_true=test_ts, y_pred=forecast_ts))
```

```{image} img/04-horizon-14-no-noise.png
:alt: Прогноз на две недели при seasonality 7 на синтетическом ряде без шума
```

Затем добавим шум в исходный ряд:

```python
df_noisy = generate_periodic_df(
    periods=42,
    start_time="2026-09-07",
    scale=67,
    period=7,
    n_segments=1,
    freq="D",
    add_noise=True,
    sigma=5,
    random_seed=42,
)

ts_noisy = TSDataset(df_noisy, freq="D")

train_ts_n, test_ts_n, forecast_ts_n = get_forecast(
    data=ts_noisy,
    horizon=14,
    window=1,
    seasonality=7,
)

forecast_values_n = forecast_ts_n[:, :, "target"].values.flatten()
week_1_n, week_2_n = forecast_values_n[:7], forecast_values_n[7:]

print("Неделя 1:", np.round(week_1_n, 2))
print("Неделя 2:", np.round(week_2_n, 2))
print("Недели совпадают:", np.allclose(week_1_n, week_2_n))
print("SMAPE:", SMAPE()(y_true=test_ts_n, y_pred=forecast_ts_n))
```

```{image} img/04-horizon-14-noisy.png
:alt: Прогноз на две недели при seasonality 7 на синтетическом ряде с шумом
```

Сохранённые результаты:

| Данные | Прогноз первой недели | Прогноз второй недели | Недели совпадают | SMAPE |
|---|---|---|---|---:|
| Без шума | `[51, 14, 60, 20, 23, 2, 21]` | Копия первой | Да | **0.00** |
| С шумом | `[49.87, 14.34, 52.88, 17.28, 23.55, -3.75, 22.88]` | Копия первой | Да | 25.28 |

**Вопрос:** почему без шума рекурсия не ухудшает метрику, а с шумом ошибка становится заметной?

:::{dropdown} Ответ
При `window=1` и `seasonality=7` прогноз на каждый день второй недели берёт значение соответствующего дня
первой недели. Поэтому вторая неделя становится точной копией первой.

В синтетическом ряде без шума недельный паттерн повторяется идеально: первая неделя спрогнозирована точно,
а её копия также совпадает с фактами. Поэтому `SMAPE = 0`, хотя модель уже использует собственные прогнозы.

При добавлении шума первая неделя содержит отклонения от идеального паттерна. Модель копирует не только
сезонный рисунок, но и эти отклонения. Ошибка первой недели переносится на вторую, поэтому `SMAPE` возрастает
до `25.28`.
:::

### Задание 2. Загружаем два юнита размера `10+`

Для основной практики используем два сегмента из размерной группы `10+`: типичный (`typical`, `p25`) и более
загруженный (`busier`, `p75`). Даты начинаются с `2026-04-01`, частота — дневная.

```python
import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.models import SeasonalMovingAverageModel
from etna.metrics import SMAPE
from etna.analysis import plot_forecast

DATA_DIR = "../local/data"

units_df = (
    pd.read_csv(f"{DATA_DIR}/units.csv")
    .rename(columns={
        "MaskedUnitId": "segment",
        "UnitSize": "unit_size",
        "PickReason": "pick_reason",
        "TotalOrders": "total_orders",
    })
    [["segment", "unit_size", "pick_reason", "total_orders"]]
)

orders_df = (
    pd.read_csv(f"{DATA_DIR}/facts-by-date.csv")
    .rename(columns={
        "CalendarDate": "timestamp",
        "MaskedUnitId": "segment",
        "Fact": "target",
    })
    [["timestamp", "segment", "target"]]
)

# Выравниваем начало рядов по дате
orders_df = orders_df[orders_df["timestamp"] >= "2026-04-01"]

# Оставляем два юнита размера 10+
orders_sample_df = orders_df.merge(
    units_df[units_df["unit_size"] == "10+"]["segment"],
    how="inner",
    on="segment",
)

orders_sample_ts = TSDataset(orders_sample_df, freq="D")

orders_sample_ts.describe()
orders_sample_ts.plot()
```

```{image} img/04-two-units.png
:alt: Дневная динамика фактических заказов двух юнитов размера 10+
```

В дальнейших экспериментах используются сегменты `70650540` (`typical`, p25) и `a6dffdeb` (`busier`, p75).

### Задание 3. Подбираем `seasonality` на одном юните

Сначала возьмём более загруженный сегмент `a6dffdeb`. Зафиксируем `window=3` и горизонт `14` дней, а затем
сравним четыре сезонных шага: `1`, `3`, `7` и `14`.

```python
orders_one_ts = TSDataset(
    df=orders_sample_ts[:, "a6dffdeb", :],
    freq=orders_sample_ts.freq,
)

records = []
forecasts_by_seasonality = {}

for seasonality in [1, 3, 7, 14]:
    train_ts, test_ts, forecast_ts = get_forecast(
        data=orders_one_ts,
        horizon=14,
        window=3,
        seasonality=seasonality,
        is_log=False,
    )

    smape_per_segment = SMAPE()(y_true=test_ts, y_pred=forecast_ts)
    records.append({"seasonality": seasonality, **smape_per_segment})
    forecasts_by_seasonality[f"seasonality={seasonality}"] = forecast_ts

results_df = pd.DataFrame(records).set_index("seasonality")
display(results_df.sort_values("a6dffdeb"))

plot_forecast(
    forecast_ts=forecasts_by_seasonality,
    test_ts=test_ts,
    train_ts=train_ts,
    n_train_samples=14,
)
```

```{image} img/04-one-unit-seasonality.png
:alt: Сравнение прогнозов одного юнита при seasonality 1, 3, 7 и 14
```

Сохранённые значения `SMAPE` (меньше — лучше):

| `seasonality` | SMAPE для `a6dffdeb` |
|---:|---:|
| 1 | 15.07 |
| 3 | 15.14 |
| 7 | 6.62 |
| 14 | **6.22** |

**Вопрос:** какая сезонность выглядит осмысленной для дневных данных о заказах? Почему `14` немного лучше
`7`, если недельный цикл равен семи дням?

:::{dropdown} Ответ
Для дневных заказов естественный кандидат — недельная сезонность `7`: она сравнивает каждый день с тем же
днём недели. Значение `14` также сохраняет эту фазу, но берёт наблюдения на две недели раньше. Поэтому оба
варианта выглядят осмысленно, в отличие от `1` и `3`, которые в этом эксперименте дали существенно большую
ошибку.

На выбранном тестовом отрезке `seasonality=14` оказался немного лучше (`SMAPE=6.22` против `6.62` у
`seasonality=7`). Это результат конкретного сегмента и периода проверки, а не доказательство универсального
двухнедельного цикла. Разницу нужно оценивать на других отложенных периодах.
:::

### Задание 4. Подбираем `window` и `seasonality` на двух юнитах

Теперь обучим модель сразу на обоих сегментах. Проверим шесть сочетаний параметров:

- `window`: `1`, `3`, `5`;
- `seasonality`: `7`, `14`.

Для каждого сочетания посчитаем `SMAPE` отдельно по сегментам и агрегированно в режиме `macro`.

```python
records = []
forecasts_by_combo = {}

for window in [1, 3, 5]:
    for seasonality in [7, 14]:
        train_ts, test_ts, forecast_ts = get_forecast(
            data=orders_sample_ts,
            horizon=14,
            window=window,
            seasonality=seasonality,
            is_log=False,
        )

        smape_per_segment = SMAPE()(y_true=test_ts, y_pred=forecast_ts)
        smape_agg = SMAPE(mode="macro")(y_true=test_ts, y_pred=forecast_ts)

        records.append({
            "window": window,
            "seasonality": seasonality,
            "smape_agg": smape_agg,
            **smape_per_segment,
        })
        forecasts_by_combo[f"window={window}, seasonality={seasonality}"] = forecast_ts

results_df = pd.DataFrame(records).set_index(["window", "seasonality"])
display(results_df.sort_index())

plot_forecast(
    forecast_ts=forecasts_by_combo,
    test_ts=test_ts,
    train_ts=train_ts,
    n_train_samples=14,
)
```

```{image} img/04-window-seasonality-comparison.png
:alt: Сравнение прогнозов для двух юнитов при разных сочетаниях window и seasonality
```

Сохранённые значения `SMAPE`:

| `window` | `seasonality` | `70650540` (typical) | `a6dffdeb` (busier) | `SMAPE macro` |
|---:|---:|---:|---:|---:|
| 1 | 7 | 8.91 | 7.58 | 8.24 |
| 1 | 14 | 11.26 | 6.93 | 9.09 |
| 3 | 7 | 8.49 | 6.62 | **7.55** |
| 3 | 14 | 12.92 | 6.22 | 9.57 |
| 5 | 7 | 9.71 | 6.70 | 8.21 |
| 5 | 14 | 13.55 | **5.79** | 9.67 |

Минимальный агрегированный `SMAPE macro` получился при `window=3`, `seasonality=7` — `7.55`. Но оптимум
отличается по сегментам: для typical-юнита лучший результат также даёт `window=3`, `seasonality=7` (`8.49`),
а для busier-юнита — `window=5`, `seasonality=14` (`5.79`).

В предыдущей главе лучший вариант `MovingAverageModel` (`window=14`) дал `SMAPE macro = 17.89` на тех же
двух сегментах и горизонте. В этом эксперименте сезонная модель с `window=3`, `seasonality=7` снизила
агрегированную ошибку до `7.55`.

**Вопрос:** означает ли лучший общий результат `window=3`, `seasonality=7`, что эти параметры нужно использовать
для каждого юнита?

:::{dropdown} Ответ
Нет. `window=3`, `seasonality=7` — лучший вариант по агрегированной метрике на двух выбранных сегментах.
Но для busier-юнита минимальный `SMAPE` получился при `window=5`, `seasonality=14` (`5.79`).

Это ещё раз показывает, что общий оптимум не обязательно является лучшим для каждого сегмента. При подборе
параметров полезно смотреть одновременно на агрегированную метрику и на ошибки отдельных рядов.
:::

## 3. Итоги

- **`SeasonalMovingAverageModel` усредняет значения с постоянным шагом `seasonality`.** При дневных данных и
  `seasonality=7` модель сопоставляет одинаковые дни недели.
- **`window` задаёт число усредняемых значений, а `seasonality` — расстояние между ними.**
  `context_size = window * seasonality`.
- **`NaiveModel` и `MovingAverageModel` — частные случаи сезонной модели.** Для первого `window=1`, для второго
  `seasonality=1`.
- **Неправильный сезонный шаг создаёт систематическую ошибку.** На синтетическом недельном ряде `seasonality=7`
  дал `SMAPE=0`, а `seasonality=3` — около `96.89`.
- **Условие `seasonality ≥ horizon` защищает от рекурсивного использования прогнозов.** Если горизонт больше
  сезонного шага, модель начинает обращаться к собственным предыдущим предсказаниям.
- **Кратные значения сезонного периода могут быть осмысленными.** Для недельного ритма `7` и `14` обращаются
  к одной фазе цикла, но на разной глубине истории.
- **Сезонная модель заметно улучшила результат на двух выбранных юнитах.** Лучшее сочетание
  `window=3`, `seasonality=7` дало `SMAPE macro=7.55` против `17.89` у лучшего `MovingAverageModel` в
  предыдущей главе.
- **Единого лучшего сочетания для всех сегментов может не быть.** Сравнивай агрегированную метрику и
  результаты по отдельным сегментам на отложенном периоде.
