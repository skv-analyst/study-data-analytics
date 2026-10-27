# ETNA: прогнозирование временных рядов

Конспекты по библиотеке [ETNA](https://github.com/etna-team/etna) — фреймворку для прогнозирования временных рядов: краткая теория по каждой модели, практические задания на реальных данных и выводы.

## Материалы

- Официальная [документация библиотеки](https://docs.etna.ai/stable/index.html).
- Маршрут изучения — книга Артема Груздева **«Пособие по библиотеке временных рядов ETNA»**. На Boosty автора можно
  посмотреть [оглавление](https://boosty.to/drains/posts/26582567-a30c-41d0-915d-0ddbadcca714) и приобрести книгу
  за [1500руб](https://boosty.to/drains/posts/8b02f186-b785-403b-a371-7581b8929eab?share=post_link).
- [Официальные ноутбуки-примеры](https://github.com/etna-team/etna/tree/3.0.0/examples) из `etna-team/etna`, тег
  `3.0.0` (совпадает с версией библиотеки в этом проекте). Полезно сверяться с ними, когда код из книги (которая
  писалась под ETNA 2.9) расходится с актуальным API 3.0.0.

## Данные

Для практических заданий взяты реальные данные о заказах, но старые и намеренно искажённые. По форме они похожи на
настоящие: в них есть недельная сезонность, тренды, юниты разного размера и масштаба — то, на чём хорошо видно, как
ведут себя модели. Но сами цифры не соответствуют реальности, и делать по ним выводы о каком-либо реальном бизнесе
нельзя.

## Настройка окружения

- Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями.
- Так как библиотека заблокирована в PyPI, то ставим из GitHub-репозитория разработчиков.

### macOS (Apple Silicon: M1/M2/M3/M4)

Зависимости `numba`/`llvmlite` для этой архитектуры не имеют прекомпилированных колёс и собираются из исходников — нужен
компилятор LLVM:

```bash
brew install llvm
uv init --python 3.11
LLVM_CONFIG="$(brew --prefix llvm)/bin/llvm-config" uv add "etna[all]@git+https://github.com/etna-team/etna.git@3.0.0"
```

### macOS (Intel) и Linux

Для этих платформ у `numba`/`llvmlite` есть готовые колёса на PyPI, компилятор не нужен:

```bash
uv init --python 3.11
uv add "etna[all]@git+https://github.com/etna-team/etna.git@3.0.0"
```

### Windows

Установи uv (PowerShell):

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Затем в терминале проекта:

```powershell
uv init --python 3.11
uv add "etna[all]@git+https://github.com/etna-team/etna.git@3.0.0"
```

Как и в случае с macOS Intel/Linux, готовые колёса `numba`/`llvmlite` под Windows есть на PyPI, поэтому дополнительный
компилятор не требуется.

### Если репозиторий уже склонирован (pyproject.toml и uv.lock уже в git)

Достаточно один раз выполнить в папке `05-etna`:

```bash
uv sync
```

### Проверка установки

```bash
uv run python -c "
from importlib.metadata import version
print('etna version:', version('etna'))
from etna.datasets.tsdataset import TSDataset
from etna.models import NaiveModel
print('OK: TSDataset and NaiveModel imported successfully')
"
```

Ожидаемый вывод:

```
etna version: 3.0.0
OK: TSDataset and NaiveModel imported successfully
```

### Модель анализатора прогнозируемости

Для тетрадки `13-predictability.ipynb` нужен предобученный `PredictabilityAnalyzer`. В библиотеку он не входит:
файл (7.8 МБ) скачивается с сервера разработчиков ETNA в папку `local/models/`, которая в git не попадает.

```bash
mkdir -p local/models
curl -o local/models/weasel_D.pickle "http://etna-github-prod.cdn-tinkoff.ru/series_classification/22_11_2022/D/weasel.pickle"
uv run python -W ignore -c "from etna.experimental.classification import PredictabilityAnalyzer as P; P.load('local/models/weasel_D.pickle'); print('OK')"
```

## Логирование экспериментов (MLflow)

Метрики моделей из ноутбуков логируются через `log_to_mlflow()` (см. [`src/custom_logger.py`](src/custom_logger.py))
в локальный MLflow — трекинг полностью на диске, ничего никуда не отправляется. База лежит в
`local/mlruns/mlflow.db` (`local/` в `.gitignore`, в репозиторий не попадает).

Чтобы посмотреть результаты в вебе, из папки `05-etna`:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///local/mlruns/mlflow.db" --port 5050
```

И открыть [http://127.0.0.1:5050](http://127.0.0.1:5050) в браузере. Все запуски лежат в одном эксперименте
`study-etna` — так удобнее сравнивать модели между собой.

> Порт `5000` на macOS (начиная с Monterey) занят системным **AirPlay Receiver** — он перехватывает часть
> запросов и отдаёт свою страницу «Доступ запрещён / HTTP ERROR 403» вместо ответа сервера. Поэтому здесь
> `5050`, а не стандартный дефолт `5000` — если всё же нужен именно 5000, отключи AirPlay Receiver в
> Системных настройках → Основные → AirDrop и Handoff.

**Как остановить:** если сервер запущен в терминале на переднем плане — `Ctrl+C` в этом терминале. Если он был
запущен в фоне (например, через `&` или отдельной вкладкой, которую уже закрыли), найти и остановить процесс:

```bash
lsof -ti tcp:5050 | xargs kill
```

Веб-интерфейс — это отдельный процесс поверх той же базы `local/mlruns/mlflow.db`; его остановка не удаляет
и не портит уже залогированные эксперименты, а сам `log_to_mlflow()` из ноутбуков от него не зависит и работает
без поднятого UI.
