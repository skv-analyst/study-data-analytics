import mlflow

# Настраиваем адрес локального сервера MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Имя эксперимента под твой учебный проект
mlflow.set_experiment("Study-Etna-Forecasting")


def log_to_mlflow(model_name: str, model_params: dict, metrics: dict, run_name_suffix: str = None):
    """
    Универсальная функция для логирования ЛЮБЫХ учебных моделей, параметров и метрик.

    :param model_name: Название модели (например, 'MovingAverage', 'SeasonalMA', 'CatBoost')
    :param model_params: Словарь с параметрами модели {'window': 5, 'seasonality': 24}
    :param metrics: Словарь с любыми метриками из курса {'WAPE': 32.2, 'MAE': 1.5, 'RMSE': 0.8}
    :param run_name_suffix: Опциональный суффикс для имени запуска (например, 'baseline' или 'v2')
    """
    # Формируем имя запуска: либо просто имя модели, либо модель + твой суффикс
    run_name = f"{model_name}_{run_name_suffix}" if run_name_suffix else model_name

    # Открываем контекст MLflow и динамически пишем всё, что пришло
    with mlflow.start_run(run_name=run_name):
        # 1. Логируем параметры модели
        mlflow.log_params(model_params)

        # 2. Логируем метрики циклом (абсолютно любые, какие посчитаешь в ноутбуке)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

    # Аккуратный вывод в консоль ноутбука для самоконтроля
    print(f"\n[MLflow] Результаты сохранены в проект 'Study-Etna-Forecasting'")
    print(f"Запуск    : {run_name}")
    print(f"Параметры : {model_params}")
    metrics_str = " | ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
    print(f"Метрики   : {metrics_str}")