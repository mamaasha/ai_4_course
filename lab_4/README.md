# Лабораторная лабора №4

**Тема:** Проведение исследований со случайным лесом

---

## 2. Бейзлайн

* **Классификация:** новости спорта, `TfidfVectorizer` + `RandomForestClassifier`. Метрики: accuracy ~0.80, macro F1 ~0.79; похожие классы различаются хуже остальных.
* **Регрессия:** удой коров, минимальный препроцесс (imputer + scaler + OneHot) и `RandomForestRegressor`. MAE ~367, MSE ~6.37e6

---

## 3. Улучшенный бейзлайн

* **Классификация:** Optuna-подбор `n_estimators`, `max_depth`, `min_samples_*`, `max_features` на эмбеддингах. Лучшая accuracy ~0.75 — ниже tf-idf бейзлайна
* **Регрессия:** очищенные данные + подбор гиперпараметров случайного леса через Optuna. Итог: MAE ~123, MSE (подписано как RMSE) ~2.39e4

---

## 4. Собственная реализация

* **Классификация:** самописный `RandomForestTextClassifier` на tf-idf; accuracy ~0.80, macro F1 ~0.79 (сопоставимо со sklearn).
* **Регрессия:** `SimpleRandomForestRegressor` (bagging деревьев). MAE ~229, MSE ~7.49e4, R² ~0.68 — заметно лучше бейзлайна, но ниже результата после Optuna.
