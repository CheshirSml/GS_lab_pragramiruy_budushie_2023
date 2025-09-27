# 🎬 Рекомендательная система: ТОП-20 фильмов для каждого пользователя  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)  
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-orange)  
![Scikit-learn / Surprise](https://img.shields.io/badge/ML-Library-lightgrey)  
![License](https://img.shields.io/badge/License-MIT-green)

Рекомендательная система, разработанная в рамках хакатона **«Программируй будущее»** (2023).  
Проект предсказывает **ТОП-20 фильмов**, которые конкретный пользователь с наибольшей вероятностью посмотрит в будущем, на основе его истории просмотров и/или оценок. 
GRU4Rec + lightfm модель  обучена на информации о просмотрах фильмов около 200 000+ пользователей за 70 дней.

---

## 🎯 Цель

Создать персонализированную систему рекомендаций, которая:
- Анализирует поведение пользователей (оценки, просмотры);
- Учитывает схожесть вкусов между пользователями и схожесть фильмов;
- Формирует индивидуальный **ТОП-20 фильмов к просмотру** для каждого пользователя.

---

## 🧠 Подход

В проекте реализована **коллаборативная фильтрация** на основе:
- **User-Based** или **Item-Based** рекомендаций  

---

* [EDA](https://github.com/CheshirSml/GS_lab_pragramiruy_budushie_2023/blob/main/notebooks/EDA.ipynb)
* [Обучение модели](https://github.com/CheshirSml/GS_lab_pragramiruy_budushie_2023/blob/main/notebooks/LfmGRU4Rec_.ipynb)
---
## 🚀 Как запустить

### Требования
- Python 3.8+
- Библиотеки: `pandas`, `numpy`, `scikit-learn` (или `surprise`, `pickle` и др.)

### Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/CheshirSml/GS_lab_pragramiruy_budushie_2023.git
   cd GS_lab_pragramiruy_budushie_2023
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Запустите рекомендательную систему:
   ```bash
   python main.py
   ```

   Или откройте Jupyter Notebook для интерактивного анализа:
   ```bash
   jupyter notebook notebooks/recommender_demo.ipynb
   ```

---

## 📋 Пример использования

```python
from recommender import recommend_top20

user_id = 12345
top20 = recommend_top20(user_id)
print("ТОП-20 фильмов для вас:")
for i, movie in enumerate(top20, 1):
    print(f"{i}. {movie}")
```

**Вывод:**
```
ТОП-20 фильмов для вас:
1. Интерстеллар
2. Начало
3. Матрица
...
```

---

## 📁 Структура проекта

```
GS_lab_pragramiruy_budushie_2023/
├── main.py                 # Точка входа
├── recommender.py          # Логика рекомендаций
├── data/
│   ├── ratings.csv         # Оценки пользователей (MovieLens или аналог)
│   └── movies.csv          # Метаданные фильмов
├── models/                 # Сохранённые модели (если есть)
├── notebooks/              # Анализ и эксперименты
├── requirements.txt        # Зависимости
└── README.md
```


## 📜 Лицензия

Проект распространяется под лицензией **MIT**.  
См. файл [LICENSE](LICENSE).




