
# 📊 Customer Churn Prediction API

Прогнозирование оттока клиентов телеком-компании с помощью Machine Learning. Проект включает полный цикл: от анализа данных до работающего API.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🎯 О проекте

Проект решает задачу бинарной классификации для прогнозирования оттока клиентов телеком-компании. Модель определяет клиентов с высоким риском ухода, что позволяет принять превентивные меры по их удержанию.

**Ключевые особенности:**
- Полный ML pipeline: от EDA до production API
- RESTful API на FastAPI с автоматической документацией
- Интерактивные визуализации и feature importance анализ
- Готовое к развертыванию решение

## 📊 Результаты

### Модель
- **Лучшая модель:** Logistic Regression (с балансировкой классов)
- **Метрика:** F1-score = 0.6324
- **Recall:** 79% (выявляем 79% уходящих клиентов)

### Key Insights
1. **Главный фактор удержания:** время сотрудничества (tenure)
2. **Высокий риск оттока:** клиенты с помесячным контрактом
3. **Неожиданная находка:** клиенты с дорогими тарифами более лояльны
4. **Проблемная услуга:** Fiber optic показывает высокий уровень оттока

## 🏗️ Структура проекта

### *customer-churn-prediction/*
├──app/  
│ ├── artifacts/  
│ │ ├── best_churn_model.pkl  
│ │ ├── scaler.pkl  
│ │ └── feature_list.json  
│ ├── main.py  
│ └── requirements.txt  
├── data/   
│ └── telco_churn.csv  
├── notebooks/  
│ ├── 01_eda.ipynb  
│ ├── 02_modeling.ipynb    
│ └── 03_results.ipynb  
**└── README.md**  


## 🚀 Быстрый старт

### Установка и запуск

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```
2. **Установите зависимости:**
```bash
pip install -r app/requirements.txt
```
3. **Запустите API:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
4. **Откройте документацию API:**
Перейдите по адресу: [localhost:8000/docs](http://localhost:8000/docs)


### Пример запроса к API
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    # ... все остальные поля
}

response = requests.post(url, json=data)
print(response.json())
```

## 📈 API Endpoints
- POST /predict - Предсказание оттока клиента

- GET /features - Список всех признаков модели

- GET /model-info - Информация о модели

- GET /docs - Интерактивная документация (Swagger)

## 🛠️ Технологии

- Python: pandas, numpy, scikit-learn, xgboost

- ML: Logistic Regression, Random Forest, GridSearchCV

- Web: FastAPI, Uvicorn, Pydantic

- Visualization: matplotlib, seaborn

- Notebooks: Jupyter, IPython

## 📊 Данные
Данные содержат информацию о 7043 клиентах телеком-компании. Включает 20 признаков:

- Демографические данные (пол, возраст)

- Информация об услугах (тип интернета, контракта)

- Платежная информация (ежемесячные платежи, общая сумма)

## 👥 Автор
### **Швед Максим Петрович**

GitHub: @doomer2009

LinkedIn: [me >>](https://www.linkedin.com/in/%D1%88%D0%B2%D0%B5%D0%B4-%D0%BC%D0%B0%D0%BA%D1%81%D0%B8%D0%BC/)