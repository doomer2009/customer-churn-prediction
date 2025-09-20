from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from typing import Dict, List, Optional
import numpy as np

# Загружаем все наши артефакты
try:
    model = joblib.load('app/artifacts/best_churn_model.pkl')
    scaler = joblib.load('app/artifacts/scaler.pkl')
    with open('app/artifacts/feature_list.json', 'r') as f:
        feature_list = json.load(f)
    print("✅ Все артефакты успешно загружены")
except FileNotFoundError as e:
    print(f"❌ Ошибка загрузки артефактов: {e}")
    print("Сначала обучи модель и сохрани артефакты!")
    exit(1)

# Создаем приложение
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API для прогнозирования оттока клиентов телеком-компании",
    version="1.0.0"
)

# Описываем формат входных данных
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    churn_risk: str
    features_importance: Optional[Dict] = None

# Функция для предобработки новых данных
def preprocess_data(input_data: Dict) -> pd.DataFrame:
    """Преобразует сырые данные в формат для модели"""
    
    # Создаем DataFrame
    df = pd.DataFrame([input_data])
    
    # 1. Кодируем бинарные признаки (как при обучении)
    binary_mapping = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].map(binary_mapping).fillna(df[col])
    
    # 2. One-Hot Encoding для категориальных признаков
    categorical_columns = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    
    # Применяем get_dummies как при обучении
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dtype=int)
    
    # 3. Убедимся, что все фичи в правильном порядке
    # Добавляем недостающие колонки (если в новых данных нет каких-то категорий)
    for feature in feature_list:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0  # Добавляем недостающую колонку с нулями
    
    # Убираем лишние колонки (если появились новые)
    df_encoded = df_encoded[feature_list]
    
    # 4. Масштабируем данные (ВАЖНО: используем тот же scaler!)
    scaled_data = scaler.transform(df_encoded)
    
    return pd.DataFrame(scaled_data, columns=feature_list)

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer_data: CustomerData):
    """Предсказывает вероятность оттока клиента"""
    try:
        # Преобразуем данные в словарь
        input_dict = customer_data.dict()
        
        # Предобрабатываем данные
        processed_data = preprocess_data(input_dict)
        
        # Делаем предсказание
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)
        
        # Определяем уровень риска
        risk_level = "high" if probability[0][1] > 0.7 else "medium" if probability[0][1] > 0.3 else "low"
        
        return {
            "churn_prediction": bool(prediction[0]),
            "churn_probability": float(probability[0][1]),
            "churn_risk": risk_level
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API is running!", "version": "1.0.0"}

@app.get("/features")
async def get_features():
    """Возвращает список всех признаков, которые ожидает модель"""
    return {"features": feature_list}

@app.get("/model-info")
async def model_info():
    """Информация о модели"""
    return {
        "model_type": str(type(model).__name__),
        "features_count": len(feature_list),
        "model_classes": model.classes_.tolist() if hasattr(model, 'classes_') else None
    }

# Запуск сервера (для разработки)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)