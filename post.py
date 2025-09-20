import requests

data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes", 
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No", 
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.9,
    "TotalCharges": 958.8
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())