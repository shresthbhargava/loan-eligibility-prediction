# test_api.py — run this while the API is running
import requests

# Test 1: Health check
resp = requests.get("http://localhost:8000/health")
print("Health:", resp.json())

# Test 2: Model info
resp = requests.get("http://localhost:8000/model-info")
print("Model info:", resp.json())

# Test 3: High-risk applicant (should be rejected)
high_risk = {
    "Gender": "Male", "Married": "No", "Dependents": "3+",
    "Education": "Not Graduate", "Self_Employed": "Yes",
    "ApplicantIncome": 2000, "CoapplicantIncome": 0,
    "LoanAmount": 500, "Loan_Amount_Term": 360,
    "Credit_History": 0, "Property_Area": "Rural"
}
resp = requests.post("http://localhost:8000/predict", json=high_risk)
print("\nHigh-risk applicant:")
print(resp.json())

# Test 4: Low-risk applicant (should be approved)
low_risk = {
    "Gender": "Female", "Married": "Yes", "Dependents": "0",
    "Education": "Graduate", "Self_Employed": "No",
    "ApplicantIncome": 8000, "CoapplicantIncome": 3000,
    "LoanAmount": 150, "Loan_Amount_Term": 360,
    "Credit_History": 1, "Property_Area": "Semiurban"
}
resp = requests.post("http://localhost:8000/predict", json=low_risk)
print("\nLow-risk applicant:")
print(resp.json())

# Test 5: Invalid input (should return 422 validation error)
invalid = {**low_risk, "Gender": "Unknown", "ApplicantIncome": -5000}
resp = requests.post("http://localhost:8000/predict", json=invalid)
print("\nInvalid input (expect 422):", resp.status_code)
print(resp.json())