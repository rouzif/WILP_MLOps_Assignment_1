import time
import random
import requests

API_URL = "http://localhost:8000"

# Sample valid patient payloads
VALID_PAYLOADS = [
    {
        "age": 63.0, "sex": 1, "cp": 1, "trestbps": 145.0, "chol": 233.0,
        "fbs": 1, "restecg": 2, "thalach": 150.0, "exang": 0,
        "oldpeak": 2.3, "slope": 3, "ca": 0.0, "thal": 6.0
    },
    {
        "age": 41.0, "sex": 0, "cp": 2, "trestbps": 130.0, "chol": 204.0,
        "fbs": 0, "restecg": 2, "thalach": 172.0, "exang": 0,
        "oldpeak": 1.4, "slope": 1, "ca": 0.0, "thal": 3.0
    },
    {
        "age": 67.0, "sex": 1, "cp": 4, "trestbps": 160.0, "chol": 286.0,
        "fbs": 0, "restecg": 2, "thalach": 108.0, "exang": 1,
        "oldpeak": 1.5, "slope": 2, "ca": 3.0, "thal": 3.0
    }
]

INVALID_PAYLOAD = {
    "age": "invalid_string",  # Triggers 422 error
    "sex": 1
}


def send_traffic(iterations=50):
    print(f"=== Generating traffic towards {API_URL} ({iterations} requests) ===")

    for i in range(1, iterations + 1):
        try:
            # Randomly select endpoint type
            choice = random.choices(["health", "predict_valid", "predict_invalid"], weights=[0.2, 0.7, 0.1])[0]

            if choice == "health":
                res = requests.get(f"{API_URL}/health")
                print(f"[{i}/{iterations}] GET /health -> Status {res.status_code}")

            elif choice == "predict_valid":
                payload = random.choice(VALID_PAYLOADS)
                res = requests.post(f"{API_URL}/predict", json=payload)
                print(f"[{i}/{iterations}] POST /predict -> Status {res.status_code}")

            elif choice == "predict_invalid":
                res = requests.post(f"{API_URL}/predict", json=INVALID_PAYLOAD)
                print(f"[{i}/{iterations}] POST /predict (Invalid) -> Status {res.status_code}")

        except Exception as e:
            print(f"[{i}/{iterations}] Request failed: {e}")

        time.sleep(0.2)  # Short delay between requests


if __name__ == "__main__":
    send_traffic(50)