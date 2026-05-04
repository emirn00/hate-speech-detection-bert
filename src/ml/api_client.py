import requests
import json

def test_prediction(text):
    url = "http://localhost:8000/predict"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"--- Analysis Result ---")
            print(f"Input Text: {result['text']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Model Used: {result['model_used']}")
            print(f"-----------------------")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    print("Hate Speech Detection Client")
    while True:
        user_input = input("\nEnter text to analyze (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        test_prediction(user_input)
