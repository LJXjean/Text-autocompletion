import requests

url = "http://localhost:8000/autocomplete"
payload = {
    "text_before_cursor": "The quick brown fox",
    "max_length": 50
}

response = requests.post(url, json=payload)
print(response.json()) 