import os
import requests

api_key = os.getenv("TINYFISH_API_KEY")

if not api_key:
    print("❌ TINYFISH_API_KEY পাওয়া যায়নি")
    raise SystemExit

print("✅ TINYFISH_API_KEY পাওয়া গেছে")
print("🔄 TinyFish Search test শুরু হচ্ছে...")

url = "https://api.search.tinyfish.ai"

headers = {
    "X-API-Key": api_key
}

params = {
    "query": "latest news in India"
}

try:
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30
    )

    print("HTTP Status:", response.status_code)
    print("Response:")
    print(response.text[:3000])

except Exception as e:
    print("❌ Error:", e)