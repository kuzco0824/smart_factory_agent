from agent import create_client, call_gemini

client = create_client()

response = call_gemini(
    client,
    "請回答：Gemini API 測試成功。"
)

print(response)