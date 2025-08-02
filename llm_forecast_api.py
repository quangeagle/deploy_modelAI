from fastapi import FastAPI, Request
import pandas as pd
import os
from openai import OpenAI

# Load kết quả dự báo (giả sử đã lưu từ pipeline ML)
forecast_df = pd.read_csv("sales_forecast.csv")

app = FastAPI()

# Khởi tạo client OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-...your-key..."))

def query_llm(question, context):
    prompt = f"""
    Dữ liệu dự báo doanh số:
    {context}
    Câu hỏi: {question}
    Trả lời:
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý phân tích dữ liệu doanh số, trả lời ngắn gọn, dễ hiểu."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data["question"]
    # Lấy context phù hợp: ví dụ 30 ngày gần nhất, hoặc toàn bộ nếu nhỏ
    context = forecast_df.tail(30).to_string()
    answer = query_llm(question, context)
    return {"answer": answer}

# Hướng dẫn sử dụng:
# 1. Chạy pipeline ML lưu kết quả dự báo ra sales_forecast.csv
# 2. Chạy file này: uvicorn llm_forecast_api:app --reload
# 3. Gửi POST tới /ask với JSON: {"question": "Doanh thu tháng tới ra sao?"} 