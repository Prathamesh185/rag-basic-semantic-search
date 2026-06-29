import ollama

from config import client


def local_llm(prompt):

    response = ollama.chat(
        model="qwen3.5:4b",
        think=False,
        options={
            "num_predict": 150,
            "temperature": 0.1,
            "num_thread": 8,
        },
        messages=[
            {
                "role": "system",
                "content": "तुम एक कृषि सहायक हो। केवल दिए गए संदर्भ से उत्तर दो। संदर्भ के बाहर कुछ मत बताओ।",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    raw = response["message"]["content"]

    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    if raw.strip():
        return raw

    return "इस विषय पर जानकारी उपलब्ध नहीं है."


def gemini_llm(prompt):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text