import os
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
documents = [
    "धान (Rice) की IR-64 किस्म भारत में उत्तर प्रदेश और छत्तीसगढ़ में अधिक उगाई जाती है और इसका औसत उत्पादन 4–5 टन प्रति हेक्टेयर होता है।",
    "गेहूं की HD-2967 किस्म पंजाब और हरियाणा में लोकप्रिय है और यह 140–145 दिनों में तैयार होती है।",
    "कपास की Bt Cotton किस्म भारत में 90% से अधिक क्षेत्र में उपयोग होती है क्योंकि यह bollworm resistance देती है।",
    "गन्ने की Co-0238 किस्म महाराष्ट्र में अधिक उत्पादक है और इसका औसत उत्पादन 100–120 टन प्रति हेक्टेयर होता है।",
    "टमाटर की Pusa Ruby किस्म भारत में सर्दी और गर्मी दोनों मौसम में उगाई जा सकती है।",
    "भारत में यूरिया (Urea) में 46% नाइट्रोजन होता है और यह सबसे अधिक उपयोग होने वाला उर्वरक है।",
    "मिट्टी परीक्षण (Soil Testing) हर 2–3 साल में करने की सलाह कृषि विश्वविद्यालय देते हैं ताकि NPK संतुलन बनाए रखा जा सके।",
    "ड्रिप सिंचाई से भारत में 30–60% तक पानी की बचत की जा सकती है, खासकर महाराष्ट्र और कर्नाटक में।",
    "जैविक खेती में नीम आधारित कीटनाशक (Neem Oil) का उपयोग भारत में प्राकृतिक pest control के लिए किया जाता है।",
    "भारत में खरीफ फसलें जून-जुलाई में मानसून पर निर्भर होती हैं और इनमें धान, मक्का और सोयाबीन प्रमुख हैं।"
]
doc_emb = model.encode(documents)

def answer(question):
    # Step 1: Encode question
    q_emb = model.encode(question)
    scores = util.cos_sim(q_emb, doc_emb)

    # Step 2: Get top 2 results
    top2 = scores[0].topk(2)

    # Step 3: Build context
    context = ""
    for idx in top2.indices:
        context += documents[int(idx)] + "\n"

    # Step 4: Create prompt
    prompt = f"""
        नीचे दिए गए संदर्भ के आधार पर उत्तर दें।
        यदि संदर्भ आंशिक है, तो उपलब्ध जानकारी के आधार पर उत्तर दें।

        संदर्भ:
        {context}

        प्रश्न: {question}

        उत्तर:
    """

    # Step 5: Call Gemini
    model_gemini = genai.GenerativeModel("gemini-2.5-flash")
    response = model_gemini.generate_content(prompt)

    return response.text

gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="अपना सवाल हिंदी में लिखें"),
    outputs=gr.Textbox(label="जवाब"),
    title="Simple semantic search engine with RAG",
    description="कृषि संबंधी सवाल पूछें"
).launch()


# test_questions = [
#     "HD-2967 wheat kitne din me ready hota hai",
#     "Bt cotton kyu use hota hai India me",
#     "IR-64 rice kahan ugta hai",
#     "sugarcane ka average yield kya hai",
#     "drip irrigation kitna paani bachata hai",
#     "urea me kitna nitrogen hota hai",
#     "soil testing kitne saal me karni chahiye",
#     "Pusa Ruby tomato kab ugta hai",
#     "India me kaunsi crops kharif hoti hain",
#     "bollworm resistance kya hota hai cotton me"
# ]