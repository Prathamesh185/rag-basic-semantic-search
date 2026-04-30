import os
import gradio as gr
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ENV SETUP
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# EMBEDDING MODEL
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# PDF LOADER (optional)
def load_pdf(path):
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    chunks = text.split(". ")
                    texts.extend([c.strip() for c in chunks if len(c) > 50])
        print(f"✅ PDF loaded: {len(texts)} passages")
    except Exception as e:
        print(f"⚠️ PDF error: {e}")
    return texts

# PURE AGRICULTURE DOCUMENTS
documents = [
    # Crop varieties
    "धान (Rice) की IR-64 किस्म भारत में उत्तर प्रदेश और छत्तीसगढ़ में अधिक उगाई जाती है।",
    "गेहूं की HD-2967 किस्म पंजाब और हरियाणा में लोकप्रिय है और 140-145 दिनों में तैयार होती है।",
    "कपास की Bt Cotton किस्म भारत में 90% से अधिक क्षेत्र में उपयोग होती है।",
    "टमाटर की Pusa Ruby किस्म भारत में सर्दी और गर्मी दोनों मौसम में उगाई जा सकती है।",
    "मक्का की Hybrid किस्में 80-85 दिनों में तैयार होती हैं और अधिक उत्पादन देती हैं।",
    "गन्ने की Co-0238 किस्म महाराष्ट्र में अधिक उत्पादक है।",
    # Fertilizers
    "यूरिया में 46% नाइट्रोजन होता है और यह सबसे अधिक उपयोग होने वाला उर्वरक है।",
    "DAP खाद में 18% नाइट्रोजन और 46% फॉस्फोरस होता है।",
    "जैविक खाद से मिट्टी की उर्वरता बढ़ती है और पर्यावरण को नुकसान नहीं होता।",
    "पोटाश उर्वरक फलों और सब्जियों की गुणवत्ता सुधारने में मदद करता है।",
    "मिट्टी परीक्षण हर 2-3 साल में करने की सलाह दी जाती है।",
    # Irrigation
    "ड्रिप सिंचाई से 30-60% तक पानी की बचत होती है।",
    "स्प्रिंकलर सिंचाई गेहूं और सब्जियों के लिए उपयुक्त है।",
    "बाढ़ सिंचाई में पानी की बर्बादी अधिक होती है।",
    # Seasons
    "भारत में खरीफ फसलें जून-जुलाई में बोई जाती हैं जैसे धान, मक्का, सोयाबीन।",
    "रबी फसलें अक्टूबर-नवंबर में बोई जाती हैं जैसे गेहूं, सरसों, चना।",
    "जायद फसलें मार्च-अप्रैल में उगाई जाती हैं जैसे तरबूज और खरबूजा।",
    # Pest control
    "नीम आधारित कीटनाशक जैविक खेती में उपयोग किया जाता है।",
    "फसल चक्र अपनाने से कीट और रोगों का प्रकोप कम होता है।",
    "एकीकृत कीट प्रबंधन में रासायनिक और जैविक दोनों तरीके उपयोग होते हैं।",
    # Government schemes
    "PM-KISAN योजना में किसानों को सालाना 6000 रुपये मिलते हैं।",
    "फसल बीमा योजना से किसानों को प्राकृतिक आपदा में मुआवजा मिलता है।",
    "MSP यानी न्यूनतम समर्थन मूल्य सरकार द्वारा तय किया जाता है।",
    "किसान क्रेडिट कार्ड से किसान कम ब्याज पर कृषि ऋण ले सकते हैं।",
    # Soil
    "काली मिट्टी कपास की खेती के लिए सबसे उपयुक्त होती है।",
    "लाल मिट्टी में लोहे की मात्रा अधिक होती है।",
    "दोमट मिट्टी अधिकांश फसलों के लिए उपयुक्त मानी जाती है।",
    "मिट्टी का pH 6-7 के बीच अधिकांश फसलों के लिए आदर्श होता है।",
    # Storage and market
    "अनाज भंडारण में नमी 14% से कम होनी चाहिए।",
    "ई-नाम पोर्टल पर किसान अपनी फसल ऑनलाइन बेच सकते हैं।",
]

# Optional PDF loading
try:
    documents += load_pdf("agriculture.pdf")
except:
    pass

print(f"✅ Total documents: {len(documents)}")

# EMBEDDINGS
doc_emb = model.encode(documents)
print("✅ Embeddings ready")

# GEMINI
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# RAG FUNCTION WITH THRESHOLD
def answer(question):
    q_emb = model.encode(question)
    scores = util.cos_sim(q_emb, doc_emb)[0]
    top_k = scores.topk(5)

    # Filter by similarity threshold
    context_list = []
    for idx, score in zip(top_k.indices, top_k.values):
        if score > 0.35:
            context_list.append(documents[int(idx)])

    if not context_list:
        return "इस विषय पर मेरे पास जानकारी नहीं है।"

    context = "\n".join(context_list)
    print(f"✅ {len(context_list)} passages used")
    print("----- CONTEXT -----")
    print(context)
    print("-------------------")

    prompt = f"""
आप एक कृषि सहायक हैं। नीचे दिए गए संदर्भ के आधार पर प्रश्न का उत्तर दें।
यदि संदर्भ में जानकारी नहीं है तो कहें: "इस विषय पर जानकारी उपलब्ध नहीं है।"

संदर्भ:
{context}

प्रश्न: {question}

उत्तर:
"""
    response = gemini_model.generate_content(prompt)
    return response.text

# GRADIO UI
gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="अपना सवाल हिंदी में लिखें"),
    outputs=gr.Textbox(label="जवाब"),
    title="🌾 RAG Agriculture Assistant",
    description="Hindi Agriculture QA using RAG + Gemini"
).launch()