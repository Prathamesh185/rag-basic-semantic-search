import os
import gradio as gr
import pdfplumber
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import numpy as np

# =========================
# ENV SETUP
# =========================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# =========================
# MODELS
# =========================
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# LOCAL LLM FUNCTION
# =========================
def local_llm(prompt):
    response = ollama.chat(
        model='qwen3.5:4b',
        think=False,
        options={
            "num_predict": 150,
            "temperature": 0.1,  # lower = more focused on context
            "num_thread": 8
        },
        messages=[
            {
                'role': 'system',
                'content': 'तुम एक कृषि सहायक हो। केवल दिए गए संदर्भ से उत्तर दो। संदर्भ के बाहर कुछ मत बताओ।'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )
    
    raw = response['message']['content']
    if '</think>' in raw:
        raw = raw.split('</think>')[-1].strip()
    
    return raw if raw.strip() else "इस विषय पर जानकारी उपलब्ध नहीं है।"


# =========================
# GEMINI FUNCTION
# =========================
def gemini_llm(prompt):

    response = gemini_model.generate_content(prompt)
    return response.text


# =========================
# BASE AGRICULTURE DOCUMENTS
# =========================
base_documents = [

    "धान (Rice) की IR-64 किस्म भारत में उत्तर प्रदेश और छत्तीसगढ़ में अधिक उगाई जाती है।",
    "गेहूं की HD-2967 किस्म पंजाब और हरियाणा में लोकप्रिय है और 140-145 दिनों में तैयार होती है।",
    "कपास की Bt Cotton किस्म भारत में 90% से अधिक क्षेत्र में उपयोग होती है।",
    "टमाटर की Pusa Ruby किस्म भारत में सर्दी और गर्मी दोनों मौसम में उगाई जा सकती है।",
    "मक्का की Hybrid किस्में 80-85 दिनों में तैयार होती हैं।",
    "गन्ने की Co-0238 किस्म महाराष्ट्र में अधिक उत्पादक है।",
    "यूरिया में 46% नाइट्रोजन होता है और यह सबसे अधिक उपयोग होने वाला उर्वरक है।",
    "DAP खाद में 18% नाइट्रोजन और 46% फॉस्फोरस होता है।",
    "जैविक खाद से मिट्टी की उर्वरता बढ़ती है।",
    "पोटाश उर्वरक फलों और सब्जियों की गुणवत्ता सुधारने में मदद करता है।",
    "मिट्टी परीक्षण हर 2-3 साल में करने की सलाह दी जाती है।",
    "ड्रिप सिंचाई से 30-60% तक पानी की बचत होती है।",
    "स्प्रिंकलर सिंचाई गेहूं और सब्जियों के लिए उपयुक्त है।",
    "भारत में खरीफ फसलें जून-जुलाई में बोई जाती हैं जैसे धान, मक्का, सोयाबीन।",
    "रबी फसलें अक्टूबर-नवंबर में बोई जाती हैं जैसे गेहूं, सरसों, चना।",
    "नीम आधारित कीटनाशक जैविक खेती में उपयोग किया जाता है।",
    "फसल चक्र अपनाने से कीट और रोगों का प्रकोप कम होता है।",
    "PM-KISAN योजना में किसानों को सालाना 6000 रुपये मिलते हैं।",
    "फसल बीमा योजना से किसानों को प्राकृतिक आपदा में मुआवजा मिलता है।",
    "MSP यानी न्यूनतम समर्थन मूल्य सरकार द्वारा तय किया जाता है।",
    "किसान क्रेडिट कार्ड से किसान कम ब्याज पर कृषि ऋण ले सकते हैं।",
    "काली मिट्टी कपास की खेती के लिए सबसे उपयुक्त होती है।",
    "दोमट मिट्टी अधिकांश फसलों के लिए उपयुक्त मानी जाती है।",
    "मिट्टी का pH 6-7 के बीच अधिकांश फसलों के लिए आदर्श होता है।",
    "ई-नाम पोर्टल पर किसान अपनी फसल ऑनलाइन बेच सकते हैं।",
]

# =========================
# GLOBAL PDF STORAGE
# =========================
pdf_documents = []
pdf_embeddings = None

# =========================
# ENCODE BASE DOCS
# =========================
base_embeddings = encoder.encode(base_documents)

print(f"✅ Base documents ready: {len(base_documents)}")


# =========================
# PDF LOADER
# =========================
def load_pdf(pdf_file):

    global pdf_documents, pdf_embeddings

    texts = []

    try:

        with pdfplumber.open(pdf_file.name) as pdf:

            for page in pdf.pages:

                text = page.extract_text()

                if text:

                    # Better Hindi chunking
                    chunks = text.replace("\n", " ").split("।")

                    texts.extend([
                        c.strip()
                        for c in chunks
                        if len(c.strip()) > 40
                    ])

        pdf_documents = texts

        pdf_embeddings = encoder.encode(texts)

        return f"✅ PDF loaded: {len(texts)} passages ready"

    except Exception as e:

        return f"❌ Error: {str(e)}"


# =========================
# RAG ANSWER FUNCTION
# =========================
def answer(question, model_choice):

    # Combine documents
    all_docs = base_documents + pdf_documents

    # Combine embeddings
    if pdf_embeddings is not None:

        all_embeddings = np.vstack([
            base_embeddings,
            pdf_embeddings
        ])

    else:

        all_embeddings = base_embeddings

    # Encode question
    q_emb = encoder.encode(question)

    # Similarity search
    scores = util.cos_sim(q_emb, all_embeddings)[0]

    # Retrieve top chunks
    top_k = scores.topk(2)

    # Filter chunks
    context_list = []

    for idx, score in zip(top_k.indices, top_k.values):

        if score > 0.35:

            context_list.append(all_docs[int(idx)])

    # No context found
    if not context_list:

        return "इस विषय पर मेरे पास जानकारी नहीं है।"

    # Create context
    context = "\n".join(context_list)

    # Limit context size
    context = context[:1000]

    print(f"✅ {len(context_list)} passages used")

    # Prompt
    prompt = f"""
        आप केवल दिए गए संदर्भ के आधार पर उत्तर दें।

        महत्वपूर्ण नियम:
        1. अपनी बाहरी जानकारी का उपयोग बिल्कुल न करें।
        2. यदि उत्तर संदर्भ में नहीं है तो केवल लिखें:
        "इस विषय पर जानकारी उपलब्ध नहीं है।"
        3. अनुमान या मनगढ़ंत उत्तर न दें।
        4. उत्तर छोटा और सीधा रखें।

        संदर्भ:
        {context}

        प्रश्न:
        {question}

        उत्तर:
    """

    # Generate answer
    try:

        if model_choice == "Gemini API":

            return gemini_llm(prompt)

        else:

            return local_llm(prompt)

    except Exception as e:

        print(f"⚠️ Error: {str(e)}")

        return local_llm(prompt)


# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="🌾 RAG Agriculture Assistant") as demo:

    gr.Markdown("# 🌾 Hindi Agriculture Knowledge Assistant")

    gr.Markdown(
        "कृषि संबंधी सवाल पूछें — PDF अपलोड करके अपना खुद का ज्ञान आधार जोड़ें"
    )

    with gr.Row():

        # LEFT SIDE
        with gr.Column(scale=1):

            pdf_input = gr.File(
                label="📄 Agriculture PDF अपलोड करें (optional)",
                file_types=[".pdf"]
            )

            pdf_status = gr.Textbox(
                label="PDF Status",
                value="No PDF loaded — using base knowledge",
                interactive=False
            )

            pdf_input.change(
                fn=load_pdf,
                inputs=pdf_input,
                outputs=pdf_status
            )

        # RIGHT SIDE
        with gr.Column(scale=2):

            model_choice = gr.Radio(
                ["Gemini API", "Local Qwen"],
                value="Local Qwen",
                label="Choose LLM"
            )

            question_input = gr.Textbox(
                label="अपना सवाल हिंदी में लिखें",
                placeholder="जैसे: धान की अच्छी किस्म कौन सी है?",
                lines=2
            )

            ask_btn = gr.Button(
                "पूछें 🔍",
                variant="primary"
            )

            answer_output = gr.Textbox(
                label="जवाब",
                lines=5,
                interactive=False
            )

            ask_btn.click(
                fn=answer,
                inputs=[question_input, model_choice],
                outputs=answer_output
            )

# =========================
# LAUNCH APP
# =========================
demo.launch()

# Q- PM-KISAN योजना में कितने रुपये मिलते हैं?
#    Expected: 6000 रुपये सालाना, 2000 प्रति तिमाही

# Q- यूरिया में कितना नाइट्रोजन है?
#    Expected: 46%

# Q- ड्रिप सिंचाई से क्या फायदा है?
#    Expected: 30-60% पानी की बचत