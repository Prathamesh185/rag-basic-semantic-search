import os
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ── REPLACE YOUR OLD COMMENTED documents LIST WITH THIS ──

from datasets import load_dataset

def load_documents():
    try:
        # Option 1: IndicQA Hindi — real Hindi Q&A passages
        dataset = load_dataset("ai4bharat/IndicQA", "hi", split="test")
        docs = [item["context"] for item in dataset if len(item["context"]) > 80]
        docs = list(set(docs))[:200]  # deduplicate, take 200
        print(f"✅ Loaded {len(docs)} passages from IndicQA Hindi")
        return docs

    except Exception as e:
        print(f"⚠️ IndicQA failed: {e}")

        try:
            # Option 2: Hindi Wikipedia subset via HuggingFace
            dataset = load_dataset("wikimedia/wikipedia", "20231101.hi", 
                                   split="train[:300]", trust_remote_code=True)
            docs = [item["text"][:500] for item in dataset 
                    if len(item["text"]) > 100]
            print(f"✅ Loaded {len(docs)} passages from Hindi Wikipedia")
            return docs

        except Exception as e2:
            print(f"⚠️ Wikipedia failed: {e2}")

            # Option 3: Hardcoded fallback — better than your original 10
            print("⚠️ Using fallback passages")
            return [
                "धान (Rice) की IR-64 किस्म भारत में उत्तर प्रदेश और छत्तीसगढ़ में अधिक उगाई जाती है और इसका औसत उत्पादन 4–5 टन प्रति हेक्टेयर होता है।",
                "गेहूं की HD-2967 किस्म पंजाब और हरियाणा में लोकप्रिय है और यह 140–145 दिनों में तैयार होती है।",
                "कपास की Bt Cotton किस्म भारत में 90% से अधिक क्षेत्र में उपयोग होती है क्योंकि यह bollworm resistance देती है।",
                "गन्ने की Co-0238 किस्म महाराष्ट्र में अधिक उत्पादक है और इसका औसत उत्पादन 100–120 टन प्रति हेक्टेयर होता है।",
                "टमाटर की Pusa Ruby किस्म भारत में सर्दी और गर्मी दोनों मौसम में उगाई जा सकती है।",
                "भारत में यूरिया (Urea) में 46% नाइट्रोजन होता है और यह सबसे अधिक उपयोग होने वाला उर्वरक है।",
                "मिट्टी परीक्षण (Soil Testing) हर 2–3 साल में करने की सलाह कृषि विश्वविद्यालय देते हैं ताकि NPK संतुलन बनाए रखा जा सके।",
                "ड्रिप सिंचाई से भारत में 30–60% तक पानी की बचत की जा सकती है, खासकर महाराष्ट्र और कर्नाटक में।",
                "जैविक खेती में नीम आधारित कीटनाशक (Neem Oil) का उपयोग भारत में प्राकृतिक pest control के लिए किया जाता है।",
                "भारत में खरीफ फसलें जून-जुलाई में मानसून पर निर्भर होती हैं और इनमें धान, मक्का और सोयाबीन प्रमुख हैं।",
                "रबी फसलें अक्टूबर-नवंबर में बोई जाती हैं और इनमें गेहूं, सरसों और चना प्रमुख हैं।",
                "भारत में DAP (Diammonium Phosphate) उर्वरक में 18% नाइट्रोजन और 46% फॉस्फोरस होता है।",
                "सोयाबीन की JS-335 किस्म मध्य प्रदेश और महाराष्ट्र में व्यापक रूप से उगाई जाती है।",
                "मक्का की संकर किस्में भारत में 6–8 टन प्रति हेक्टेयर उत्पादन देती हैं।",
                "प्रधानमंत्री फसल बीमा योजना (PMFBY) किसानों को प्राकृतिक आपदाओं से होने वाले नुकसान की भरपाई करती है।",
                "भारत में न्यूनतम समर्थन मूल्य (MSP) सरकार द्वारा किसानों को उनकी फसलों का उचित मूल्य सुनिश्चित करने के लिए दिया जाता है।",
                "जल संरक्षण के लिए बूंद-बूंद सिंचाई और स्प्रिंकलर सिंचाई तकनीक का उपयोग बढ़ रहा है।",
                "कृषि में जैव उर्वरक जैसे राइजोबियम और अज़ोटोबैक्टर का उपयोग मिट्टी की उर्वरता बढ़ाता है।",
                "भारत में हरित क्रांति के बाद गेहूं और धान के उत्पादन में भारी वृद्धि हुई।",
                "सटीक कृषि (Precision Farming) में GPS और IoT तकनीक का उपयोग फसल उत्पादन बढ़ाने के लिए किया जाता है।"
            ]

documents = load_documents()

# ── REST OF YOUR CODE UNCHANGED ──

doc_emb = model.encode(documents, show_progress_bar=True)
print(f"✅ Encoded {len(doc_emb)} documents")

def answer(question):
    q_emb = model.encode(question)
    scores = util.cos_sim(q_emb, doc_emb)
    top2 = scores[0].topk(2)

    context = ""
    for idx in top2.indices:
        context += documents[int(idx)] + "\n"

    prompt = f"""
        नीचे दिए गए संदर्भ के आधार पर उत्तर दें।
        यदि संदर्भ आंशिक है, तो उपलब्ध जानकारी के आधार पर उत्तर दें।

        संदर्भ:
        {context}

        प्रश्न: {question}

        उत्तर:
    """

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