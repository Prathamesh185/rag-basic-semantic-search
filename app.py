import gradio as gr
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
documents = [
    "मधुमेह में चीनी और मीठे पदार्थों का सेवन कम करना चाहिए।",
    "मधुमेह के रोगियों को नियमित रूप से रक्त शर्करा की जांच करनी चाहिए।",
    "बुखार में शरीर को हाइड्रेटेड रखने के लिए पानी और तरल पदार्थ अधिक पीने चाहिए।",
    "बुखार में पेरासिटामोल डॉक्टर की सलाह से ली जा सकती है।",
    "हृदय को स्वस्थ रखने के लिए रोज 30 मिनट व्यायाम करना चाहिए।",
    "हृदय रोग से बचने के लिए तेल और वसायुक्त भोजन कम करें।",
    "रात को 7 से 8 घंटे की नींद लेना शरीर के लिए जरूरी है।",
    "नींद पूरी न होने से तनाव और चिड़चिड़ापन बढ़ता है।",
    "उच्च रक्तचाप में नमक का सेवन कम करना चाहिए।",
    "उच्च रक्तचाप के रोगियों को नियमित दवाई लेनी चाहिए।"
]
doc_emb = model.encode(documents)

def answer(question):
    q_emb = model.encode(question)
    scores = util.cos_sim(q_emb, doc_emb)
    
    # Show TOP 2 results instead of just 1
    top2 = scores[0].topk(2)
    result = ""
    for i, (score, idx) in enumerate(zip(top2.values, top2.indices)):
        result += f"Result {i+1} (confidence: {score:.2f}):\n{documents[idx]}\n\n"
    return result

gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="अपना सवाल हिंदी में लिखें"),
    outputs=gr.Textbox(label="जवाब"),
    title="Simple semantic search engine",
    description="स्वास्थ्य संबंधी सवाल पूछें"
).launch()


# Questions to ask
# मधुमेह (Diabetes)
# मधुमेह में क्या खाना चाहिए?
# मधुमेह के मरीजों को मीठा क्यों नहीं खाना चाहिए?
# ब्लड शुगर कंट्रोल कैसे करें?

# बुखार (Fever)
# बुखार होने पर क्या करना चाहिए?
# बुखार में कौन सी दवा ली जा सकती है?
# बुखार में शरीर को कैसे हाइड्रेट रखें?

# हृदय स्वास्थ्य (Heart Health)
# दिल को स्वस्थ कैसे रखें?
# हृदय रोग से बचने के उपाय क्या हैं?
# क्या रोज़ व्यायाम जरूरी है?

# नींद (Sleep)
# अच्छी नींद के लिए क्या करना चाहिए?
# नींद पूरी न होने के नुकसान क्या हैं?
# कितने घंटे की नींद जरूरी है?

# उच्च रक्तचाप (High BP)
# हाई बीपी में क्या नहीं खाना चाहिए?
# उच्च रक्तचाप कैसे नियंत्रित करें?
# क्या नमक कम करना जरूरी है?