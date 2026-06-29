import numpy as np
from sentence_transformers import util

from config import encoder
from data import base_documents
from llm import gemini_llm, local_llm
import pdf_loader


# Pre-compute embeddings for base knowledge
base_embeddings = encoder.encode(
    base_documents,
    normalize_embeddings=True
)

print(f"Base documents loaded: {len(base_documents)}")


def answer(question, model_choice):
    """
    Main RAG pipeline.
    """

    # -----------------------------
    # Combine documents
    # -----------------------------
    all_documents = list(base_documents)

    if pdf_loader.pdf_documents:
        all_documents.extend(pdf_loader.pdf_documents)

    # -----------------------------
    # Combine embeddings
    # -----------------------------
    if pdf_loader.pdf_embeddings is not None:

        all_embeddings = np.vstack([
            base_embeddings,
            pdf_loader.pdf_embeddings
        ])

    else:

        all_embeddings = base_embeddings

    # -----------------------------
    # Encode question
    # -----------------------------
    question_embedding = encoder.encode(
        question,
        normalize_embeddings=True
    )

    # -----------------------------
    # Similarity Search
    # -----------------------------
    similarity_scores = util.cos_sim(
        question_embedding,
        all_embeddings
    )[0]

    top_results = similarity_scores.topk(5)

    context_chunks = []
    source_list = []

    print("\n========== Retrieved Chunks ==========")

    for index, score in zip(
        top_results.indices,
        top_results.values
    ):

        index = int(index)
        score = float(score)

        source = (
            "PDF"
            if index >= len(base_documents)
            else "Base Knowledge"
        )

        print(f"\nSource : {source}")
        print(f"Score  : {score:.4f}")
        print(all_documents[index])

        if score > 0.30:

            context_chunks.append(all_documents[index])

            source_list.append(
                f"{source} (Score: {score:.2f})"
            )

    print("======================================\n")

    # -----------------------------
    # No context found
    # -----------------------------
    if not context_chunks:

        return "इस विषय पर जानकारी उपलब्ध नहीं है।"

    # -----------------------------
    # Build Context
    # -----------------------------
    context = "\n".join(context_chunks)
    context = context[:3000]

    print(f"Using {len(context_chunks)} retrieved passages")

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
आप एक कृषि सहायक हैं।

केवल नीचे दिए गए संदर्भ के आधार पर उत्तर दें।

नियम:
1. उत्तर केवल संदर्भ से दें।
2. बाहरी जानकारी का उपयोग न करें।
3. यदि उत्तर संदर्भ में नहीं है तो लिखें:
   "इस विषय पर जानकारी उपलब्ध नहीं है।"
4. उत्तर सरल हिन्दी में दें।
5. सभी तथ्य और संख्याएँ सही रखें।

संदर्भ:
{context}

प्रश्न:
{question}

उत्तर:
"""

    # -----------------------------
    # Generate Answer
    # -----------------------------
    try:

        if model_choice == "Gemini API":
            answer_text = gemini_llm(prompt)
        else:
            answer_text = local_llm(prompt)

    except Exception as e:

        print(f"Generation Error: {e}")

        answer_text = local_llm(prompt)

    # -----------------------------
    # Add Sources
    # -----------------------------
    sources_text = "\n\nSources:\n"

    for source in source_list:
        sources_text += f"• {source}\n"

    return answer_text + sources_text