import re
import fitz

from config import encoder

# Global storage for uploaded PDF knowledge
pdf_documents = []
pdf_embeddings = None


def load_pdf(pdf_file):
    """
    Load a PDF, extract text using PyMuPDF,
    split into chunks, generate embeddings,
    and store them globally.
    """

    global pdf_documents
    global pdf_embeddings

    texts = []

    try:
        # Open PDF
        doc = fitz.open(pdf_file.name)

        for page in doc:

            # Extract text
            text = page.get_text("text")

            if not text:
                continue

            # -----------------------------
            # Clean extracted text
            # -----------------------------
            clean_text = text.replace("\n", " ")
            clean_text = " ".join(clean_text.split())

            # -----------------------------
            # Split into sentences
            # -----------------------------
            sentences = re.split(
                r"(?<=[।.])\s+",
                clean_text
            )

            sentences = [
                s.strip()
                for s in sentences
                if len(s.strip()) > 40
            ]

            # -----------------------------
            # Create chunks (~400 chars)
            # -----------------------------
            current_chunk = ""

            for sentence in sentences:

                if len(current_chunk) + len(sentence) <= 400:

                    current_chunk += " " + sentence

                else:

                    if len(current_chunk.strip()) > 80:
                        texts.append(current_chunk.strip())

                    current_chunk = sentence

            # Add remaining chunk

            if len(current_chunk.strip()) > 80:
                texts.append(current_chunk.strip())

        doc.close()

        # -----------------------------
        # No text extracted
        # -----------------------------

        if not texts:
            return "No text could be extracted from the PDF."

        # -----------------------------
        # Store documents
        # -----------------------------

        pdf_documents = texts

        pdf_embeddings = encoder.encode(
            texts,
            normalize_embeddings=True
        )

        print(f"\n===== PDF CHUNKS ({len(texts)} total) =====")

        for i, chunk in enumerate(texts[:10]):
            print(f"\n[Chunk {i + 1}]")
            print(chunk)

        print("=============================\n")

        return f"PDF loaded successfully. {len(texts)} passages indexed."

    except Exception as e:

        return f"Error loading PDF: {str(e)}"