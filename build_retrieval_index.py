
"""
Part 3 retrieval/index builder.

Builds sentence-wise policy chunks, embeddings with
all-MiniLM-L6-v2, and a Faiss IndexFlatIP index.
"""

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


REPO_DIR = Path(__file__).resolve().parent
KB_PATH = REPO_DIR / "policy_knowledge_base.json"


def load_policy_documents():
    with open(
        KB_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def build_policy_chunks(policy_documents):

    chunks = []

    for doc in policy_documents:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            doc["text"].strip()
        )

        for i, sentence in enumerate(
            sentences,
            start=1
        ):

            sentence = sentence.strip()

            if sentence:
                chunks.append({
                    "chunk_id":
                        f"{doc['doc_id']}_CH{i}",
                    "parent_doc_id":
                        doc["doc_id"],
                    "parent_title":
                        doc["title"],
                    "text":
                        sentence,
                })

    return chunks


def build_faiss_index():

    documents = load_policy_documents()

    chunks = build_policy_chunks(
        documents
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        [
            chunk["text"]
            for chunk in chunks
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return (
        chunks,
        model,
        embeddings,
        index,
    )


if __name__ == "__main__":

    chunks, model, embeddings, index = (
        build_faiss_index()
    )

    print("POLICY RETRIEVAL BUILD")
    print(
        "Documents:",
        len(load_policy_documents())
    )
    print(
        "Chunks:",
        len(chunks)
    )
    print(
        "Embedding shape:",
        embeddings.shape
    )
    print(
        "Vectors in Faiss:",
        index.ntotal
    )
