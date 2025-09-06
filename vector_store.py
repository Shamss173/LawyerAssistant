from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import json

class VectorStore:
    def __init__(self, data_path="data/legal_cases.json"):
        # ✅ Use CaseLawBERT
        self.tokenizer = AutoTokenizer.from_pretrained("zlucia/custom-legalbert")
        self.model = AutoModel.from_pretrained("zlucia/custom-legalbert")

        with open(data_path, "r", encoding="utf-8") as f:
            self.cases = json.load(f)

        # Encode dataset
        self.embeddings = np.array([self.embed_text(c["summary"]) for c in self.cases])

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def embed_text(self, text: str):
        """Convert text into embedding vector using CaseLawBERT"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def search(self, query: str, top_k: int = 3):
        """Find top-k similar cases"""
        q_vec = np.array([self.embed_text(query)])
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.cases[idx])
        return results
