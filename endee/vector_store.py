import numpy as np

class EndeeVectorStore:
    def __init__(self):
        self.content_vectors = []
        self.skill_vectors = []
        self.metadata = []

    def add(self, content_embedding, skill_embedding, meta):
        self.content_vectors.append(np.array(content_embedding))
        self.skill_vectors.append(np.array(skill_embedding))
        self.metadata.append(meta)

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def query_content(self, query_embedding):
        return [
            (self.cosine_similarity(query_embedding, vec), meta)
            for vec, meta in zip(self.content_vectors, self.metadata)
        ]

    def query_skill(self, query_embedding):
        return [
            (self.cosine_similarity(query_embedding, vec), meta)
            for vec, meta in zip(self.skill_vectors, self.metadata)
        ]
