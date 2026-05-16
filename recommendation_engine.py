import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            self.model = xgb.Booster()
            self.model.load_model('ai_models/career_model.json')
        except:
            self.model = None

    def analyze_profile(self, profile: dict):
        text = f"{profile.get('research_areas', '')} {profile.get('publications', '')} {profile.get('teaching_subjects', '')}"
        embedding = self.embedder.encode([text])[0]

        research_score = min(100, len(profile.get('publications', '').split(',')) * 8 + profile.get('citations', 0) / 10)
        teaching_score = profile.get('teaching_feedback', 0) * 20
        service_score = profile.get('admin_roles', 0) * 15

        overall = (research_score * 0.5) + (teaching_score * 0.3) + (service_score * 0.2)

        return {
            "research_score": round(research_score, 2),
            "teaching_score": round(teaching_score, 2),
            "service_score": round(service_score, 2),
            "overall_score": round(overall, 2),
            "strength": "Research" if research_score > teaching_score else "Teaching"
        }

    def predict_promotion(self, features: dict):
        if not self.model:
            return 65.0  # fallback
        df = pd.DataFrame([features])
        dmatrix = xgb.DMatrix(df)
        prob = self.model.predict(dmatrix)[0] * 100
        return round(float(prob), 2)

    def get_recommendations(self, profile: dict):
        return [
            {"type": "Journal", "title": "Submit to Scopus Q1 Journal in your domain", "deadline": "Within 3 months"},
            {"type": "Conference", "title": "Present paper at IEEE/Elsevier International Conference", "deadline": "Next 6 months"},
            {"type": "Skill", "title": "Complete AI/ML or Data Analytics Certification", "deadline": "2 months"}
        ]
