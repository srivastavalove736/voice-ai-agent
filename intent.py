import json
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agent.schemas import IntentResult


class IntentClassifier:
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, intents_path: Path) -> None:
        self.intents_path = intents_path
        self.model: Pipeline = Pipeline(
            [
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", LogisticRegression(max_iter=200)),
            ]
        )
        self._labels: List[str] = []
        self._train()

    def _load_training_data(self) -> Tuple[List[str], List[str]]:
        with self.intents_path.open("r", encoding="utf-8") as f:
            intent_map = json.load(f)

        texts: List[str] = []
        labels: List[str] = []
        for intent, examples in intent_map.items():
            for sample in examples:
                texts.append(sample.lower().strip())
                labels.append(intent)

        return texts, labels

    def _train(self) -> None:
        texts, labels = self._load_training_data()
        if not texts:
            raise ValueError("No training data found in intents file")

        self._labels = sorted(set(labels))
        self.model.fit(texts, labels)

    def predict(self, text: str) -> List[IntentResult]:
        cleaned = text.lower().strip()
        if not cleaned:
            return [IntentResult(intent="help", confidence=0.0)]

        probabilities = self.model.predict_proba([cleaned])[0]
        classes = self.model.classes_

        results = [
            IntentResult(intent=str(cls), confidence=float(prob))
            for cls, prob in zip(classes, probabilities)
            if prob > self.CONFIDENCE_THRESHOLD
        ]

        if not results:
            top_idx = int(probabilities.argmax())
            return [
                IntentResult(
                    intent=str(classes[top_idx]),
                    confidence=float(probabilities[top_idx]),
                )
            ]

        return sorted(results, key=lambda r: r.confidence, reverse=True)
