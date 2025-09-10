from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval() 
    
    def analyze(self, text):
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.5, "score": -1}
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment = torch.argmax(probs, dim=1).item() 
        confidence = probs[0][sentiment].item()
        
        return {
            "sentiment": "positive" if sentiment == 1 else "negative",
            "confidence": confidence,
            "score": sentiment
        }