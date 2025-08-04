import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


model_path = "/Users/kannavsethi/Desktop/nlp-final-project/models/checkpoint-36543"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)


short_texts = [
    "Yes.",             
    "I agree.",         
    "Thanks!"           
]

false_positives = [
    "I believe this approach would be beneficial for achieving optimal results.", 
    "Furthermore, we should consider the implications of this decision.",         
    "It is important to note that various factors contribute to this outcome."    
]


false_negatives = [
    "Honestly, I think this whole thing is pretty messed up, you know?",          
    "Ugh, don't even get me started on that topic ",                          
    "My bad! Totally forgot about that meeting yesterday."                        
]


test_cases = {
    "Short texts": short_texts,
    "False positives": false_positives, 
    "False negatives": false_negatives
}

def test_model_predictions(model, tokenizer):
    """
    Simple function to test model on hardcoded examples
    """
    import torch
    
    for category, examples in test_cases.items():
        print(f"\n{category}:")
        for text in examples:
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
            
            
            label = "positive" if predicted_class == 1 else "negative"
            confidence = predictions[0][predicted_class].item()
            
            print(f"'{text}' -> {label} (confidence: {confidence:.3f})")


test_model_predictions(model, tokenizer)
