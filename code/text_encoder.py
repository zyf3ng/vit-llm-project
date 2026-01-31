import pandas as pd
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

def merge(csv_path):
    df = pd.read_csv(csv_path)
    
    df['findings'] = df['findings'].fillna("")
    df['impression'] = df['impression'].fillna("")
    
    def merge_rows(row):
        findings = str(row['findings']).strip()
        impression = str(row['impression']).strip()
    
        if not findings:
            findings = "No findings."
        if not impression:
            impression = "No impression."

        return f"Findings: {findings} Impression: {impression}"

    text_list = df.apply(merge_rows, axis=1).tolist()
    
    return text_list

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_length = 400
        
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        return outputs.last_hidden_state, attention_mask

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'archive', 'indiana_reports.csv')
    texts = merge(csv_path)
    print(texts[0])

    encoder = TextEncoder()
    
    #test
    embeddings, mask = encoder(texts[:2])
    
    print(embeddings.shape) 