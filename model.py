import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import pandas as pd
import torch
from transformers import AutoTokenizer, LongformerModel


model_name = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder   = LongformerModel.from_pretrained(model_name)
encoder.eval()  

def get_text_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        outputs = encoder(**inputs)
    embedding = outputs.pooler_output 
    return embedding.squeeze(0) 


def create_dataset(congressman_file, bills_location= None):
    congressman = pd.read_csv(congressman_file)
    for index, row in congressman.iterrows():
        print(f"{row['Congress']} {row['Bill Type']} {row['Bill Number']}")
    
    
# Example usage
bills = pd.read_csv("data_collection/bills/106.csv")
create_dataset('data_collection/congressman_data/Adam_Schiff_CA_Democrat_sen_S001150_S427.csv')
#print(get_text_embedding(bills.iloc[0]['text']))
#print(len(bills))