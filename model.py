import torch.nn as nn
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification
import pandas as pd
from transformers import AutoTokenizer, LongformerModel
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.utils.data import Dataset
import random
import ast
import newbilldata

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

def import_dataset(file='test.csv'):
    df = pd.read_csv(file)
    dataset = []
    for i, row in df.iterrows():
        embedding_list = ast.literal_eval(row["features"][len("tensor("):].rstrip(")"))
        embedding_tensor = torch.tensor(embedding_list, dtype=torch.float)
        label_str = row["label"]
        label_val = 0.0 if label_str == "Yea" else 1.0
        dataset.append((embedding_tensor, label_val)) 
    X_list = []
    Y_list = []
    for feats, label_str in dataset:
        label_val = label_str
    
        X_list.append(feats)         
        Y_list.append(label_str)     
    X = torch.stack(X_list)        
    Y = torch.tensor(Y_list).unsqueeze(1)      
                
    return X, Y
    


def create_dataset(congressman_file, bills_location= "data_collection/bills"):
    congressman = pd.read_csv(congressman_file).dropna()
    dataset = set()
    length = len(congressman)
    print(f"Processing {length} entries.")
    for congress in range (102,120):
        bills = pd.read_csv(f"{bills_location}/{congress}.csv")
        bills_index = dict()
        
        for index, row in bills.iterrows():
            bills_index[(congress, row['bill_type'], int(row['bill_number']))] = row['text']
            
        for index, row in congressman.iterrows():
            if row['Congress'] == congress and (congress, row['Bill Type'], int(row['Bill Number'])) in bills_index:
                if type(bills_index[(congress, row['Bill Type'], int(row['Bill Number']))]) != str:
                    continue 
                dataset.add((get_text_embedding(bills_index[(congress, row['Bill Type'], int(row['Bill Number']))]), row['Vote Position']))
                
                print(f"Procesed entry {len(dataset)} of {length}. Done with {len(dataset)/length*100:.2f}%")
    
    
    
    
    data_list = list(dataset)
    df = pd.DataFrame(data_list, columns=['features', 'label'])
    df.to_csv(f"test.csv", index=False)
    X_list = []
    Y_list = []
    for feats, label_str in data_list:
        label_val = 1.0 if label_str == "Yea" else 0.0
    
        X_list.append(feats)        
        Y_list.append(label_val)

    X = torch.stack(X_list)                 
    Y = torch.tensor(Y_list).unsqueeze(1)     
                
    return X, Y
        


def train_vote_classifier(congressman_file):
    #X,Y = create_dataset(congressman_file)
    X,Y = import_dataset()
    total_samples = len(Y)

    
    if total_samples < 10:
        print(f"Not enough data for member {congressman_file}. Found only {total_samples} samples.")
        return None
    
    indices = list(range(total_samples))
    random.shuffle(indices)
    train_size = int(0.8 * total_samples) 
    train_indices = indices[:train_size]
    test_indices  = indices[train_size:]

    X_train = X[train_indices]  
    Y_train = Y[train_indices] 

    X_test  = X[test_indices]
    Y_test  = Y[test_indices]
    
    
    
    model = nn.Sequential(
        nn.Linear(768, 128),
        nn.ReLU(),
        nn.Linear(128, 1)   
    )
    
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_train.size(0))
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        logits = model(X_train)
        loss   = criterion(logits, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs  = torch.sigmoid(logits).squeeze(1)
        preds  = (probs >= 0.5).float()
        correct = (preds == Y_test.squeeze(1)).sum().item()
        total   = Y_test.size(0)
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.4f}")
    return model
    
def predict_on_bill(model, congress, bill_type, bill_number):
    model.eval()
    text = newbilldata.get_bill_text(congress, bill_type, bill_number, 1)
    if text is None or type(text) != str:
        return None
    embedding = get_text_embedding(text)
    with torch.no_grad():
        logits = model(embedding.unsqueeze(0))
        probs  = torch.sigmoid(logits).squeeze(1)
        pred   = (probs >= 0.5)
        
        return "Yea" if pred == False else "Nay"
    
    
    
bills = pd.read_csv("data_collection/bills/106.csv")
#data = create_dataset('data_collection/congressman_data/Adam_Schiff_CA_Democrat_sen_S001150_S427.csv')

model = train_vote_classifier('data_collection/congressman_data/Scott_Peters_CA_Democrat_rep_P000608_None.csv')
torch.save(model.state_dict(), "test.pth")

#print(predict_on_bill(model, 118, 'hr', 2))
#print(predict_on_bill(model, 118, 'hr', 3442))


#data_collection/congressman_data/Scott_Peters_CA_Democrat_rep_P000608_None.csv
#print(get_text_embedding(bills.iloc[0]['text']))
#print(len(bills))