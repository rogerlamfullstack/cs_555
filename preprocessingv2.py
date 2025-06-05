from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Disable dropout for inference

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
dataset = load_dataset(
    "daily_dialog",
    cache_dir="./hf_cache_clean"        # use a new local cache folder
)
print(dataset)

test_dialogues = dataset['test']['dialog']
true_labels_test = dataset['test']['emotion']  # Ground truth labels (emotion)
# Prepare BERT embeddings
batch_size = 32
utterance_embeddings = []
dialogues = dataset['train']['dialog']
all_original_utterances = []
for dialogue_list in dialogues:
    all_original_utterances.extend(dialogue_list)
with torch.no_grad():
    for i in tqdm(range(0, len(all_original_utterances), batch_size)):
        batch_texts = all_original_utterances[i:i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        outputs = model(**encoded_input)
        last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
        
        # Mean pooling (exclude padding tokens)
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        utterance_embeddings.extend(mean_pooled.cpu().numpy())

X = np.array(utterance_embeddings)
