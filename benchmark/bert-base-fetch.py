import logging

from transformers import BertTokenizer, BertModel

logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

with open('mock_data.txt') as f:
    text = f.read().split("\n")

encoded_input = tokenizer(text, return_tensors='pt', padding=True)

print(encoded_input)
print(tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][0]))
print(encoded_input["input_ids"].size())
