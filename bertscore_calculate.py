# The code was adapted from https://haticeozbolat17.medium.com/text-summarization-how-to-calculate-bertscore-771a51022964

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sys import argv
import json

def main(argv):
    # Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    #phi3_text = json.load(f"results_eval_{[argv[2]]}.json")

    with open(argv[1]) as f, open(f"results_eval_{[argv[2]]}.json") as f1:
        data_list = f.read().split("<split>")
        phi3_text = json.load(f1)

    for example_text in data_list:

        # Step 4: Prepare the texts for BERT
        inputs1 = tokenizer(phi3_text, return_tensors="pt", padding=True, truncation=True)
        inputs2 = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True)

        # Step 5: Feed the texts to the BERT model
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

        # Step 6: Obtain the representation vectors
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

        # Step 7: Calculate cosine similarity
        similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

        # Step 8: Print the result
        print("Similarity between the texts: {:.4f}".format(similarity[0][0]))

if __name__ == "__main__":
    main(argv)