# The code was adapted from https://medium.com/@abonia/bertscore-explained-in-5-minutes-0b98553bfb71
#
#
#

from bert_score import BERTScorer
#import torch
from sys import argv
import json
import os
import re

def preprocess(text):
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\n{2,}', '\n', text)

    return text.split("\\n")

def main(argv):
    with open(f"doc{argv[1]}.txt") as file, open(f"results_eval_{argv[1]}.json") as file1:
        data_list = file.read().split("<split>")
        phi3_text = json.load(file1)

    phi3_list = preprocess(phi3_text['results'])

    # creating a list of LLMs in order they are set in the text file
    llm_list = ["GPT3.5", "Copilot", "Gemini"]
    llm_scores_dict = dict()

    # loading the BERTscore
    scorer = BERTScorer(model_type='bert-base-uncased')

    # calculating scores per LLM
    for i, example_text in enumerate(data_list):
        sent_list = preprocess(example_text)

        #p, r, f1 = scorer.score([phi3_text['results']], [example_text])
        p, r, f1 = scorer.score(phi3_list, sent_list)

        # adding scores to the dictionary
        llm_scores_dict[llm_list[i]] = {"Precission" : f"{p.mean():.4f}",
                                        "Recall" : f"{r.mean():.4f}",
                                        "F1-score" : f"{f1.mean():.4f}"}

    # adding scores to the json file
    file_name = "bertscores.json"
    if os.path.exists(file_name):
        with open(file_name) as f:
            data = json.load(f)
    else:
        data = dict()

    data[f"Doc {argv[1]}"] = llm_scores_dict

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=3)

if __name__ == "__main__":
    main(argv)
