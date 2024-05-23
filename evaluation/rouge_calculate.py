# The code is adapted from https://medium.com/@eren9677/text-summarization-387836c9e178
#
#
#


import evaluate
from sys import argv
import json
import re
import os

def preprocess(text):
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\n{2,}', '\n', text)

    return text.split("\\n")

def main(argv):
    with open(f"doc{argv[1]}.txt") as f, open(f"results_eval_{argv[1]}.json") as f1:
        data_list = f.read().split("<split>")
        phi3_text = json.load(f1)

    phi3_list = preprocess(phi3_text['results'])
    
    # creating a list of LLMs in order they are set in the text file
    llm_list = ["GPT3.5", "Copilot", "Gemini"]
    llm_scores_dict = dict()

    rouge = evaluate.load('rouge')
    for i, text in enumerate(data_list):
        sent_list = preprocess(text)
        scores = rouge.compute(predictions=phi3_list,
                               references=sent_list)
        llm_scores_dict[llm_list[i]] = scores

    file_name = "rouge_scores.json"
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
