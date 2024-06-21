# Name: bertscore_calculate.py
# Author: Katja Kamyshanova
# Date: 21/06/2024
# This program takes a .json file with four summaries of a document, 
# loads the LLM comparison summaries of the same document, calculates the
# BertScore F1 per input summary, saves the scores in the eva_data file and
# returns mean F1 scores per rank of LLM summary and per LLM as a .json files

# The BertScore F1 code was partially adapted from https://medium.com/@abonia/bertscore-explained-in-5-minutes-0b98553bfb71

# Usage: python3 bertscore_calculate.py <doc number> <short/long> <prompt type that must be ignored in the F1 calculation if exists>
# Example usage: python3 bertscore_calculate.py 7 long "Few-shot Expl"


from bert_score import BERTScorer
from sys import argv
import json
import os
import re
import numpy as np
from statistics import fmean


def locate_scores(scores_dict, label, prompt, f1):

    '''
    Receives a dictionary, a lable the f1 scores correspond to,
    the current running prompt and f1 score, places the values
    into a dictionary and returns them in a changed dictionary
    '''

    # put the scores in the dictionary
    scores_dict[label] = scores_dict.get(label, dict())
    scores_dict[label][prompt] = scores_dict[label].get(prompt, dict())
    scores_dict[label][prompt]['BertScore F1'] = scores_dict[label][prompt].get('BertScore F1', [])
    scores_dict[label][prompt]['BertScore F1'].append(float(f"{f1.mean()}"))

    # calculate the mean F1 if all 10 scores are calculated
    if len(scores_dict[label][prompt]['BertScore F1']) == 10:
        scores = scores_dict[label][prompt]['BertScore F1']
        scores_dict[label][prompt]['BertScore F1 mean'] = f"{fmean(scores):.4f}"

    return scores_dict


def preprocess(text):
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\t +', '', text)

    return text

def main(argv):

    # create the path to the necessary files
    doc_path = str(os.getcwd()) + f'/llm_summaries/eval_set'
    eval_path = str(os.getcwd()) + f'/results_eval'
    scores_path = str(os.getcwd()) + f'/results_eval/scores'

    with open(f"{doc_path}/doc{argv[1]}.txt") as file, open(f"{eval_path}/eval_data_{argv[2]}.json") as file1:
        data_list = file.read().split("<split>")
        eval_data = json.load(file1)

    # create a list of LLMs in order they are set in the text file
    llm_list = ["GPT-3.5", "Copilot", "Gemini"]

    # load the file with previous Bertscores
    if os.path.exists(f"{scores_path}/rank_f1_scores_{argv[2]}.json") and \
       os.path.exists(f"{scores_path}/llm_f1_scores_{argv[2]}.json") and argv[1] != "1":

        with open(f"{scores_path}/rank_f1_scores_{argv[2]}.json") as f, open(f"{scores_path}/llm_f1_scores_{argv[2]}.json") as f1:
            scores_rank_dict = json.load(f)
            scores_llm_dict = json.load(f1)

    # or create the dictionaries for the Bertscores
    else:
        scores_rank_dict = dict()
        scores_llm_dict = dict()

    # load the BERTscore
    scorer = BERTScorer(model_type='bert-base-uncased')

    for prompt, summary in eval_data[argv[1]]["Phi-3"].items():
    
        # assign a prompt type to skip in the calculations (due to invalid summary)
        if len(argv) > 3:
            prompt_skip = argv[3]
            # erase the invalid summary text if exists
            if prompt == prompt_skip:
                    summary_new = ""
            else:
                summary_new = preprocess(summary)
        else:
            summary_new = preprocess(summary)

        # calcule scores per LLM
        for i, example_text in enumerate(data_list):
            llm_dict = eval_data[argv[1]][llm_list[i]]

            # add the LLM summary in the json file
            llm_dict['Text'] = example_text
            
            example_new = preprocess(example_text)

            #p, r, f1 = scorer.score([phi3_text['results']], [example_text])
            p, r, f1 = scorer.score([summary_new], [example_new])

            # add scores to the dictionary
            llm_dict["BertScore"] = llm_dict.get("BertScore", dict())
            llm_dict["BertScore"][prompt] = llm_dict["BertScore"].get(prompt, dict())
            llm_dict["BertScore"][prompt] = {"Precission" : f"{p.mean():.4f}",
                                                "Recall" : f"{r.mean():.4f}",
                                                "F1-score" : f"{f1.mean():.4f}"}

            # initialize a dictionary with ranks and Bertscores per rank
            llm_rank = str(llm_dict['Rank'])
            scores_rank_dict = locate_scores(scores_rank_dict, llm_rank, prompt, f1)

            # initialize a dictionaries with LLM names and Bertscores per LLM
            scores_llm_dict = locate_scores(scores_llm_dict, llm_list[i], prompt, f1)
                

    with open(f"{eval_path}/eval_data_{argv[2]}.json", 'w') as f1, \
         open(f"{scores_path}/rank_f1_scores_{argv[2]}.json", 'w') as f2, \
         open(f"{scores_path}/llm_f1_scores_{argv[2]}.json", 'w') as f3:

        json.dump(eval_data, f1, indent=3)
        json.dump(scores_rank_dict, f2, indent=4)
        json.dump(scores_llm_dict, f3, indent=4)


if __name__ == "__main__":
    main(argv)
