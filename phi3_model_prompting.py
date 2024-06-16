#
#
#
#
 
# If used for dev prompts: python3 phi3_model_prompting.py dev <xml/phi3> <v number>
# If used for eval prompts: python3 phi3_model_prompting.py eval <short/long> <doc number> eval_data_<short/long>.json

# Example use: python3 phi3_model_prompting.py eval short 1 eval_data_short.json

import json
import sys
import os
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          # BitsAndBytesConfig,
                          pipeline)

def main(argv):

    # TODO: ADD ARGV CHECK FOR eval_data_<short/long>.json AND short/long
    # TODO: ADD ARGV CHECK FOR xml AND v number 1,2,3,3_5 SO THEY MATCH

    # define the model
    model_phi = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        # device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # define the tokenizer
    tokenizer_phi = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    # define the pipeline for the prompt
    pipe_phi = pipeline(
        "text-generation",
        model=model_phi,
        tokenizer=tokenizer_phi,
    )

    # define the arguments for the output
    generation_args = {
        "max_new_tokens": 1000,
        "return_full_text": False,
        "temperature": 0.2,
        "do_sample": False,
    }

    # loading the prompts
    if argv[1] == 'dev':
        dev_path = str(os.getcwd()) + f'/prompt_dev/v{argv[3]}'
        if argv[2] == 'phi3':
            with open(f'{dev_path}/zeroshot_phi3_v{argv[3]}.txt') as f1,\
                open(f'{dev_path}/oneshot_phi3_v{argv[3]}.txt') as f2, \
                open(f'{dev_path}/fewshot_rank_phi3_v{argv[3]}.txt') as f3, \
                open(f'{dev_path}/fewshot_expl_phi3_v{argv[3]}.txt') as f4:
                prompt_dict = {
                    'Zero-shot' : f1.read(),
                    'One-shot' : f2.read(),
                    'Few-shot rank' : f3.read(),
                    'Few-shot expl' : f4.read()
                    }
        elif argv[2] == 'xml':
            with open(f'{dev_path}/zeroshot_v{argv[3]}.txt') as f1,\
                open(f'{dev_path}/oneshot_v{argv[3]}.txt') as f2, \
                open(f'{dev_path}/fewshot_rank_v{argv[3]}.txt') as f3, \
                open(f'{dev_path}/fewshot_expl_v{argv[3]}.txt') as f4:
                prompt_dict = {
                    'Zero-shot' : f1.read(),
                    'One-shot' : f2.read(),
                    'Few-shot rank' : f3.read(),
                    'Few-shot expl' : f4.read()
                    }
    
    elif argv[1] == 'eval':
        eval_path = str(os.getcwd()) + f'/prompt_eval/{argv[2]}/doc{argv[3]}'
        with open(f'{eval_path}/zeroshot_eval_doc{argv[3]}.txt') as f1, \
             open(f'{eval_path}/oneshot_eval_doc{argv[3]}.txt') as f2, \
             open(f'{eval_path}/fewshot_rank_eval_doc{argv[3]}.txt') as f3, \
             open(f'{eval_path}/fewshot_expl_eval_doc{argv[3]}.txt') as f4:
            prompt_dict = {
                'Zero-shot' : f1.read(),
                'One-shot' : f2.read(),
                'Few-shot rank' : f3.read(),
                'Few-shot expl' : f4.read()
                }
    
    data_dict = dict()

    # run the prompts
    for key, prompt in prompt_dict.items():
        output = pipe_phi(prompt, **generation_args)
        data_dict[key] = output[0]['generated_text']

    # store the summaries in the eval_data file
    if argv[1] == 'eval':

        with open(argv[4]) as fp:
            eval_dict = json.load(fp)

        eval_dict[argv[3]]["Phi-3"] = data_dict

        with open(argv[4], 'w') as fp:
            json.dump(eval_dict, fp, indent=3)

    # in case of dev summaries, save them in the current directory
    elif argv[1] == 'dev' and argv[2] == 'xml':
        with open(f'phi3_results_xml_{argv[2]}.json', 'w') as fp:
            json.dump(data_dict, fp, indent=1)

    elif argv[1] == 'dev' and argv[2] == 'phi3':
        with open(f'phi3_results_phi3_{argv[2]}.json', 'w') as fp:
            json.dump(data_dict, fp, indent=1)

if __name__ == "__main__":
    main(sys.argv)
