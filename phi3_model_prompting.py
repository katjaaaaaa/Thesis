import json
import sys
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          # BitsAndBytesConfig,
                          pipeline)

def main(argv):

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
    with open(argv[1]) as f1, open(argv[2]) as f2, open(argv[3]) as f3, open(argv[4]) as f4:
        prompt_dict = {'zeroshot': f1.read(),
                   'oneshot' : f2.read(),
                   'fewshot_rankings' : f3.read(),
                   'fewshot_explanation' : f4.read()
                   }

    results_dict = dict()

    # run the prompts
    for key, prompt in prompt_dict.items():
        output = pipe_phi(prompt, **generation_args)
        results_dict[key] = output[0]['generated_text']

    # store the outputs in a json file
    with open('phi3_results.json', 'w') as fp:
        json.dump(results_dict, fp, indent=1)

if __name__ == "__main__":
    main(sys.argv)