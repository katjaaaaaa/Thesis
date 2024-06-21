# BA Thesis

This repository contains the code used in the experiment made for the Informatiekunde programme bachelor thesis at Groningen University. The study explored the use of various prompting techniques to automatically generate summaries of the related work sections of scientific documents. Please refer to the paper for a full explanation of the approach.  

Below we elaborate on the use of Hábrók, and the process of running the necessary code to conduct the experiment.

## The experiment environment and Hábrók

Hábrók is a computing cluster provided by the Center of Information Technology (CIT) of the University of Groningen. With Hábrók, we were able to run multiple scripts that required a lot of time and computational power without using the local machine. We used it for the following scripts:  
- `phi3_model_prompting.py`
- `bertscore_calculate.py`

A jobscript is required to run the scripts on Hábrók. We provided the configuration jobscript files used during the experiment in the __habrok_config__ directory. The jobscripts are based on the specific virtual environment name, please change the name or also use the name `thesis_venv` to run the scripts. The necessary packages needed for the experiment are provided in `requirements.txt`.

For further help with the use of Hábrók, please visit its official documentation: https://wiki.hpc.rug.nl/habrok/start .

## Running the code
Before running the programs, install the necessary packages with this command:
```
pip install -r requirements. txt
```
### 1. Obtaining the data
The data used in the experiment was collected by Kobayashi et al. (2022). The link to their repository: https://github.com/citation-minami-lab/acl-citation-dataset/tree/main

To obtain a random selection of 20 related work sections, run the following command:
```
python3 extract_section.py 20
```

### 2. Generating the summaries
We use the small language model Phi-3 to generate the summaries of the related work sections. As input, we use the manually-created prompts stored as the .txt files in the __prompt_dev__ and __prompt_eval__ directories. The model takes the following inputs:

#### a. Development set summary generation
Use the command below to generate the version 3_5 prompt summaries with _XML structure_. The choice of the versions is: 2, 3, 3_5
```
python3 phi3_model_prompting.py dev xml 3_5
```
Use the command below to generate the version 5 prompt summaries with _Phi-3 structure_. The choice of the versions is: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
```
python3 phi3_model_prompting.py dev phi3 5
```

#### b. Evaluation set summary generation
The evaluation set prompt division is based on the length of the task instructions applied in the prompt: short instructions and long instructions. Further, each prompt selection directly correlates with the document number from which the related work section is taken. The list of the document numbers is as follows: 1, 5, 7, 8, 9, 10, 14, 15, 17, 20  

To generate the document 1 summaries using the _short prompt instructions_, use the command below.
```
phi3_model_prompting.py eval short 1 eval_data_short.json
```
The `eval_data_short.json` will collect the generated summaries and put them in the __results_eval__ directory. Applying the same file name when running the code is important to ensure all generated summaries are stored in one place.

### 3. Applying BertScore F1
A crucial part of the evaluation discussed in the paper is calculating the semantic similarity score between the Phi-3-generated summaries and LLM-generated summaries. We apply the BertScore to provide the scores. To calculate the BertScore of the document 7 summaries, use the command below:
```
python3 bertscore_calculate.py 7 long "Few-shot Expl"
```
The `long` argument indicates which task instruction must be applied (short/long). 

The `"Few-shot Expl"` argument indicates which of the summary's prompt types (Zero-shot/One-shot/Few-shot rank/Few-shot expl) must receive an F1 score of 0. If no argument is included, the script takes all summaries in the evaluation. The output of the script is placed in the __results_eval__ directory.
