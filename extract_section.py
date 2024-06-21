# Name: extract_section.py
# Author: Katja Kamyshanova
# Date: 04/06/2024
# This program takes n of entries as argument, loops through 
# the dataset, randomly chooses n entries, pre-processes them
# and returns as a .json file

# Usage: python3 extract_section.py <required amound of sections in output>
# Example usage: python3 extract_section.py 20

import sys
import json
import regex as re
import random
import os

def main(argv):

    data_new = dict()    

    data_dir = str(os.getcwd()) + '/data'
    for file in os.listdir(data_dir):
        with open(os.path.join(data_dir, file)) as f1:
            data = f1.read()

        data_json = json.loads(data)

        for item in data_json:
            title = item['Title'] # Title
            data_new[title] = dict()

            # Pattern for subsection()
            pattern = r"%subsection\{(.*?)\}%"

            section_text = ' '.join(item['Sentences']) # Sentences
            section_text = section_text.replace('%cite{', '')
            section_text = section_text.replace('%{', '')
            section_text = section_text.replace('<par>', '')
            section_text = re.sub(pattern, r"subsection(\1)", section_text)
            section_text = section_text.replace('}%', '')
            data_new[title]['Sentences'] = section_text

        # The structure of the dataset: 
        # ['Title', 'Author', 'Url', 'Sentences', 
        # 'AnswersCitationWorthiness', 'CitedNumberList',
        # 'CollectedCitedNumberList', 'CitationAnchorList',
        # 'CitedPaperIndexList', 'CitedPaperTitle', 'CitedPaperText']

    samples = random.sample(data_new.items(), int(argv[1]))
    # Remove samples from the dataset to avoid dublicates in the test data
    for text in samples:
        del data_new[text[0]]

    # Dump the data for open-source LLMs in a json file
    with open('aclDataset_short_test.json', 'w') as fp:
        json.dump(samples, fp, indent=2)

    # Print the resulted entries
    for i in samples:
        print(i)

if __name__ == "__main__":
    main(sys.argv)