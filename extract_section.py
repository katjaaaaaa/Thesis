# Paper: Dataset Construction for Scientific-Document Writing Support by Extracting Related Work Section and Citations from PDF Papers
# Link: https://github.com/citation-minami-lab/acl-citation-dataset

import sys
import json
import numpy as np
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

        # ['Title', 'Author', 'Url', 'Sentences', 
        # 'AnswersCitationWorthiness', 'CitedNumberList',
        # 'CollectedCitedNumberList', 'CitationAnchorList',
        # 'CitedPaperIndexList', 'CitedPaperTitle', 'CitedPaperText']

    samples = random.sample(data_new.items(), 25)
    with open('aclDataset_short.json', 'w') as fp:
        json.dump(samples, fp, indent=2)
    
    for i in samples:
        print(i)

if __name__ == "__main__":
    main(sys.argv)