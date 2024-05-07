import json
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          # BitsAndBytesConfig,
                          pipeline)

def main():

    # define the model
    model_phi = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="cuda",
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

    prompt_zeroshot = '''
        <|user|>
        You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
        - <Authors goes here>: <contribution of the authors goes here>
        Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

        Text: "To the best of our knowledge, HappyDB is the first crowdsourced corpus of happy moments which can be used for understanding the language people use to describe happy events.  There has been recent interest in creating datasets in the area of mental health.  Althoff et al. (Althoff et al., 2016) conducted a large scale analysis on counseling conversational logs collected from short message services (SMS) for mental illness study.  They studied how various linguistic aspects of conversations are correlated with conversation outcomes.  Mihalcea et al. (Mihalcea and Liu, 2006) performed text analysis on blog posts from LiveJournal (where posts can be assigned happy/sad tags by their authors).  Lin et al. (Lin et al., 2016) measure stress from short texts by identifying the stressors and the stress levels.  They classified tweets into 12 stress categories defined by the stress scale in (Holmes and Rahe, 1967).  Last year alone, there were multiple research efforts that obtained datasets via crowdsourcing and applied natural language techniques to understand different corpora.  For example, SQuAD (Rajpurkar et al., 2016) created a largescale dataset for question-answering.  The crowdsourced workers were asked to create questions based on a paragraph obtained from Wikipedia.  They employed MTurk workers with strong experience and a high approval rating to ensure the quality of the dataset.  We did not select the workers based on their qualification for HappyDB as our task is cognitively easier than SQuAD's and we want to avoid bias in our corpus.  HappyDB is similar to SQuAD in terms of the scale of the crowdsourced dataset.  However, unlike SQuAD, which was designed specifically for studying the question answering problem, the problems that HappyDB can be used for are more open-ended. "<|end|>
        <|assistant|>
    '''

    prompt_oneshot = '''
        <|user|>
        You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
        - <Authors goes here>: <contribution of the authors goes here>
        Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

       Text: The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.<|end|>
       <|assistant|>
       **Utterance-Level Multimodal Feature Fusion**

        - **Zadeh et al. (2017)**: Proposed Tensor Fusion to capture unimodal, bimodal, and trimodal interactions. However, this method uses the three-fold Cartesian product to fuse the multimodal features, which results in high time cost.
        - **Liu et al. (2018)**: Presented the Efficient Low-rank Multimodal Fusion, which applies multimodal fusion using low-rank tensors to accelerate the fusion process.
        - **Mai et al. (2020)**: Proposed a graph fusion network to model unimodal, bimodal, and trimodal interactions successively.

        **Word-Level Multimodal Feature Fusion**

        - **Zadeh et al. (2018a)**: Proposed the Memory Fusion Network (MFN) to capture the interactions across both different modalities and timesteps.
        - **Wang et al. (2019)**: Proposed the Recurrent Attended Variation Embedding Network (RAVEN). This model applies the Attention Gating module to fuse the word-level features, which can dynamically use the non-verbal features to shift the word embeddings.
        - **Tsai et al. (2019)**: Presented multimodal transformer (Mult), which uses the cross-modal attention to capture the bimodal interactions, motivated by the great success of transformer in NLP (Vaswani et al., 2017).
        - **Pham et al. (2019)**: Proposed that translation from a source to a target modality provides a way to learn joint representations and proposed the Multimodal Cyclic Translation Network model (MCTN) to learn joint multimodal representations.

        **Authors of the Paper**

        - **Authors of the Paper**: Their work is based on word-level features. They use the crossmodal prediction task to distinguish the shared and private non-textual features instead of training the model as an auxiliary task. This approach allows them to obtain more useful information by deeply probing the cross-modal prediction model.<|end|>
        <|user|>
        You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
        - <Authors goes here>: <contribution of the authors goes here>
        Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.
        Text: "To the best of our knowledge, HappyDB is the first crowdsourced corpus of happy moments which can be used for understanding the language people use to describe happy events.  There has been recent interest in creating datasets in the area of mental health.  Althoff et al. (Althoff et al., 2016) conducted a large scale analysis on counseling conversational logs collected from short message services (SMS) for mental illness study.  They studied how various linguistic aspects of conversations are correlated with conversation outcomes.  Mihalcea et al. (Mihalcea and Liu, 2006) performed text analysis on blog posts from LiveJournal (where posts can be assigned happy/sad tags by their authors).  Lin et al. (Lin et al., 2016) measure stress from short texts by identifying the stressors and the stress levels.  They classified tweets into 12 stress categories defined by the stress scale in (Holmes and Rahe, 1967).  Last year alone, there were multiple research efforts that obtained datasets via crowdsourcing and applied natural language techniques to understand different corpora.  For example, SQuAD (Rajpurkar et al., 2016) created a largescale dataset for question-answering.  The crowdsourced workers were asked to create questions based on a paragraph obtained from Wikipedia.  They employed MTurk workers with strong experience and a high approval rating to ensure the quality of the dataset.  We did not select the workers based on their qualification for HappyDB as our task is cognitively easier than SQuAD's and we want to avoid bias in our corpus.  HappyDB is similar to SQuAD in terms of the scale of the crowdsourced dataset.  However, unlike SQuAD, which was designed specifically for studying the question answering problem, the problems that HappyDB can be used for are more open-ended. "<|end|>
        <|assistant|>
    '''

    prompt_fewshot_rankings = '''
    <|user|>
    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text: The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.<|end|>
    <|assistant|>
    ## Multimodal sentiment analysis: Utterance-level vs. word-level features
    This passage discusses two main approaches for multimodal sentiment analysis: utterance-level and word-level feature fusion.
    **Utterance-level multimodal feature fusion**
    * Focuses on capturing sentiment of the entire utterance (e.g., sentence)
    * Methods:
    * Average frame-level visual/acoustic features to create utterance-level features (the authors of the paper being summarized)
    * Apply RNNs on words to get utterance-level textual features (the authors of the paper being summarized)
    * Fusion models:
    * Tensor Fusion (Zadeh et al., 2017) - captures unimodal, bimodal and trimodal interactions (expensive to compute)
    * Efficient Low-rank Multimodal Fusion (Liu et al., 2018) - faster alternative to Tensor Fusion using low-rank tensors
    * Graph fusion network (Mai et al., 2020) - models interactions sequentially

    **Word-level multimodal feature fusion**
    * Aims to capture sentiment variations within an utterance by focusing on individual words
    * Requires force alignment to identify timestamps for each word
    * Methods to extract word-level features:
    * Split utterance into video clips based on word timestamps (the authors of the paper being summarized)
    * Average frame-level features within each clip (the authors of the paper being summarized)
    * Fusion models:
    * Memory Fusion Network (MFN) (Zadeh et al., 2018a) - captures interactions between modalities and time steps
    * Recurrent Attended Variation Embedding Network (RAVEN) (Wang et al., 2019) - dynamically shifts word embeddings based on non-verbal cues using Attention Gating
    * Multimodal Transformer (Mult) (Tsai et al., 2019) - employs cross-modal attention to capture interactions between modalities

    **Additionally mentioned work:**
    * Multimodal Cyclic Translation Network (MCTN) (Pham et al., 2019) - learns joint representations by translating between modalities (the authors of the paper being summarized)
    * This work differs from the approach proposed in the current paper by using a cross-modal prediction task instead of an auxiliary translation task.<|end|>
    <|user|>
    I rated this summary based on the following criteria:
    1. Quality of the summary's structure: The summary follows the paper's structure and contains necessary subsections if they were introduced in the paper. In other terms, if the text adds a concept and then provides the cited papers related to the concept, the model must structure the summary to highlight the relation between them.
    2. Quality of the summarization: The summary has no hallucinations, and all the information can be found in the source text.
    3. Quality of the sources' citations: Each cited paper mentioned in the target paper is introduced in the summary and contains a brief but clear description of what this source's contribution is.
    Based on these criteria I ranked the summary as follows:
    Quality structure: 2/5
    Quality translation: 3/5
    Quality citations: 4/5

    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text: The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.<|end|>
    <|assistant|>
    **Key Concepts and Contributions:**
    1. **Utterance-level Multimodal Feature Fusion:**
    - Techniques involve extracting overall utterance features and applying fusion methods.
    - Zadeh et al. (2017): Proposed Tensor Fusion method explicitly capturing unimodal, bimodal, and trimodal interactions.
    - Liu et al. (2018): Introduced Efficient Low-rank Multimodal Fusion using low-rank tensors for faster fusion process.
    - Mai et al. (2020): Proposed a graph fusion network for modeling unimodal, bimodal, and trimodal interactions successively.
    2. **Word-level Multimodal Feature Fusion:**
    - Recent focus due to limitations of utterance-level features.
    - Process involves force alignment for word timestamps and averaging frame-level features of video clips.
    - Zadeh et al. (2018a): Introduced Memory Fusion Network (MFN) capturing interactions across modalities and timesteps.
    - Wang et al. (2019): Developed Recurrent Attended Variation Embedding Network (RAVEN) using Attention Gating module for dynamic fusion.
    - Tsai et al. (2019): Presented Multimodal transformer (Mult) using cross-modal attention to capture bimodal interactions.

    3. **Comparison with Related Work:**
    - Pham et al. (2019): Proposed Multimodal Cyclic Translation Network model (MCTN) for joint multimodal representations through translation.
    - Authors emphasized the use of crossmodal prediction task instead of training as an auxiliary task, aiming for deeper insights into shared and private non-textual features.
    This summary provides an overview of the main concepts and contributions in the scientific paper regarding multimodal sentiment analysis, as well as the advancements proposed by various researchers.<|end|>
    <|user|>
    I rated this summary based on the following criteria:
    1. Quality of the summary's structure: The summary follows the paper's structure and contains necessary subsections if they were introduced in the paper. In other terms, if the text adds a concept and then provides the cited papers related to the concept, the model must structure the summary to highlight the relation between them.
    2. Quality of the summarization: The summary has no hallucinations, and all the information can be found in the source text.
    3. Quality of the sources' citations: Each cited paper mentioned in the target paper is introduced in the summary and contains a brief but clear description of what this source's contribution is.
    Based on these criteria I ranked the summary as follows:
    Quality structure: 5/5
    Quality summarization: 3/5
    Quality sources: 5/5

    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text: The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.<|end|>
    <|assistant|>
    The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.

    **Utterance-Level Multimodal Feature Fusion**

    - **Zadeh et al. (2017)**: Proposed Tensor Fusion to capture unimodal, bimodal, and trimodal interactions. However, this method uses the three-fold Cartesian product to fuse the multimodal features, which results in high time cost.
    - **Liu et al. (2018)**: Presented the Efficient Low-rank Multimodal Fusion, which applies multimodal fusion using low-rank tensors to accelerate the fusion process.
    - **Mai et al. (2020)**: Proposed a graph fusion network to model unimodal, bimodal, and trimodal interactions successively.

    **Word-Level Multimodal Feature Fusion**

    - **Zadeh et al. (2018a)**: Proposed the Memory Fusion Network (MFN) to capture the interactions across both different modalities and timesteps.
    - **Wang et al. (2019)**: Proposed the Recurrent Attended Variation Embedding Network (RAVEN). This model applies the Attention Gating module to fuse the word-level features, which can dynamically use the non-verbal features to shift the word embeddings.
    - **Tsai et al. (2019)**: Presented multimodal transformer (Mult), which uses the cross-modal attention to capture the bimodal interactions, motivated by the great success of transformer in NLP (Vaswani et al., 2017).
    - **Pham et al. (2019)**: Proposed that translation from a source to a target modality provides a way to learn joint representations and proposed the Multimodal Cyclic Translation Network model (MCTN) to learn joint multimodal representations.

    **Authors of the Paper**

    - **Authors of the Paper**: Their work is based on word-level features. They use the crossmodal prediction task to distinguish the shared and private non-textual features instead of training the model as an auxiliary task. This approach allows them to obtain more useful information by deeply probing the cross-modal prediction model.<|end|>
    <|user|>
    I rated this summary based on the following criteria:
    1. Quality of the summary's structure: The summary follows the paper's structure and contains necessary subsections if they were introduced in the paper. In other terms, if the text adds a concept and then provides the cited papers related to the concept, the model must structure the summary to highlight the relation between them.
    2. Quality of the summarization: The summary has no hallucinations, and all the information can be found in the source text.
    3. Quality of the sources' citations: Each cited paper mentioned in the target paper is introduced in the summary and contains a brief but clear description of what this source's contribution is.
    Based on these criteria I ranked the summary as follows:
    Quality structure: 5/5
    Quality summarization: 5/5
    Quality sources: 5/5

    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text:"To the best of our knowledge, HappyDB is the first crowdsourced corpus of happy moments which can be used for understanding the language people use to describe happy events.  There has been recent interest in creating datasets in the area of mental health.  Althoff et al. (Althoff et al., 2016) conducted a large scale analysis on counseling conversational logs collected from short message services (SMS) for mental illness study.  They studied how various linguistic aspects of conversations are correlated with conversation outcomes.  Mihalcea et al. (Mihalcea and Liu, 2006) performed text analysis on blog posts from LiveJournal (where posts can be assigned happy/sad tags by their authors).  Lin et al. (Lin et al., 2016) measure stress from short texts by identifying the stressors and the stress levels.  They classified tweets into 12 stress categories defined by the stress scale in (Holmes and Rahe, 1967).  Last year alone, there were multiple research efforts that obtained datasets via crowdsourcing and applied natural language techniques to understand different corpora.  For example, SQuAD (Rajpurkar et al., 2016) created a largescale dataset for question-answering.  The crowdsourced workers were asked to create questions based on a paragraph obtained from Wikipedia.  They employed MTurk workers with strong experience and a high approval rating to ensure the quality of the dataset.  We did not select the workers based on their qualification for HappyDB as our task is cognitively easier than SQuAD's and we want to avoid bias in our corpus.  HappyDB is similar to SQuAD in terms of the scale of the crowdsourced dataset.  However, unlike SQuAD, which was designed specifically for studying the question answering problem, the problems that HappyDB can be used for are more open-ended. "<|end|>
    <|assistant|>
    '''

    prompt_fewshot_explanation = '''
    <|user|>
    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text: The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.<|end|>
    <|assistant|>
    ## Multimodal sentiment analysis: Utterance-level vs. word-level features
    This passage discusses two main approaches for multimodal sentiment analysis: utterance-level and word-level feature fusion.
    **Utterance-level multimodal feature fusion**
    * Focuses on capturing sentiment of the entire utterance (e.g., sentence)
    * Methods:
    * Average frame-level visual/acoustic features to create utterance-level features (the authors of the paper being summarized)
    * Apply RNNs on words to get utterance-level textual features (the authors of the paper being summarized)
    * Fusion models:
    * Tensor Fusion (Zadeh et al., 2017) - captures unimodal, bimodal and trimodal interactions (expensive to compute)
    * Efficient Low-rank Multimodal Fusion (Liu et al., 2018) - faster alternative to Tensor Fusion using low-rank tensors
    * Graph fusion network (Mai et al., 2020) - models interactions sequentially

    **Word-level multimodal feature fusion**
    * Aims to capture sentiment variations within an utterance by focusing on individual words
    * Requires force alignment to identify timestamps for each word
    * Methods to extract word-level features:
    * Split utterance into video clips based on word timestamps (the authors of the paper being summarized)
    * Average frame-level features within each clip (the authors of the paper being summarized)
    * Fusion models:
    * Memory Fusion Network (MFN) (Zadeh et al., 2018a) - captures interactions between modalities and time steps
    * Recurrent Attended Variation Embedding Network (RAVEN) (Wang et al., 2019) - dynamically shifts word embeddings based on non-verbal cues using Attention Gating
    * Multimodal Transformer (Mult) (Tsai et al., 2019) - employs cross-modal attention to capture interactions between modalities

    **Additionally mentioned work:**
    * Multimodal Cyclic Translation Network (MCTN) (Pham et al., 2019) - learns joint representations by translating between modalities (the authors of the paper being summarized)
    * This work differs from the approach proposed in the current paper by using a cross-modal prediction task instead of an auxiliary translation task.<|end|>
    <|user|>
    I rated this summary based on the following criteria:
    1. Quality of the summary's structure: The summary follows the paper's structure and contains necessary subsections if they were introduced in the paper. In other terms, if the text adds a concept and then provides the cited papers related to the concept, the model must structure the summary to highlight the relation between them.
    2. Quality of the summarization: The summary has no hallucinations, and all the information can be found in the source text.
    3. Quality of the sources' citations: Each cited paper mentioned in the target paper is introduced in the summary and contains a brief but clear description of what this source's contribution is.
    Based on these criteria I ranked the summary as follows:
    Quality structure: 2/5 (Reasoning: multiple structure hallucinations)
    Quality translation: 3/5 (Reasoning: bad formulation)
    Quality citations: 4/5 (Reasoning: incorrect use of author citation "(the authors of the paper being summarized)")

    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text: The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.<|end|>
    <|assistant|>
    **Key Concepts and Contributions:**
    1. **Utterance-level Multimodal Feature Fusion:**
    - Techniques involve extracting overall utterance features and applying fusion methods.
    - Zadeh et al. (2017): Proposed Tensor Fusion method explicitly capturing unimodal, bimodal, and trimodal interactions.
    - Liu et al. (2018): Introduced Efficient Low-rank Multimodal Fusion using low-rank tensors for faster fusion process.
    - Mai et al. (2020): Proposed a graph fusion network for modeling unimodal, bimodal, and trimodal interactions successively.
    2. **Word-level Multimodal Feature Fusion:**
    - Recent focus due to limitations of utterance-level features.
    - Process involves force alignment for word timestamps and averaging frame-level features of video clips.
    - Zadeh et al. (2018a): Introduced Memory Fusion Network (MFN) capturing interactions across modalities and timesteps.
    - Wang et al. (2019): Developed Recurrent Attended Variation Embedding Network (RAVEN) using Attention Gating module for dynamic fusion.
    - Tsai et al. (2019): Presented Multimodal transformer (Mult) using cross-modal attention to capture bimodal interactions.

    3. **Comparison with Related Work:**
    - Pham et al. (2019): Proposed Multimodal Cyclic Translation Network model (MCTN) for joint multimodal representations through translation.
    - Authors emphasized the use of crossmodal prediction task instead of training as an auxiliary task, aiming for deeper insights into shared and private non-textual features.
    This summary provides an overview of the main concepts and contributions in the scientific paper regarding multimodal sentiment analysis, as well as the advancements proposed by various researchers.<|end|>
    <|user|>
    I rated this summary based on the following criteria:
    1. Quality of the summary's structure: The summary follows the paper's structure and contains necessary subsections if they were introduced in the paper. In other terms, if the text adds a concept and then provides the cited papers related to the concept, the model must structure the summary to highlight the relation between them.
    2. Quality of the summarization: The summary has no hallucinations, and all the information can be found in the source text.
    3. Quality of the sources' citations: Each cited paper mentioned in the target paper is introduced in the summary and contains a brief but clear description of what this source's contribution is.
    Based on these criteria I ranked the summary as follows:
    Quality structure: 5/5
    Quality summarization: 3/5 (Reasoning: one hallucination present)
    Quality sources: 5/5

    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text: The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.<|end|>
    <|assistant|>
    The text discusses two main lines of work in multimodal sentiment analysis: utterance-level multimodal feature fusion and word-level multimodal feature fusion.

    **Utterance-Level Multimodal Feature Fusion**

    - **Zadeh et al. (2017)**: Proposed Tensor Fusion to capture unimodal, bimodal, and trimodal interactions. However, this method uses the three-fold Cartesian product to fuse the multimodal features, which results in high time cost.
    - **Liu et al. (2018)**: Presented the Efficient Low-rank Multimodal Fusion, which applies multimodal fusion using low-rank tensors to accelerate the fusion process.
    - **Mai et al. (2020)**: Proposed a graph fusion network to model unimodal, bimodal, and trimodal interactions successively.

    **Word-Level Multimodal Feature Fusion**

    - **Zadeh et al. (2018a)**: Proposed the Memory Fusion Network (MFN) to capture the interactions across both different modalities and timesteps.
    - **Wang et al. (2019)**: Proposed the Recurrent Attended Variation Embedding Network (RAVEN). This model applies the Attention Gating module to fuse the word-level features, which can dynamically use the non-verbal features to shift the word embeddings.
    - **Tsai et al. (2019)**: Presented multimodal transformer (Mult), which uses the cross-modal attention to capture the bimodal interactions, motivated by the great success of transformer in NLP (Vaswani et al., 2017).
    - **Pham et al. (2019)**: Proposed that translation from a source to a target modality provides a way to learn joint representations and proposed the Multimodal Cyclic Translation Network model (MCTN) to learn joint multimodal representations.

    **Authors of the Paper**

    - **Authors of the Paper**: Their work is based on word-level features. They use the crossmodal prediction task to distinguish the shared and private non-textual features instead of training the model as an auxiliary task. This approach allows them to obtain more useful information by deeply probing the cross-modal prediction model.<|end|>
    <|user|>
    I rated this summary based on the following criteria:
    1. Quality of the summary's structure: The summary follows the paper's structure and contains necessary subsections if they were introduced in the paper. In other terms, if the text adds a concept and then provides the cited papers related to the concept, the model must structure the summary to highlight the relation between them.
    2. Quality of the summarization: The summary has no hallucinations, and all the information can be found in the source text.
    3. Quality of the sources' citations: Each cited paper mentioned in the target paper is introduced in the summary and contains a brief but clear description of what this source's contribution is.
    Based on these criteria I ranked the summary as follows:
    Quality structure: 5/5
    Quality summarization: 5/5
    Quality sources: 5/5

    You are a summarization model. Generate a summary of a text taken from a scientific paper. Divide the summary into key concepts and/or subsections. Per concept list each cited paper's authors and contributions. The examples of how authors can be notated in the text are "Author et al. (2019)" and "(Author et al., 2017)". If no authors are introduced, assign the contribution to the authors of the paper the text was taken from. The structure of each contribution must be:
    - <Authors goes here>: <contribution of the authors goes here>
    Ensure that your answer is accurate and doesn't contain any information not directly supported by the text.

    Text:"To the best of our knowledge, HappyDB is the first crowdsourced corpus of happy moments which can be used for understanding the language people use to describe happy events.  There has been recent interest in creating datasets in the area of mental health.  Althoff et al. (Althoff et al., 2016) conducted a large scale analysis on counseling conversational logs collected from short message services (SMS) for mental illness study.  They studied how various linguistic aspects of conversations are correlated with conversation outcomes.  Mihalcea et al. (Mihalcea and Liu, 2006) performed text analysis on blog posts from LiveJournal (where posts can be assigned happy/sad tags by their authors).  Lin et al. (Lin et al., 2016) measure stress from short texts by identifying the stressors and the stress levels.  They classified tweets into 12 stress categories defined by the stress scale in (Holmes and Rahe, 1967).  Last year alone, there were multiple research efforts that obtained datasets via crowdsourcing and applied natural language techniques to understand different corpora.  For example, SQuAD (Rajpurkar et al., 2016) created a largescale dataset for question-answering.  The crowdsourced workers were asked to create questions based on a paragraph obtained from Wikipedia.  They employed MTurk workers with strong experience and a high approval rating to ensure the quality of the dataset.  We did not select the workers based on their qualification for HappyDB as our task is cognitively easier than SQuAD's and we want to avoid bias in our corpus.  HappyDB is similar to SQuAD in terms of the scale of the crowdsourced dataset.  However, unlike SQuAD, which was designed specifically for studying the question answering problem, the problems that HappyDB can be used for are more open-ended. "<|end|>
    <|assistant|>
    '''

    prompt_dict = {'zeroshot': prompt_zeroshot, 'oneshot' : prompt_oneshot, 'fewshot_rankings' : prompt_fewshot_rankings, 'fewshot_explanation' : prompt_fewshot_explanation}
    results_dict = dict()

    # run the prompts
    for key, prompt in prompt_dict.items():
        output = pipe_phi(prompt, **generation_args)
        results_dict[key] = output

    # store the outputs in a json file
    with open('phi3_results.json', 'w') as fp:
        json.dump(results_dict, fp, indent=1)

if __name__ == "__main__":
    main()