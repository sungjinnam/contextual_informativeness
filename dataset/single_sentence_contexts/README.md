This directory contains the raw crowdsource data and processed results for the single-sentence context dataset.

Dataset file: `cloze_df_scores_all3.pickle`

Data Columns:
- `cloze_resp`: collected cloze responses and counts
- `sentID_GM`: internal sentence ID
- `sentence`: sentence with target word marked as blank
- `target`: target word from the sentence
- `targ_POS`: POS of the target word
- `cloze_resp_len`: number of collected cloze responses
- `ent_cloze`: entropy score of cloze response counts
- `ent_elmo`: entropy score of cloze responses from ELMo model
- `bert_score`: semantic density of cloze responses from BERT model
- `elmo_score`: semantic density of cloze responses from ELMo model
- `glove_score`: semantic density of cloze responses from GloVE model	- `sent_idx`: sentence index
- `scores`: raw BWS scores
- `scores_sum`: Sum of BWS scores
- `scores_avg`: Average of BWS scores
- `scores_std`: Standard deviation of BWS scores
- `scores_avg_rank`: Sentence ranking by BWS score average	
- `sent_len`: Length of the sentence in words
- `targ_loc`: Number of words that exist before the target word
- `targ_loc_before`: Number of words that exist after the target word
- `targ_loc_end`: If the target word is located at the end of the sentence
- `targ_loc_rel`: Target word's relative location