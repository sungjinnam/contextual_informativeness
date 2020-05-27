# Introduction
- The repository includes Jupyter notebook files for *Attention-based Learning Models for Predicting Contextual Informativeness* paper. 

# Installing dependencies
Using Conda:
```
conda env create -f environment.yml
```

# External datasets
- https://github.com/kapelner/predicting_contextual_informativeness
- https://github.com/esantus/EVALution

# Other downloads
- Model weights and prediction results will be shared soon. 

# Notebook files
- 1. Predicting contexutal informativeness for the single-sentence context dataset.
    - 1-0: Baseline models
    - 1-1: ELMo and BERT based models
    - 1-2: Summerizing the results
- 2. Predicting contexutal informativeness for the multi-sentence context dataset (Kapelner et al.).
    - 2-0: Pre-processing the dataset
    - 2-1: Baseline models
    - 2-2: ELMo and BERT based models
    - 2-3: Cross prediction between models
    - 2-4: Summerizing the results
- 3. Evaluating the attention weights with the EVALution dataset
