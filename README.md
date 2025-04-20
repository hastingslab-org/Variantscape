# Variantscape

## Overview
Variantscape is a large-scale literature mining approach that combines LLM-driven entity extraction with co-association and network-based analysis to uncover variant–treatment–cancer relationships from biomedical abstracts.<br><br>
<img width="500" alt="Variantscape_3_cancers" src="https://github.com/user-attachments/assets/2efa2769-6ab3-4f12-82ed-9a94c0ae7e63" />

---
## Features
- Retrieving biomedical literature from a defined search string and timeframe, including preprocessing and cleaning
- ML- and LLM-enhanced extraction of genes, variants, treatments, and cancer types from titles and abstracts of biomedical literature
- Co-occurrence analysis for identifying variant–treatment–cancer relationships  
- Study design–weighted scoring to prioritize stronger evidence  
- Graph-based representation of biomedical associations  
- Integration of external biomedical APIs (OpenAlex, CIViC, MONDO, etc.) for enrichment  
- Exploratory analysis of rare or under-characterized variants
<br><br>

---
## Installation & Setup

All required packages and model downloads are handled directly within each Jupyter notebook.
<br><br>

## Technologies used
- Python and Jupyter Notebook for execution
- Named Entity Recognition (NER) models (e.g., SciSpaCy) for information extraction
- BioBERT (Biomedical BERT model) for information extraction
- Large Language Models (LLMs) (e.g., Llama 3.3) for variant extraction and classification
- OpenAlex API (for fetching of biomedical publications)
- CIViC API (for disease and treatment information)
- MyCancerInfo API (for gene name augmentation)
- MyDiseaseInfo API (for disease and cancer type synonyms)
- MyTreatmentInfo API (for treatment synonyms and aliases)
- MONDO API (for cancer ontology mapping and synonyms)
- General classifier (for study design classification)

> Note: This project uses several external APIs (OpenAlex, CIViC, MONDO, etc.) and NLP models (e.g., SciSpaCy, BioBERT).  
> Some notebooks use LLMs via DeepInfra (e.g., Llama 3.3). To use them, a DeepInfra account, API key, and credits are needed.
<br><br>
---

## Usage
This project is structured as a modular pipeline, with Jupyter notebooks organized into sequential folders:

```
01_fetching_articles
02_cleaning_and_normalization
03_gene_extraction
04_categorization
05_LLM_variant_extraction
    ...
```
<br><br>

### How to run
1. Start from `01_fetching_articles` and move step-by-step through each numbered folder.
2. Inside each folder, execute notebooks in order (e.g., `01.1`, `02.1`, etc.)
3. Each notebook handles its own imports, installations, and API requests.
4. Modify parameters (e.g., variant or cancer type) inside the notebook as needed.

> Internet access is required for live API calls (e.g., OpenAlex, CIViC, MONDO).  
> All notebooks are designed to be executed sequentially for a complete analysis workflow.
<br><br>
---
## Evaluation Studies

Two dedicated evaluation modules are included to validate and assess extraction quality:
- `NLP_gene_evaluation/`: Evaluates gene name extraction using traditional NLP techniques and biomedical NER models. Results are discussed in "Wosny, M. & Hastings, J., "Automated gene identification in oncology literature: A comparative evaluation of Natural Language Processing approaches."
- `LLM_variant_evaluation/`: Evaluates the performance of large language model–driven variant extraction. Results and methodology are described in "Wosny, M. & Hastings, J., "Large Language Models for Detection of Genetic Variants in Biomedical Literature."


These studies provide insight into the reliability and limitations of automated extraction methods for genes and variants for downstream analyses.
<br><br>

---

## Interactive Web Tool

To translate our findings into a usable resource, we developed a publicly accessible web tool that enables interactive exploration of the variant–treatment–cancer associations identified in this study.

The tool allows users to:
- Search by variant or gene  
- Filter by cancer type  
- Visualize co-association strengths with treatments and other cancer types

This interface supports exploratory navigation of literature-derived associations and may serve as a starting point for hypothesis generation and knowledge discovery in research settings.

Access the tool here: **[HASTINGSLAB]**
<br><br>
---

## How to cite

If you use this work, please cite the accompanying manuscripts:

- Wosny M., Boesch M., Peres T., Niederhauser T., Frueh M., Rothermundt C., Hastings J (2025) "Variantscape: Using large language models to build a comprehensive landscape of cancer variants" (Preprint)
- Wosny, M. & Hastings, J. (2025) Automated gene identification in oncology literature: A comparative evaluation of Natural Language Processing approaches. Studies in Health Technology and Informatics (Preprint)
- Wosny, M. & Hastings, J. (2025) Large Language Models for Detection of Genetic Variants in Biomedical Literature. (Preprint)

---
