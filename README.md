# Single Document Automatic Summarization with Semantic Role Labeling
A Document summarization with span SRL based with regression linear as sentence importance model adapted from Multi Documment Summarization System by Khairunnisa (2021) [[1]](#1) 

# Installation
```
pip install -r requirements.txt
```

# Folders
Run this script to create needed folders in root and also pretrained models
```
make fetch-data
```

# Datasets
Datasets used in this summarization is Koto, et al. (2020) [[2]](#2) however to simplify I convert all the datas into a pandas dataframe, therefore we will have three files (train, validation, test) and each file has 'idx_news', 'clean_summary', 'clean_article', 'url' and 'extractive_ids' (following the field name in every json file in dataset)

# Semantic Role Labeling

## Feature Extraction
SRL used in this repository is a span-based SRL here, you can clone the repository in its respective place in this repository, which is in `src` folders because we will need its feature extracting module

## Pretrained Models
As this summarization system is SRL-based, we will need a pretrained SRL model, ready to be used (or you can train your own model first), place the pretrained SRL models (with any pretrained models) inside `data/pretrained`

# Testing with pretrained linear regression models
1. Download pretrained models
The provided models can be downloaded from [here](https://drive.google.com/drive/folders/1AF1VVCKjFOmr8SdFlWBNEj3Y45C2iXtL), please remember to put the pretrained regression models into `models` folder in root
2. Modify the `configurations.json` to fit the configurations the pretrained models was trained with. (The default is in this repository)
3. Run predict script
```
make predict config=$(config) out=$(filename)
```
There are two arguments that you can fill. First, config name, the default is `default` if you do not fill anything and the second one is output filename. The program will ask you two filenames, which contains the article and the title respectively, please note that the provided script can only summarize one article at a time, if you want to do bulk summarize, please modify as you see fit.

# Training with data
1. Make sure your data format follows the data format used in this system, the example will be provided in `data/raw/example.csv` (or you can make your own preprocess code, what we need article and its summary, title if can)
2. Run the script
```
make train config=$(config)
```
You can see the log while training, this will take times mostly in extracting features because we need to SRL all the sentences in all the articles and convert them into features. The trained models will be in `models`
3. If you want to test model with your validation data or test data, you can run either of the following scripts
```
make validate config=$(config)
```
or
```
make test config=$(config)
```.

The test data can be found at [https://drive.google.com/drive/folders/1AF1VVCKjFOmr8SdFlWBNEj3Y45C2iXtL](https://drive.google.com/drive/folders/1AF1VVCKjFOmr8SdFlWBNEj3Y45C2iXtL). Save it at `data/raw`.

# References
<a id="1">[1]</a> 
Khairunnisa, N.Y. (2021). Peringkasan Otomatis Kumpulan Berita Berbahasa 
Indonesia dengan Semantic Role Labeling dan Model Regresi Linier.
<br/>
<a id="2">[2]</a> 
Koto, F., Lau, J.H., & Baldwin, T. (2020). Liputan6: A Large-scale Indonesian 
Dataset for Text Summarization. AACL.
