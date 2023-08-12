Note: This program has been tested only on Windows.

It is recommended to use Python virtual environment before using this program.

For more information, please check the [original repository](https://github.com/feliciagojali/sentence-imp).

# Installation
```
pip install -r requirements.txt
fetch_data
```

# Testing
1. Download the pretrained model and the test data
from [https://drive.google.com/drive/folders/1AF1VVCKjFOmr8SdFlWBNEj3Y45C2iXtL](https://drive.google.com/drive/folders/1AF1VVCKjFOmr8SdFlWBNEj3Y45C2iXtL). Put the pretrained regression models into `models` folder in project root, and the test data into `data/raw`.
2. Run this command:
```
predict <config>
```
For `<config>` argument, the default is `default` if you do not fill anything. The program will ask you three paths: article file path, article title file path, and reference summary file path. These files should exists in `data/interactive` path. Please note that the provided script can only summarize one article at a time, if you want to do bulk summarize, please modify as you see fit. You can also customize or add new configuration by modifying `configuration.json`.
