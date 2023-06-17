@echo off

set DATA_PATH=".\data"
set RAW_PATH=".\raw"
set FEATURES_PATH=".\features"
set RESULTS_PATH=".\results"

if not exist %DATA_PATH% (
  mkdir %DATA_PATH%
)

cd %DATA_PATH%

if not exist %RAW_PATH% (
  mkdir %RAW_PATH%
)

if not exist %FEATURES_PATH% (
  mkdir %FEATURES_PATH%
)

if not exist %RESULTS_PATH% (
  mkdir %RESULTS_PATH%
)

cd %RESULTS_PATH%

if not exist ".\new" (
  mkdir ".\new"
)

cd ..

set PRETRAINED_PATH=".\pretrained"

if not exist %PRETRAINED_PATH% (
  mkdir %PRETRAINED_PATH%
)

cd %PRETRAINED_PATH%

echo Downloading pretrained models ...
if not exist "word2vec-input_sent.txt-s300-c5-w5-e10-SG.model.trainables.syn1neg.zip" (
  gdown 1553_9shAUrQpFAB0vqXbQrpHqJvWYgp4
) else (
  echo File word2vec-input_sent.txt-s300-c5-w5-e10-SG.model.trainables.syn1neg.zip already exists.
)

if not exist "kyubyong.ft.id.bin" (
  gdown 1vL8vyfJbaj3i91peTu738jq25N-yhsU3
) else (
  echo File kyubyong.ft.id.bin already exists.
)

if not exist "word2vec_news.model.wv.vectors.zip" (
  gdown 1MQjcRLBCJsdk3AyCBWfAkltzRTHhI9ED
) else (
  echo File word2vec_news.model.wv.vectors.zip already exists.
)

cd ..
cd ..

echo Please unarchive all archived files that has been downloaded.
