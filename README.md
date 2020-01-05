# Detecting Deception In Political Debates

## Requirements
  - Python 3.6.7+
  - Install Python packages `pip3 install --upgrade pip && pip3 install -r requirements.txt`
  - Download OpenSmile (https://www.audeering.com/what-we-do/opensmile/)
  - Create `config.json` in root directory (see `config.json.example`) and set the correct OpenSmile path
  - Run `python3 data_preprocessing.py`
  - Download pretrained BERT uncased_L-12_H-768_A-12 model to `estimators/models/bert/` (https://github.com/google-research/bert)
  - Start `bert-as-service` for BERT CLS token extraction in the background: `cd estimators/models/bert/ && bert-serving-start -pooling_strategy CLS_TOKEN -max_seq_len NONE -model_dir ./uncased_L-12_H-768_A-12/`
