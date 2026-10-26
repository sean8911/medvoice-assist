---
dataset_info:
  features:
  - name: diagnosis
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 120825750
    num_examples: 469195
  - name: test
    num_bytes: 5880848
    num_examples: 24695
  download_size: 28443487
  dataset_size: 126706598
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
license: apache-2.0
language:
- en
tags:
- Symptom
- Disease
- Diagnosis
- Healthcare
- medical
pretty_name: js
---
## Disclaimer
This dataset contains information on various diseases and their associated symptoms. While every effort has been made to ensure the accuracy of the data, it is possible that some information may be outdated, incomplete, or inaccurate. This dataset is intended for research and educational purposes only. Users are advised not to make medical decisions or take actions based solely on the information provided in this dataset. Always consult with a qualified healthcare professional for medical advice, diagnosis, and treatment. The creators of this dataset are not responsible for any consequences resulting from the use of the information contained herein.

## Acknowledgement
The underlanying dataset can be found [here](https://huggingface.co/datasets/fhai50032/SymptomsDisease246k). Thanks to [fhai50032](https://huggingface.co/fhai50032).