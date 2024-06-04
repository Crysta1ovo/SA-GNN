# SA-GNN
Source code for ICWSM 2024 paper: [Online Social Behavior Enhanced Detection of Political Stances in Tweets](https://ojs.aaai.org/index.php/ICWSM/article/view/31383).
## Environments
- Python 3.8.8
- Cuda 11.4
## Requirements
- pytorch 1.12.0
- dgl 0.9.1
- transformers 4.30.2
- networkx 3.1
- sentence-transformers 2.2.2
## Preparation
Download the preprocessed data files from [here](https://drive.google.com/file/d/1Vy6XiSrfEeh5WXWnwn7LAGQ9R71uFJCD/view?usp=sharing) and put them into the `data` directory.
## Run
```
python main.py
```