import argparse
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import re
import dgl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_state', type=int, default=0)
    parser.add_argument(
        '--network_files', type=list, 
        default=[
            # './data/network/202008-network.lj', 
            # './data/network/202009-network.lj', 
            './data/network/202010-network.lj'
        ]
    )
    parser.add_argument(
        '--text_files', type=list, 
        default=[
            # './data/text/202008-text.lj', 
            # './data/text/202009-text.lj', 
            './data/text/202010-text.lj'
        ]
    )
    parser.add_argument('--hashtag_file', type=str, default='./data/hashtags.csv')
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--save_name', type=str, default='sagnn.pt')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--random_walk_length', type=int, default=2)
    parser.add_argument('--num_random_walks', type=int, default=10)
    parser.add_argument('--num_neighbors', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--max_norm', type=float, default=1.)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--logging_dir', type=str, default='logs')
    parser.add_argument('--logging_name', type=str, default='sagnn')

    args = parser.parse_args()

    return args
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_graph(network_files, text_files):
    data_dict = defaultdict(list)
    
    tids = []
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                data = eval(line)
                tid = data['tweet_id']
                tids.append(tid)
    tid_set = set(tids)

    t_mapping = {id: idx for idx, id in enumerate(tids)}
    u_mapping = {}
    for network_file in network_files:
        with open(network_file, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                data = eval(line)
                if 'r_tid' in data: # retweet
                    r_tid = data['r_tid']
                    if r_tid in tid_set:
                        uid = data['uid']
                        if uid not in u_mapping:
                            u_mapping[uid] = len(u_mapping)
                        t_idx = t_mapping[r_tid]
                        u_idx = u_mapping[uid]
                        data_dict[('user', 'retweet', 'tweet')].append((u_idx, t_idx))
                        data_dict[('tweet', 'retweeted_by', 'user')].append((t_idx, u_idx))
                else: # post
                    tid = data['tid']
                    if tid in tid_set:
                        uid = data['uid']
                        if uid not in u_mapping:
                            u_mapping[uid] = len(u_mapping)
                        t_idx = t_mapping[tid]
                        u_idx = u_mapping[uid]
                        data_dict[('user', 'post', 'tweet')].append((u_idx, t_idx))
                        data_dict[('tweet', 'posted_by', 'user')].append((t_idx, u_idx))
    
    g = dgl.heterograph(data_dict)
        
    return g

def load_data(network_files, text_files, hashtag_file):
    g = build_graph(network_files, text_files)
    label_hashtags = list(pd.read_csv(hashtag_file)['hashtag'].apply(lambda x: x.lower()).values)
    train_texts, train_labels = [], []
    for text_file in text_files:
        with open(text_file, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                data = eval(line)
                text = data['text']
                text = text.replace('#', ' #')
                text = text.strip() + ' '
                for hashtag in data['hashtags']:
                    if hashtag['text'].lower() in label_hashtags:
                        text = text.replace('#' + hashtag['text'] + ' ', ' ')
                text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', 'URL', text)
                text = re.sub(r'\s+', ' ', text)
                text = text.strip()
                train_texts.append(text)
                label = data['label']
                if label == 'JB':
                    train_labels.append(0)
                elif label == 'DT':
                    train_labels.append(1)
                else:
                    raise ValueError('Unexpected label')

    return g, train_texts, train_labels

def evaluate(model, text_features, text_labels, dataloader, criterion):
    model.eval()
    total_loss = 0.
    all_labels, all_preds = [], []
    for input_nodes, output_nodes, blocks in tqdm(dataloader):
        with torch.no_grad():
            features = text_features[blocks[0].srcdata[dgl.NID]]
            labels = text_labels[blocks[-1].dstdata[dgl.NID]]
            logits = model(blocks, features)
            loss = criterion(logits, labels)
            all_labels += labels.tolist()
            all_preds += logits.argmax(1).tolist()
            total_loss += loss.item()

    total_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    
    return total_loss, accuracy, f1, auc

def get_sent_features(model_name, sents, device):
    model = SentenceTransformer(model_name)
 
    return model.encode(sents, batch_size=32, device=device, show_progress_bar=True)
