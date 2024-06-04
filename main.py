from transformers import get_linear_schedule_with_warmup
from utils import load_data, parse_args, set_seed, evaluate, get_sent_features
from sklearn.model_selection import train_test_split
import logging
import dgl
from model import SAGNN
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from sampler import NeighborSampler
import os


def main():
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        filename=f'./logs/{args.logging_name}.log',
        filemode='w',
        force=True
    )
    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        args.mode = 'cpu'
    device = torch.device(args.mode)
    set_seed(args.seed)

    # hete_g, texts, text_labels = load_data(args.network_files, args.text_files, args.hashtag_file)
    
    # pickle.dump(hete_g, open('./data/hete_g.pkl', 'wb'))
    # pickle.dump(texts, open('./data/texts.pkl', 'wb'))
    # pickle.dump(text_labels, open('./data/text_labels.pkl', 'wb'))

    hete_g = pickle.load(open('./data/hete_g.pkl', 'rb'))
    # texts = pickle.load(open('./data/texts.pkl', 'rb'))
    text_labels = pickle.load(open('./data/text_labels.pkl', 'rb'))

    homo_g = dgl.to_homogeneous(hete_g, store_type=True)

    num_texts = len(text_labels)
    train_idx = list(range(num_texts))
    train_idx, valid_idx = train_test_split(
        train_idx, test_size=args.valid_size, random_state=args.random_state, stratify=text_labels)
    valid_idx, test_idx = train_test_split(
        valid_idx, test_size=0.5, random_state=args.random_state, stratify=[text_labels[idx] for idx in valid_idx])

    text_labels = torch.tensor(text_labels).to(device)

    # text_features = get_sent_features(args.model_name, texts, device)
    # text_features = torch.tensor(text_features)
    # pickle.dump(text_features, open('./data/text_features.pkl', 'wb'))

    text_features = pickle.load(open('./data/text_features.pkl', 'rb')).to(device)

    model = SAGNN(in_size=text_features.size(1), hid_size=args.hidden_size, out_size=2, num_layers=args.num_layers).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    sampler = NeighborSampler(
        random_walk_length=args.random_walk_length, num_random_walks=args.num_random_walks, num_neighbors=args.num_neighbors, num_layers=args.num_layers)
    train_dataloader = dgl.dataloading.DataLoader(
        homo_g, train_idx, sampler, batch_size=args.batch_size, shuffle=True, device=device, drop_last=False)
    valid_dataloader = dgl.dataloading.DataLoader(
        homo_g, valid_idx, sampler, batch_size=args.batch_size, shuffle=False, device=device, drop_last=False)
    test_dataloader = dgl.dataloading.DataLoader(
        homo_g, test_idx, sampler, batch_size=args.batch_size, shuffle=False, device=device, drop_last=False)

    num_training_steps = args.num_epochs * len(train_dataloader)
    num_warmup_steps = num_training_steps * args.warmup_ratio
    no_decay = ['bias']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    with tqdm(total=num_training_steps) as pbar:
        for epoch in range(args.num_epochs):
            model.train()
            for i, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                features = text_features[blocks[0].srcdata[dgl.NID]]
                labels = text_labels[blocks[-1].dstdata[dgl.NID]]
                logits = model(blocks, features)
                loss = criterion(logits, labels)
                
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (i + 1) % args.gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    pbar.set_postfix_str(f'{loss.item():.4f}')
                    pbar.update(1)

            eval_loss, eval_accuracy, eval_f1, eval_auc = \
                evaluate(model, text_features, text_labels, valid_dataloader, criterion)

            print(f'epoch: {epoch+1:02}')
            print(f'\teval_loss: {eval_loss:.3f} | eval_accuracy: {eval_accuracy*100:.2f}% | eval_f1: {eval_f1*100:.2f}% | eval_auc: {eval_auc*100:.2f}%')
            logger.info(f'epoch: {epoch+1:02}')
            logger.info(f'\teval_loss: {eval_loss:.3f} | eval_accuracy: {eval_accuracy*100:.2f}% | eval_f1: {eval_f1*100:.2f}% | eval_auc: {eval_auc*100:.2f}%')
    
    test_loss, test_accuracy, test_f1, test_auc = evaluate(model, text_features, text_labels, test_dataloader, criterion)
    print(f'\ttest_loss: {test_loss:.3f} | test_accuracy: {test_accuracy*100:.2f}% | test_f1: {test_f1*100:.2f}% | test_auc: {test_auc*100:.2f}%')
    logger.info(f'\ttest_loss: {test_loss:.3f} | test_accuracy: {test_accuracy*100:.2f}% | test_f1: {test_f1*100:.2f}% | test_auc: {test_auc*100:.2f}%')
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, args.save_name))

if __name__ == '__main__':
    main()
