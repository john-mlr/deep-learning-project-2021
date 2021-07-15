import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import argparse, os, socket
import pandas as pd
import numpy as np
from sklearn import metrics

from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch, label_to_levels
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import proba_to_label

import time

from braak_attention import BRAAK_attention, EmbeddingsDataset, BRAAK_no_attention


def main(gpu, args):
    # distribution stuff
    rank = args.nr * args.gpus + gpu
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    print(f"Process on cuda:{rank} initialized", flush=True)
    
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # Dataset Creation
    embed_dir = '/sc/arion/projects/tauomics/BYOL-tau-scoring/pretrained-resnet-embeddings'
    csv = '/sc/arion/projects/tauomics/BYOL-tau-scoring/cases2.csv'

    master_dataset = EmbeddingsDataset(csv, embed_dir)

    train_dataset, test_dataset = torch.utils.data.random_split(master_dataset, [int(len(master_dataset)*args.split), len(master_dataset) - int(len(master_dataset)*args.split)])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=True, # want to shuffle the dataset
                            num_workers=4) # number processes/CPUs to use

    test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=1,
                            num_workers=4) # number processes/CPUs to use

    if dist.get_rank() == 0:
        print(f'Datasets created with {len(train_dataset)} training and {len(test_dataset)} testing subjects.', flush=True)
        print('Creating and distributing model...', flush=True)

    LR = 1e-4 
    REG = 1e-5

    # Model creation and distibution to all GPUs
    model = BRAAK_no_attention(L=2048, size='big', dropout=False, n_classes=5)
    model = model.cuda(gpu)

    if args.restart:
        model.load_state_dict(torch.load(args.restart))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=REG)
    criterion = coral_loss

    dist.barrier()

    if dist.get_rank() == 0:
        print('Model created, beginning training.')

    best_val_rmse = 100 # just an impossibly high rmse
    training_data = {'train_loss' : [],
                     'train_rmse': [],
                     'train_acc': [],
                     'val_loss': [],
                     'val_rmse': [],
                     'val_acc': []}
    for epoch in range(args.start, args.epochs):
        print(f"=========================   EPOCH: {epoch+1}   =========================" , flush=True)
        train_loss, elapsed, train_rmse, train_acc = train(train_loader,
                                                model,
                                                criterion,
                                                optimizer,
                                                gpu, args)
        training_data['train_loss'].append(train_loss)
        training_data['train_rmse'].append(train_rmse)
        training_data['train_acc'].append(train_acc)
        print(f"Epoch {epoch+1} Train Set\tElapsed: {elapsed:.1f}\tLoss: {train_loss:.4f}\tRMSE: {train_rmse:.4f}\tAcc: {train_acc:.4f}", flush=True)
        val_loss, val_elapsed, val_rmse, val_acc = evaluate(test_loader,
                                                   model,
                                                   criterion,
                                                   gpu, args)
        print(f"Test Set\tElapsed: {val_elapsed:.1f}\tLoss: {val_loss:.4f}\tRMSE: {val_rmse:4f}\tAcc:{val_acc:.4f}", flush=True)
        training_data['val_loss'].append(val_loss)
        training_data['val_rmse'].append(val_rmse)
        training_data['val_acc'].append(val_acc)
        if val_rmse < best_val_rmse:
            print('New best val RMSE -- saving model.')
            torch.save(model.state_dict(), os.path.join(args.dump_path, "models", f"attention_model_epoch_{epoch+1}.pth"))
            best_val_rmse = val_rmse

    torch.save(training_data, os.path.join(args.dump_path, 'training_data.pth'))



def train(loader, model, criterion, optim, gpu, args):
    model.train()

    start_t = time.perf_counter()
    epoch_loss = 0
    truths = []
    preds = []
    for it, (feats, score, levels) in enumerate(loader):
        feats = torch.squeeze(feats)
        feats = feats.cuda(gpu)
        levels = levels.cuda(gpu)
        logits, _, Y_hat = model(feats)
        loss = criterion(logits.view(1, -1), levels.view(1, -1))
        epoch_loss += loss.item()
        loss /= args.batch_size
        loss.backward()
        truths.append(score.item())
        preds.append(Y_hat.item())

        if it % args.batch_size == 0:
            optim.step()
            model.zero_grad()

    preds = np.array(preds)
    truths = np.array(truths)
    epoch_loss /= it
    rmse = metrics.mean_squared_error(truths, preds, squared=False)
    accuracy = metrics.accuracy_score(truths, preds)
    elapsed_time = time.perf_counter() - start_t
    return epoch_loss, elapsed_time, rmse, accuracy


def evaluate(loader, model, criterion, gpu, args):
    model.train()

    start_t = time.perf_counter()
    epoch_loss = 0
    truths = []
    preds = []
    with torch.no_grad():
        for it, (feats, score, levels) in enumerate(loader):
            feats = torch.squeeze(feats)
            feats = feats.cuda(gpu)
            levels = levels.cuda(gpu)
            logits, _, Y_hat = model(feats)
            loss = criterion(logits.view(1, -1), levels.view(1, -1))
            epoch_loss += loss

            truths.append(score.item())
            preds.append(Y_hat.item())

        preds = np.array(preds)
        truths = np.array(truths)
        epoch_loss /= it
        rmse = metrics.mean_squared_error(truths, preds, squared=False)
        accuracy = metrics.accuracy_score(truths, preds)
        elapsed_time = time.perf_counter() - start_t
    return epoch_loss, elapsed_time, rmse, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-e', '--epochs', default=200, type=int,
                        help='number of pretraining epochs')
    parser.add_argument('-ee', '-eval_epochs', default=500, type=int,
                        help='number of linear eval epochs')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='pseudo batch size for training, will update model every batch_size iterations.')
    parser.add_argument('--dump_path', default='.', type=str,
                        help='path to store experiment information in.')
    parser.add_argument('--split', default=.75, type=float,
                        help='Fraction of dataset to use for training set')
    parser.add_argument('--restart', default='', type=str,
                        help='Model path to restart from.')
    parser.add_argument('--start', default=0, type=int,
                        help='Epoch that training has been started at.')
    
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    os.environ['MASTER_PORT'] = '55000'
    os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())

    print('Master port and address established. Spawning processes...', flush=True)

    mp.spawn(main, nprocs=args.gpus, args=(args,))