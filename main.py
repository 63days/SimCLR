import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from data_loader import DataSetWrapper
from torch.utils.data import DataLoader
from torchvision import datasets
from model import *
from tqdm import tqdm


def main(args):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    ### Hyperparameters setting ###
    epochs = args.epochs
    batch_size = args.batch_size
    num_workers = args.num_worker
    T = args.temperature
    proj_dim = args.out_dim
    patience = args.patience

    '''
    Model-- ResNet18(encoder network) + MLP with one hidden layer(projection head)
    Loss -- NT-Xent Loss
    '''
    step = 0
    best_loss = float('inf')
    if not args.test:
        ### DataLoader ###
        dataset = DataSetWrapper(args.batch_size, args.num_worker, args.valid_size, input_shape=(96, 96, 3))
        train_loader, valid_loader = dataset.get_data_loaders()

        model = SimCLR(proj_dim=proj_dim, temperature=T)
        model.to(device)
        ### You may use below optimizer & scheduler ###
        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        for epoch in range(epochs):

            # TODO: Traninig
            model.train()
            pbar = tqdm(train_loader)
            train_losses = []
            val_losses = []
            for (xi, xj), _ in pbar:
                optimizer.zero_grad()
                xi, xj = xi.to(device), xj.to(device)

                loss = model(xi, xj)
                train_losses.append(loss.item())

                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.set_description('E: {:2d} L: {:.5f}'.format(epoch + 1, loss.item()))

            # TODO: Validation
            # You have to save the model using early stopping
            val_loss = []
            model.eval()
            with torch.no_grad():
                for (val_xi, val_xj), _ in valid_loader:
                    val_xi, val_xj = val_xi.to(device), val_xj.to(device)
                    loss = model(val_xi, val_xj)
                    val_loss.append(loss.item())
                    val_losses.append(loss.item())

                val_loss = sum(val_loss) / len(val_loss)

                if val_loss > best_loss:
                    step += 1
                    if step > patience:
                        print('Early stopped...')
                        return
                else:
                    step = 0
                    best_loss = val_loss
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        'loss': best_loss,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                    }, args.path)

            tl = sum(train_losses) / len(train_losses)
            vl = sum(val_losses) / len(val_losses)

            print('TL: {:.5f} VL: {:.5f}'.format(tl, vl))
    else:
        train_dataset = datasets.STL10('./data', split='train', download=True)
        test_dataset = datasets.STL10('./data', split='test', download=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=num_workers)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers)

        model = SimCLR(proj_dim=proj_dim, temperature=T)
        model.to(device)
        load_state = torch.load(args.path, map_location=device)
        model.load_state_dict(load_state['model_state_dict'])

        linear = LinearRegression(proj_dim, 10)

        optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)

        criterion = torch.nn.CrossEntropyLoss()

        for param in model.encoder.parameters():  # freeze baseline
            param.requires_grad = False

        for epoch in range(epochs):
            pbar = tqdm(train_loader)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)

                h = model.get_representation(x)
                pred = linear(h)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                pred = pred.argmax(1) + 1
                acc = (pred == y).sum().item() / y.size(0)
                print('Top-1 ACC: {:.1f}'.format(acc * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimCLR implementation")

    parser.add_argument(
        '--epochs',
        type=int,
        default=40)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5)
    parser.add_argument(
        '--out_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--num_worker',
        type=int,
        default=2)

    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05)

    parser.add_argument(
        '--patience',
        type=int,
        default=5)

    parser.add_argument(
        '--path',
        type=str,
        default='./ckpt/simclr.ckpt')

    parser.add_argument(
        '--test',
        action='store_true')

    args = parser.parse_args()
    main(args)




