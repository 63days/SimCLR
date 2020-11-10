from model import *
import torch
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import models
import torchvision.transforms as transforms
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.STL10('./data', split='train', download=True, transform=preprocess)
    test_dataset = datasets.STL10('./data', split='test', download=True, transform=preprocess)
    #train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=preprocess)
    #test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    if args.baseline:
        module = list(models.resnet18(pretrained=True).children())[:-1]
        model = nn.Sequential(*module)
        fc = nn.Linear(512, 10)
        model.to(device)
        fc.to(device)

        optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=1e-5)
    else:
        model = SimCLR(proj_dim=args.out_dim, temperature=args.temperature)
        model.to(device)

        load_state = torch.load(args.path, map_location=device)
        model.load_state_dict(load_state['model_state_dict'])

        linear = LinearRegression(512, 10)  # h vector: 512, z vector: 256
        linear.to(device)

        if args.freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(linear.parameters(), 3e-4, weight_decay=1e-5)
        else:
            optimizer = torch.optim.Adam(list(model.parameters()) + list(linear.parameters()), 3e-4, weight_decay=1e-5)

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        for (x, y) in pbar:
            x, y = x.to(device), y.to(device)

            if args.baseline:
                output = model(x)
                output = output.squeeze()
                pred = fc(output)
                loss = criterion(pred, y)
            else:
                h = model.get_representation(x)
                pred = linear(h)
                loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('E: {:2d} L: {:.5f}'.format(epoch + 1, loss))

    model.eval()

    accs = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if args.baseline:
                output = model(x)
                output = output.squeeze()
                pred = fc(output)
            else:
                h = model.get_representation(x)
                pred = linear(h)

            loss = criterion(pred, y)
            pred = pred.argmax(1)
            acc = (pred == y).sum().item() / y.size(0)
            accs.append(acc)
            print('L: {:.5f} ACC: {:.1f}'.format(loss.item(), acc * 100))
        print('Top-1 Acc: {:.1f}%'.format(sum(accs) / len(accs) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR Test')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '--out_dim',
        type=int,
        default=256
    )
    parser.add_argument(
        '--path',
        type=str,
        default='./ckpt/simclr.ckpt'
    )
    parser.add_argument(
        '--freeze',
        action='store_true'
    )
    parser.add_argument(
        '--baseline',
        action='store_true'
    )
    args = parser.parse_args()
    main(args)