# SimCLR
A Simple Framework for Contrastive Learning of Visual Representations(SimCLR): Pytorch Implementation
SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. 
## What is the SimCLR?
<img src="https://user-images.githubusercontent.com/37788686/98641734-94386d00-236f-11eb-9266-e0a07882cb39.png" width="40%">

Like the picture above, model that output the same examples as "same" and the different examples as "different" is contrastive learning model.

<img src="https://user-images.githubusercontent.com/37788686/98642481-e3cb6880-2370-11eb-822d-64941dde44f9.png" width="80%">

SimCLR applies two random transformations to an image to get a pair of two augmented images x_i and x_j. Each image in that pair is passed through an encoder to get representations. Then a non-linear fully connected layer is applied to get representations z. The task is to maximize the similarity between these two representations z_i and z_j for the same image.

## SimCLR of Visual Representations
<img src="https://user-images.githubusercontent.com/37788686/98642661-242ae680-2371-11eb-8dc1-2212921ee281.png" width="60%">

Two separate data augmentation operators are sampled from the same family of augmentations(t~T and t'~T)and applied to each data example to obtain two correlated views. A base encoder network f(\*) and a projection head g(\*) are trained to maximize agreement using a contrastive loss. After training is completed, we throw away the projection head g(\*) and use encoder f(\*) and representation __h__ for downstream tasks.
## SimCLR Pseudocode
<img src="https://user-images.githubusercontent.com/37788686/98642798-5ccac000-2371-11eb-917f-c662da4e8086.png" width="50%">

## Results
|   | Acc |
| - | --- |
| Baseline(supervised) | 48.7% |
| No finetuning | 73.6% |
| Finetuning | 78% |

#### Requirements
  * numpy
  * torch
  * torchvision
  * opencv-python
  
## command
  - python3 main.py --epochs [epochs] --batch_size [B] --temperature [T] --out_dim [out_dim] --num_worker [N] --valid_size [val_size]

## Code Explanation
* model.py
```python
class SimCLR(nn.Module):

    def __init__(self, proj_dim, temperature):
        super(SimCLR, self).__init__()
        self.encoder = nn.ModuleList(list(resnet18(pretrained=True, progress=True).children())[:-1])
        self.temperature = temperature
        self.proj_head = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=2)
```
I uses resnet18 as baseline and MLP with one hidden layer as projection head. 
```python
    def pairwise_similarity(self, zi, zj):
        # zi, zj: [N, D]
        concat = torch.cat([zi, zj], dim=0)  # positive pair: (i, N+i)
        row = concat.unsqueeze(1)  # [2N, 1, D]
        col = concat.unsqueeze(0)  # [1, 2N, D]
        s = self.cos(row, col) / self.temperature  # [2N, 2N]

        return s
```
This function calculates similarity of each sample. zi and zj is a context vector from the projection head. I calculate cosine similarity.
```python
    def get_loss(self, s):
        N = int(s.size(0) / 2)

        s_ij = torch.diag(s, N)  # [N, 1]
        s_ji = torch.diag(s, -N)  # [N, 1]
        positive_pair = torch.cat([s_ij, s_ji], dim=0).reshape(2 * N, 1)

        negative_pair = s[self.negative_mask(N)].reshape(2 * N, -1)

        labels = torch.zeros(2 * N).to(device).long()
        logits = torch.cat([positive_pair, negative_pair], dim=1)

        loss = self.criterion(logits, labels)
        return loss
```
This function calculates NT-Xent(the normalized temperature-scalesd cross entropy loss). 
<img src="https://user-images.githubusercontent.com/37788686/98812929-be208b00-2466-11eb-97fc-f615ee198660.png" width="40%">
This loss function makes model to maximize positive pair and minimize negative pair.
* main.py
```python
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
```
I uses Adam optimizer(lr=3e-4, weight_decay=1e-5).
```python
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
```
At testing, I freezed all parameters of baseline(resnet18) and connect linear regression(fc(512->10)).
## Reference
1. A Simple Framework for Contrastive Learning of Visual Representations (https://arxiv.org/abs/2002.05709)
2. STL-10 Dataset (https://cs.stanford.edu/~acoates/stl10/)
3. https://amitness.com/2020/03/illustrated-simclr/
