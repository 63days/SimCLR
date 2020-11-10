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

## Reference
1. A Simple Framework for Contrastive Learning of Visual Representations (https://arxiv.org/abs/2002.05709)
2. STL-10 Dataset (https://cs.stanford.edu/~acoates/stl10/)
3. https://amitness.com/2020/03/illustrated-simclr/
