import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torchvision.transforms import ToTensor
from tqdm import tqdm


@dataclass
class Config:
    n_embed: int
    n_head: int
    dropout: float
    n_layers: int
    codebook_size: int
    seq_length: int
    image_size: int
    patch_size: int

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embed % config.n_head == 0, "Embedding dimension must be divisible by the number of heads"
        self.attn = nn.Linear(config.n_embed, 3*config.n_embed, bias=False)
        self.proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.n_head = config.n_head

    def forward(self, x):
        batch_size, seq_length, n_embed = x.size()

        q, k, v = self.attn(x).split(n_embed, dim=2)
        q = q.view(batch_size, seq_length, self.n_head, n_embed // self.n_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, n_embed // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, n_embed // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, n_embed)
        y = self.resid_dropout(self.proj(y)) # TODO: maybe want to apply dropout before projection, check out https://arxiv.org/pdf/2302.06112.pdf
        return y

class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=False)
        self.proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.silu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embed)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ImageToPatchEmbeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        assert self.image_size % self.patch_size == 0, 'Image size is not divisible by patch size'

        self.projection = nn.Conv2d(3, config.n_embed, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.layers = nn.ModuleDict(dict(
            patch_embed = ImageToPatchEmbeddings(config),
            pos_embed = nn.Embedding(config.seq_length, config.n_embed),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.n_embed),
            linear_layer = nn.Linear(config.n_embed, config.codebook_size, bias=False),
            dropout = nn.Dropout(config.dropout),
        ))

        self.mask_embed = nn.Embedding(1, config.n_embed)
    
    def forward(self, x, mask=None):
        x = self.layers.patch_embed(x)
        if mask is not None:
            # remove all target mask embeddings and will those places with mask embedding
            x = x * ~mask.unsqueeze(-1) + self.mask_embed.weight * mask.unsqueeze(-1)
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)
        x = x + self.layers.pos_embed(pos)
        x = self.layers.dropout(x)
        for block in self.layers.blocks:
            x = block(x)
        x = self.layers.ln_f(x)
        logits = self.layers.linear_layer(x)
        return logits

if __name__ == "__main__":
    torch.set_default_device("cuda")

    config = Config(
        n_embed=768,
        n_head=12,
        dropout=0.2,
        n_layers=12,
        codebook_size=768,
        seq_length=(250//25)*(250//25),
        image_size=250,
        patch_size=25,
    )

    model = VisionTransformer(config)

    print(model)
    print(f"model has {sum(p.numel() for p in model.parameters()):,} parameters")

    images = []

    for root, dirs, files in os.walk("./lfw/lfw-deepfunneled/lfw-deepfunneled/"):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                image = Image.open(file_path)
                images.append(image)

    n_images = len(images)
    indices = np.random.permutation(n_images)

    train_size = int(n_images * 0.9)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_images = [images[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]

    batch_size = 32
    mask_prob = 0.2

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())
    transform = ToTensor()
    train_losses = []
    val_losses = []

    print("steps:", len(train_images) // batch_size)

    for i in range(0, len(train_images)-batch_size, batch_size):
        optimizer.zero_grad()

        train_batch_images = [transform(image).to("cuda") for image in train_images[i : i + batch_size]]
        train_batch = torch.stack(train_batch_images, dim=0)

        no_mask_output = model(train_batch)

        random_tensor = torch.rand(batch_size, config.seq_length)
        mask = random_tensor < mask_prob
        mask_output = model(train_batch, mask)

        train_loss = loss_fn(
            mask_output.view(-1, mask_output.size(-1)),
            torch.argmax(no_mask_output, dim=-1).view(-1)
        ) + loss_fn(
            no_mask_output.view(-1, no_mask_output.size(-1)),
            torch.argmax(mask_output, dim=-1).view(-1)
        )

        print(f"Train loss for batch {i // batch_size}: {train_loss.item()}")
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        # Validate every 10 samples
        if i % (10 * batch_size) == 0:
            with torch.no_grad():
                val_batch_images = [transform(image).to("cuda") for image in val_images[:batch_size]] # Take first batch from validation images
                val_batch = torch.stack(val_batch_images, dim=0)

                no_mask_output_val = model(val_batch)

                random_tensor_val = torch.rand(batch_size, config.seq_length)
                mask_val = random_tensor_val < mask_prob
                mask_output_val = model(val_batch, mask_val)

                val_loss = loss_fn(
                    mask_output_val.view(-1, mask_output_val.size(-1)),
                    torch.argmax(no_mask_output_val, dim=-1).view(-1)
                ) + loss_fn(
                    no_mask_output_val.view(-1, no_mask_output_val.size(-1)),
                    torch.argmax(mask_output_val, dim=-1).view(-1)
                )

                print(f"Validation loss: {val_loss.item()}")
                val_losses.append(val_loss.item())

    plt.plot(train_losses, label='Training loss')
    plt.plot(np.arange(0, len(train_losses), step=10), val_losses, label='Validation loss')
    plt.yscale("log")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'mim_transformer.pt')


    # for _ in tqdm(range(1)):
    #     random_tensor = torch.rand(batch_size, config.seq_length)
    #     mask = random_tensor < mask_prob
    #     print(mask)
    #     output = model(batch, mask)

    # print(output)
    # print(output.shape)

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # plt.figure(figsize=(10, 10))
    # sns.heatmap(mask.detach().cpu().numpy(), cmap="YlGnBu")
    # plt.ylabel("each image patch")
    # plt.xlabel("probability distribution over codebook tokens")
    # plt.show()