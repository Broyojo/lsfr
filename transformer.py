import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (
    AdaptiveAvgPool1d,
    Conv2d,
    Dropout,
    Embedding,
    Linear,
    Module,
    ModuleList,
    Parameter,
)


class MultiHeadSelfAttention(Module):
    def __init__(self, d_embed, n_head, dropout):
        super().__init__()
        assert d_embed % n_head == 0, "Embedding dimension must be divisible by the number of heads"
        self.attn = Linear(d_embed, 3 * d_embed, bias=False)
        self.proj = Linear(d_embed, d_embed, bias=False)
        self.resid_dropout = Dropout(dropout)
        self.dropout = dropout
        self.n_head = n_head

    def forward(self, x):
        batch_size, seq_length, d_embed = x.size()

        q, k, v = self.attn(x).split(d_embed, dim=2)
        q = q.view(batch_size, seq_length, self.n_head, d_embed // self.n_head).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.n_head, d_embed // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.n_head, d_embed // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_length, d_embed)
        y = self.resid_dropout(self.proj(y)) # TODO: maybe want to apply dropout before projection, check out https://arxiv.org/pdf/2302.06112.pdf
        return y

class MLP(Module):
    def __init__(self, d_embed, dropout):
        super().__init__()
        self.fc = Linear(d_embed, 4 * d_embed, bias=False)
        self.proj = Linear(4 * d_embed, d_embed, bias=False)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        x = self.fc(x)
        x = F.silu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class RMSNorm(Module):
    def __init__(self, size, dim=-1, eps=1e-5):
        super().__init__()
        self.scale = Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed

class Block(Module):
    def __init__(self, d_embed, n_head, dropout):
        super().__init__()
        self.norm_1 = RMSNorm(d_embed)
        self.attn = MultiHeadSelfAttention(d_embed, n_head, dropout)
        self.norm_2 = RMSNorm(d_embed)
        self.mlp = MLP(d_embed, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x

class PatchEmbedding(Module):
    def __init__(self, img_size, patch_size, d_embed, in_chans=3):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.proj = Conv2d(in_chans, d_embed, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):     # B x C x I x I
        x = self.proj(x)      # B x D x I/P x I/P
        x = x.flatten(2)      # B x D x L
        x = x.transpose(1, 2) # B x L x D
        return x

class ViTForFeatureExtraction(Module):
    def __init__(self, d_embed, n_layers, n_head, dropout, img_size, patch_size, n_features):
        super().__init__()
        seq_length = (img_size // patch_size)**2
        self.patch_embed = PatchEmbedding(img_size, patch_size, d_embed)
        self.pos_embed = Embedding(seq_length, d_embed)
        self.dropout = Dropout(dropout)
        self.blocks = ModuleList([Block(d_embed, n_head, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d_embed)
        self.avg_pool = AdaptiveAvgPool1d(1)
        self.head = Linear(d_embed, n_features, bias=False)
        
    def forward(self, x):
        x = self.patch_embed(x)
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)
        x = x + self.pos_embed(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x) # B x L x D
        x = self.avg_pool(x.transpose(1, 2)) # B x D x 1
        # x = x.reshape(1, -1) <-- maybe will cause overfitting and also too many parameters
        x = x.flatten(1) # B x D
        x = self.head(x) # 1 x F
        return x

class ViTForImageReconstruction(Module):
    def __init__(self, d_embed, n_layers, n_head, dropout, img_size, patch_size):
        super().__init__()
        self.seq_length = (img_size // patch_size)**2
        self.patch_embed = PatchEmbedding(img_size, patch_size, d_embed)
        self.pos_embed = Embedding(self.seq_length, d_embed)
        self.dropout = Dropout(dropout)
        self.blocks = ModuleList([Block(d_embed, n_head, dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d_embed)
        self.unflatten = nn.Unflatten(2, (img_size // patch_size, img_size // patch_size))
        self.head = nn.ConvTranspose2d(d_embed, 3, kernel_size=patch_size, stride=patch_size)
        self.mask_embed = nn.Embedding(1, d_embed)

    def forward(self, x, mask=None):
        x = self.patch_embed(x)
        if mask is not None:
            x = x * ~mask.unsqueeze(-1) + self.mask_embed.weight * mask.unsqueeze(-1)
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)
        x = x + self.pos_embed(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x) # B x L x D

        # B x L x D --> B x C x I x I
        x = x.transpose(2, 1)
        x = self.unflatten(x)
        x = self.head(x)
        return F.sigmoid(x) # make sure values are normalized between 0 and 1 for RGB

if __name__ == "__main__":
    import os
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from torch.nn import L1Loss
    from torch.optim import AdamW
    from torchvision import transforms
    from tqdm import tqdm

    torch.set_default_device("cuda")

    images = []

    for root, dirs, files in tqdm(os.walk("./lfw/lfw-deepfunneled/lfw-deepfunneled/")):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as image:
                    images.append(np.array(image))

    to_tensor = transforms.ToTensor()
    to_image = transforms.ToPILImage()

    train_size = 0.9
    split_index = int(len(images) * train_size)

    train_images = images[:split_index]
    eval_images = images[split_index:]

    model = ViTForImageReconstruction(d_embed=768, n_layers=12, n_head=12, dropout=0.0, img_size=250, patch_size=25)
    print(model)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = AdamW(model.parameters())
    loss_fn = L1Loss()

    batch_size = 64
    n_epochs = 100
    mask_prob = 0.7

    train_losses = []
    eval_losses = []

    total_iter = 0

    for n in range(n_epochs):
        random.shuffle(train_images)
        random.shuffle(eval_images)
        for i in range(0, len(train_images)-batch_size, batch_size):
            optimizer.zero_grad()
            batch = torch.stack([to_tensor(image) for image in train_images[i : i + batch_size]]).cuda()
            random_tensor = torch.rand(batch_size, model.seq_length)
            mask = random_tensor < mask_prob
            outputs = model(batch, mask)
            # masked_outputs = outputs[mask]
            # masked_batch = batch[mask]
            loss = loss_fn(outputs, batch)
            print(f"Epoch {n}: Train loss for batch {i // batch_size}/{len(train_images)//batch_size}: {loss.item()}")
            loss.backward()
            optimizer.step()

            if (i // batch_size) % 10 == 0:
                image = to_tensor(eval_images[0]).unsqueeze(0).cuda()  # original image
                random_tensor = torch.rand(1, model.seq_length)
                mask = random_tensor < mask_prob
                with torch.no_grad():
                    output = model(image, mask)

                output_image = to_image(output.cpu()[0])

                # construct pixel_mask using mask
                pixel_mask = torch.ones((3, 250, 250))  # 3 channel, same height and width as your images
                for idx, val in enumerate(mask[0]):
                    if val.item():
                        row = (idx // 10) * 25
                        col = (idx % 10) * 25
                        pixel_mask[:, row: row + 25, col: col + 25] = 0

                # mask is in range [0, 1], we need to convert it to [0, 255] for PIL
                pixel_mask = pixel_mask.mul(255).byte()
                pixel_mask_image = to_image(pixel_mask)

                # load original image
                original_image = to_image(image.cpu()[0])

                # Composite the original image with the mask
                composite_image = Image.composite(original_image, pixel_mask_image, pixel_mask_image.convert('L'))

                # create a new image big enough to hold the original, masked, and output images side by side
                combined_image = Image.new('RGB', (output_image.width * 3, output_image.height))

                # paste the images into the combined image
                combined_image.paste(to_image(eval_images[0]), (0, 0))
                combined_image.paste(composite_image, (output_image.width, 0))
                combined_image.paste(output_image, (output_image.width * 2, 0))

                combined_image.save(f"output-{str(total_iter).zfill(5)}.png")
                # image = to_tensor(eval_images[0]).unsqueeze(0).cuda()
                # random_tensor = torch.rand(1, model.seq_length)
                # mask = random_tensor < mask_prob
                # with torch.no_grad():
                #     output = model(image, mask)
                # image = to_image(output[0])

                # pixel_mask = torch.ones(250, 250)

                # print(mask)

                # for i, val in enumerate(mask[0]):
                #     if val.item():
                #         row = (i % 10) * 25
                #         col = (i // 10) *25
                #         pixel_mask[row : row + 25, col : col + 25] = 0

                # pixel_mask = to_image(pixel_mask)

                # pixel_mask.save(f"output-{str(total_iter).zfill(5)}.png")

            if (i // batch_size) % 50 == 0:
                train_losses.append(loss.item())
                losses = []
                for i in range(0, len(eval_images)-batch_size, batch_size):
                    batch = torch.stack([to_tensor(image) for image in eval_images[i : i + batch_size]]).cuda()
                    random_tensor = torch.rand(batch_size, model.seq_length)
                    mask = random_tensor < mask_prob
                    with torch.no_grad():
                        outputs = model(batch, mask)
                    # masked_outputs = outputs[mask]
                    # masked_batch = batch[mask]
                    loss = loss_fn(outputs, batch)
                    print(f"Epoch {n}: Eval loss for batch {i // batch_size}/{len(eval_images)// batch_size}: {loss.item()}")
                    losses.append(loss.item())
                mean_loss = sum(losses)/len(losses)
                print(f"Epoch {n}: Mean eval loss for batch {i // batch_size}/{len(eval_images)// batch_size}: {mean_loss}")
                eval_losses.append(mean_loss)
            
            total_iter += 1
        torch.save(model.state_dict(), "image_reconstruction_transformer_masked_2.pt")

    plt.plot(train_losses, label="train loss")
    plt.plot(eval_losses, label="eval loss")
    plt.xlabel("time step")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.show()