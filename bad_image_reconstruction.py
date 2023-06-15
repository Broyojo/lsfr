import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import save_image

from transformer import Block, Config, RMSNorm


class ImageToPatchEmbeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        assert config.image_size % config.patch_size == 0, 'Image size is not divisible by patch size'

        self.projection = nn.Conv2d(3, config.n_embed, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        return x

class TokensToImage(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        assert config.image_size % config.patch_size == 0, 'Image size is not divisible by patch size'

        self.deprojection = nn.ConvTranspose2d(config.n_embed, 3, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # Swap seq_length and n_embed dimensions
        x = x.view(x.shape[0], x.shape[1], config.image_size // config.patch_size, config.image_size // config.patch_size)  # Reshape into 2D image format
        x = self.deprojection(x)
        return x

class GumbelSoftmax(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp

    def forward(self, logits):
        return F.gumbel_softmax(logits, tau=self.temp, hard=True)

class ImageReconstructionVisionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.layers = nn.ModuleDict(dict(
            patch_embed = ImageToPatchEmbeddings(config),
            pos_embed = nn.Embedding(config.seq_length, config.n_embed),
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = RMSNorm(config.n_embed),
            linear_layer = nn.Linear(config.n_embed, config.codebook_size, bias=False),
            dropout = nn.Dropout(config.dropout),
            gumbel_softmax = GumbelSoftmax(temp=0.01),
            tokens_to_image = TokensToImage(config),
        ))

    def forward(self, x):
        x = self.layers.patch_embed(x)
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)
        x = x + self.layers.pos_embed(pos)
        x = self.layers.dropout(x)
        for block in self.layers.blocks:
            x = block(x)
        x = self.layers.ln_f(x)
        logits = self.layers.linear_layer(x)
        tokens = self.layers.gumbel_softmax(logits)
        x_reconstructed = self.layers.tokens_to_image(tokens)
        return x_reconstructed

if __name__ == "__main__":
    torch.set_default_device("cuda")

    TRAIN = False

    config = Config(
        n_embed=768,
        n_head=12,
        dropout=0.5,
        n_layers=12,
        codebook_size=768,
        seq_length=(250//25)*(250//25),
        image_size=250,
        patch_size=25,
    )

    model = ImageReconstructionVisionTransformer(config)

    print(model)
    print(f"model has {sum(p.numel() for p in model.parameters()):,} parameters")

    images = []

    for root, dirs, files in os.walk("./lfw/lfw-deepfunneled/lfw-deepfunneled/"):
        for file in files:
            if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as image:
                    images.append(np.array(image))

    n_images = len(images)
    indices = np.random.permutation(n_images)

    train_size = int(n_images * 0.9)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    train_images = [images[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]

    if TRAIN:
        batch_size = 32

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        transform = ToTensor()
        train_losses = []
        val_losses = []
        num_epochs = 3

        print("steps:", len(train_images) // batch_size * num_epochs)

        # Initialize a list to hold validation batch indices
        val_batches = []

        for _ in range(num_epochs):
            for i in range(0, len(train_images)-batch_size, batch_size):
                optimizer.zero_grad()

                train_batch_images = [transform(image).to("cuda") for image in train_images[i : i + batch_size]]
                train_batch = torch.stack(train_batch_images, dim=0)

                output = model(train_batch)

                train_loss = loss_fn(output, train_batch)

                print(f"Train loss for batch {i // batch_size}: {train_loss.item()}")
                train_losses.append(train_loss.item())
                train_loss.backward()
                optimizer.step()

                # Validate every 10 samples
                if i % (10 * batch_size) == 0:
                    with torch.no_grad():
                        val_batch_images = [transform(image).to("cuda") for image in val_images[:batch_size]] # Take first batch from validation images
                        val_batch = torch.stack(val_batch_images, dim=0)

                        val_output = model(val_batch)

                        val_loss = loss_fn(val_output, val_batch)

                        print(f"Validation loss: {val_loss.item()}")
                        val_losses.append(val_loss.item())
                        # Record the batch index for validation
                        val_batches.append(i // batch_size)

        plt.plot(train_losses, label='Training loss')
        plt.plot(val_batches, val_losses, label='Validation loss') # Use the actual batch indices for x-values
        plt.yscale("log")
        plt.legend()
        plt.show()

        torch.save(model.state_dict(), 'image_reconstruction_transformer.pt')
    else:
        model.load_state_dict(torch.load("./image_reconstruction_transformer.pt"))

        model.eval()

        # Apply necessary transformations to the image
        transform = ToTensor()
        image = images[0]
        image = transform(image).unsqueeze(0).to("cuda")

        # Forward pass through the model
        with torch.no_grad():
            output = model(image)

        # Normalize output to [0, 1] range
        output_normalized = (output + 1) / 2.0

        # Move to CPU and detach from the computation graph
        output_normalized = output_normalized.detach().cpu().squeeze(0)

        # Save the image to disk
        save_image(output_normalized, "output.png")




# if __name__ == "__main__":
#     torch.set_default_device("cuda")

#     config = Config(
#         n_embed=768,
#         n_head=12,
#         dropout=0.2,
#         n_layers=12,
#         codebook_size=768,
#         seq_length=(250//25)*(250//25),
#         image_size=250,
#         patch_size=25,
#     )

#     model = ImageReconstructionVisionTransformer(config)

#     print(model)

#     images = []

#     for root, dirs, files in os.walk("./lfw/lfw-deepfunneled/lfw-deepfunneled/"):
#         for file in files:
#             if file.endswith(".jpg"):
#                 file_path = os.path.join(root, file)
#                 with Image.open(file_path) as image:
#                     images.append(np.array(image))

#     transform = ToTensor()
#     image = transform(images[0]).to("cuda").unsqueeze(0)

#     print(image.shape)

#     output = model(image)

#     print(output)
#     print(output.shape)

#     to_pil_image = ToPILImage()

#     output_normalized = (output + 1) / 2.0

#     # Detach and move to cpu
#     output_normalized = output_normalized.detach().cpu()

#     # Loop over all images in the batch
#     for i in range(output_normalized.shape[0]):
#         # Select image
#         img_normalized = output_normalized[i]
        
#         # Convert to PIL Image
#         img_pil = to_pil_image(img_normalized)
        
#         # Display image
#         img_pil.show()
