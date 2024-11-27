import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

from params import HParams
from diffusion import DDPM
from unet import ContextUnet

CLASSES = [
    "airplane",
    "bus",
    "car",
    "fish",
    "guitar",
    "laptop",
    "pizza",
    "sea turtle",
    "star",
    "t-shirt",
    "The Eiffel Tower",
    "yoga",
]


class Sampler:
    def __init__(self, params: HParams):
        self.params = params

        self.unet = ContextUnet(params.in_channels, params.n_feat, params.n_classes)
        self.ddpm = DDPM(
            self.unet,
            betas=params.betas,
            n_T=params.n_T,
            device=params.device,
            drop_prob=params.drop_prob,
        ).to(params.device)

        load_path = os.path.join(params.output_dir, "ckpts", "best.pth")
        try:
            self.ddpm.load_state_dict(torch.load(load_path, map_location=params.device))
        except:
            raise ValueError("No checkpoint found. Please train a model first.")

    def sample(self, n_samples):
        self.ddpm.eval()
        with torch.no_grad():
            sample_dir = self.params.sample_dir
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            for w in self.params.ws_test:
                sample_w_dir = os.path.join(sample_dir, f"guidance_{w}")
                if not os.path.exists(sample_w_dir):
                    os.makedirs(sample_w_dir)

                sample_w_sub_dirs = [os.path.join(sample_w_dir, c) for c in CLASSES]
                for dir in sample_w_sub_dirs:
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                x_gen, x_gen_store = self.ddpm.sample(
                    n_samples * 12, (1, 28, 28), device=self.params.device, guide_w=w
                )

                for n_sample in range(n_samples):
                    for n_class in range(self.params.n_classes):
                        image = x_gen[n_sample * 12 + n_class]
                        image = image.detach().cpu().numpy()
                        image = np.resize(image, (28, 28))
                        image = cv2.resize(image, (256, 256))
                        plt.imsave(
                            os.path.join(sample_w_sub_dirs[n_class], f"{n_sample}.png"),
                            -image,
                            cmap="gray",
                            vmin=(-image).min(),
                            vmax=(-image).max(),
                        )

                print("All samples saved")

                grid = make_grid(x_gen * -1 + 1, nrow=12)
                save_image(
                    grid,
                    os.path.join(sample_w_dir, f"grid.png"),
                )
                fig, axs = plt.subplots(
                    nrows=n_samples,
                    ncols=self.params.n_classes,
                    sharex=True,
                    sharey=True,
                    figsize=(8, 3),
                )

                print("Image grid saved")

                def animate_diff(i, x_gen_store):
                    print(
                        f"gif animating frame {i} of {x_gen_store.shape[0]}",
                        end="\r",
                    )
                    plots = []
                    for row in range(n_samples):
                        for col in range(self.params.n_classes):
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                            plots.append(
                                axs[row, col].imshow(
                                    -x_gen_store[
                                        i,
                                        (row * self.params.n_classes) + col,
                                        0,
                                    ],
                                    cmap="gray",
                                    vmin=(-x_gen_store[i]).min(),
                                    vmax=(-x_gen_store[i]).max(),
                                )
                            )
                    return plots

                ani = FuncAnimation(
                    fig,
                    animate_diff,
                    fargs=[x_gen_store],
                    interval=200,
                    blit=False,
                    repeat=True,
                    frames=x_gen_store.shape[0],
                )
                ani.save(
                    os.path.join(sample_w_dir, f"process.gif"),
                    dpi=100,
                    writer=PillowWriter(fps=5),
                )
                print("Gif saved")
                # print(x_gen.shape)  # n_samples*12, 1, 28, 28
                # print(x_gen_store.shape)  # 12, n_samples*12, 1, 28, 28


if __name__ == "__main__":
    sampler = Sampler(HParams())
    sampler.sample(10)
