import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import os

from data_utils import sub12Dataset
from params import HParams
from diffusion import DDPM
from unet import ContextUnet


class Trainer:

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

        self.dataset = sub12Dataset(params.dataset_dir)
        self.dataloader = DataLoader(
            self.dataset,
            params.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=params.num_workers,
        )

        self.output_dir = params.output_dir
        self.log_dir = os.path.join(self.output_dir, "logs")
        self.ckpt_dir = os.path.join(self.output_dir, "ckpts")
        self.eval_dir = os.path.join(self.output_dir, "eval")
        self.eval_image_dir = os.path.join(self.eval_dir, "image")
        self.eval_gif_dir = os.path.join(self.eval_dir, "gif")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        if not os.path.exists(self.eval_image_dir):
            os.makedirs(self.eval_image_dir)
        if not os.path.exists(self.eval_gif_dir):
            os.makedirs(self.eval_gif_dir)

        self.writer = SummaryWriter(self.log_dir)

        self.optim = torch.optim.Adam(self.ddpm.parameters(), lr=params.lrate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, params.step_size, params.gamma
        )

        self.epoch = 0
        self.step = 0

        print("Initialized a trainer with following configs:")

        print(self.params.to_dict())

        num_paras = sum(p.numel() for p in self.ddpm.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_paras}")

    def train(self):
        best_loss = 1e10
        for ep in range(self.params.n_epoch):
            print(f"epoch {ep}")
            self.ddpm.train()
            loss_ema = None
            from tqdm import tqdm

            pbar = tqdm(self.dataloader)
            for x, c in pbar:
                self.optim.zero_grad()
                x = x.to(self.params.device)
                c = c.to(self.params.device)
                loss = self.ddpm(x, c)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                self.writer.add_scalar("loss", loss.item(), self.step)
                self.step += 1
                self.optim.step()

            self.scheduler.step()
            # for eval, save an image of currently generated samples (top rows)
            # followed by real images (bottom rows)
            self.ddpm.eval()
            with torch.no_grad():
                n_sample = 4 * self.params.n_classes
                for w_i, w in enumerate(self.params.ws_test):
                    x_gen, x_gen_store = self.ddpm.sample(
                        n_sample, (1, 28, 28), self.params.device, guide_w=w
                    )

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(self.params.device)
                    for k in range(self.params.n_classes):
                        for j in range(int(n_sample / self.params.n_classes)):
                            try:
                                idx = torch.squeeze((c == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k + (j * self.params.n_classes)] = x[idx]

                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all * -1 + 1, nrow=12)

                    save_image(
                        grid,
                        os.path.join(self.eval_image_dir, f"image_ep{ep}_w{w}.png"),
                    )
                    print(
                        "saved image at "
                        + os.path.join(self.eval_image_dir, f"image_ep{ep}_w{w}.png")
                    )

                    if ep % 5 == 0 or ep == int(self.params.n_epoch - 1):
                        # create gif of images evolving over time, based on x_gen_store
                        fig, axs = plt.subplots(
                            nrows=int(n_sample / self.params.n_classes),
                            ncols=self.params.n_classes,
                            sharex=True,
                            sharey=True,
                            figsize=(8, 3),
                        )

                        def animate_diff(i, x_gen_store):
                            print(
                                f"gif animating frame {i} of {x_gen_store.shape[0]}",
                                end="\r",
                            )
                            plots = []
                            for row in range(int(n_sample / self.params.n_classes)):
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
                            os.path.join(self.eval_gif_dir, f"gif_ep{ep}_w{w}.gif"),
                            dpi=100,
                            writer=PillowWriter(fps=5),
                        )
                        print(
                            "saved image at "
                            + os.path.join(self.eval_gif_dir, f"gif_ep{ep}_w{w}.gif")
                        )

            if ep % 5 == 0:
                is_best = False
                if loss_ema < best_loss:
                    best_loss = loss_ema
                    is_best = True
                torch.save(
                    self.ddpm.state_dict(),
                    os.path.join(self.ckpt_dir, f"model_{ep}.pth"),
                )
                print(
                    "saved model at " + os.path.join(self.ckpt_dir, f"model_{ep}.pth")
                )
                if is_best:
                    torch.save(
                        self.ddpm.state_dict(), os.path.join(self.ckpt_dir, "best.pth")
                    )
                    print(
                        "saved best model at "
                        + os.path.join(self.ckpt_dir, f"best.pth")
                    )


if __name__ == "__main__":
    trainer = Trainer(HParams())
    trainer.train()
