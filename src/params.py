import torch


class HParams:
    def __init__(self):
        # Dataset
        self.dataset_dir = "subclass12"
        self.batch_size = 256
        self.num_workers = 5

        # Diffusion
        self.n_T = 100
        self.betas = (1e-4, 0.02)
        self.drop_prob = 0.1

        # UNet
        self.in_channels = 1
        self.n_feat = 128
        self.n_classes = 12

        # Training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lrate = 2e-4
        self.n_epoch = 50
        self.step_size = 12
        self.gamma = 0.1
        self.save_model = True
        self.output_dir = "outputs"

        # Sampling
        self.ws_test = [0.5, 2.0, 5.0]  # strength of generative guidance
        self.sample_dir = "samples"

    def to_dict(self):
        return {
            "dataset_dir": self.dataset_dir,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "n_T": self.n_T,
            "betas": self.betas,
            "drop_prob": self.drop_prob,
            "in_channels": self.in_channels,
            "n_feat": self.n_feat,
            "n_classes": self.n_classes,
            "device": self.device,
            "lrate": self.lrate,
            "n_epoch": self.n_epoch,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "save_model": self.save_model,
            "output_dir": self.output_dir,
            "ws_test": self.ws_test,
            "sample_dir": self.sample_dir,
        }
