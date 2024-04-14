import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.utils import label_to_color, figure_to_array, PD_metric_to_ellipse, random_metric_field_generator
from evaluation.eval import Multi_Evaluation

from geometry import (
    relaxed_distortion_measure,
    get_pullbacked_Riemannian_metric,
    get_pushforwarded_Riemannian_metric,
    get_flattening_scores,
    get_pullback_metric,
)


class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "mse": loss.item(), "reg": 0.}

    def validation_step(self, x, **kwargs):
        recon = self(x)
        loss = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        return {"loss": loss.item()}

    def eval_step(self, dl, **kwargs):
        device = kwargs["device"]
        CN = []
        voR = []
        VP = []

        x_all = []
        z_all = []
        labels_all = []
        for x, labels in dl:
            z = self.encode(x.to(device))
            G = get_pushforwarded_Riemannian_metric(self.encode, x.view(x.shape[0], -1).to(device))
            CN.append(get_flattening_scores(G, mode="condition_number"))
            voR.append(get_flattening_scores(G, mode="variance"))
            VP.append(get_flattening_scores(G, mode="volume_preserving"))
            x_all.append(x)
            z_all.append(z)
            labels_all.append(labels)

        recon_all = self.decode(torch.cat(z_all).to(device))
        mse = ((recon_all - torch.cat(x_all).to(device)) ** 2).mean()

        voR = torch.cat(voR)
        CN = torch.cat(CN)
        VP = torch.cat(VP)

        G0 = random_metric_field_generator(len(VP), 2, 1, local_coordinates="exponential").to(device)
        voR0 = get_flattening_scores(G0, mode="variance")
        CN0 = get_flattening_scores(G0, mode="condition_number") - 1
        VP0 = get_flattening_scores(G0, mode="volume_preserving")

        voRrel = (voR / voR0).detach()
        CNrel = (CN / CN0).detach()
        VPrel = (VP / VP0).detach()

        # mean flattening scores
        mean_condition_number = CNrel.mean()
        mean_variance = voRrel.mean()
        mean_vp = VPrel.mean()

        results = {
            "CN_": mean_condition_number.item(),
            "voR_": mean_variance.item(),
            "VP_": mean_vp.item(),
            "mse_": mse.item(),
        }

        # generic eval metrics
        x_all = torch.cat(x_all).detach().cpu().numpy()
        z_all = torch.cat(z_all).detach().cpu().numpy()
        labels_all = torch.cat(labels_all)

        # TODO: ks here should be set in config
        ks = torch.arange(10, 210, 10)
        s = 201
        indices = torch.randperm(len(x_all))[:s]

        evaluator = Multi_Evaluation(dataloader=dl, model=self)
        ev_result = evaluator.get_multi_evals(
            x_all[indices].reshape(len(x_all[indices]), -1),
            z_all[indices],
            labels_all[indices],
            ks=ks,
        )

        for key, value in ev_result.items():
            results[key + "_"] = value

        return results

    def visualization_step(self, dl, **kwargs):
        device = kwargs["device"]

        # original iamge and recon image plot
        num_figures = 100
        num_each_axis = 10

        x = dl.dataset.data[torch.randperm(len(dl.dataset.data))[:num_figures]]
        recon = self.decode(self.encode(x.to(device)))
        x_img = make_grid(
            x.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1
        )
        recon_img = make_grid(
            recon.detach().cpu(), nrow=num_each_axis, value_range=(0, 1), pad_value=1
        )

        # 2d graph (latent sapce)
        num_points_for_each_class = 200
        num_G_plots_for_each_class = 20
        label_unique = torch.unique(dl.dataset.targets)
        z_ = []
        z_sampled_ = []
        label_ = []
        label_sampled_ = []
        G_ = []
        for label in label_unique:
            temp_data = dl.dataset.data[dl.dataset.targets == label][
                :num_points_for_each_class
            ].to(device)
            temp_z = self.encode(temp_data.to(device))
            z_sampled = temp_z[torch.randperm(len(temp_z))[:num_G_plots_for_each_class]]
            x_sampled = temp_data[torch.randperm(len(temp_data))[:num_G_plots_for_each_class]]
            G = get_pushforwarded_Riemannian_metric(self.encode, x_sampled.view(x_sampled.shape[0], -1))

            z_.append(temp_z)
            label_.append(label.repeat(temp_z.size(0)))
            z_sampled_.append(z_sampled)
            label_sampled_.append(label.repeat(z_sampled.size(0)))
            G_.append(G)

        z_ = torch.cat(z_, dim=0).detach().cpu().numpy()
        label_ = torch.cat(label_, dim=0).detach().cpu().numpy()
        color_ = label_to_color(label_)
        G_ = torch.cat(G_, dim=0).detach().cpu()
        z_sampled_ = torch.cat(z_sampled_, dim=0).detach().cpu().numpy()
        label_sampled_ = torch.cat(label_sampled_, dim=0).detach().cpu().numpy()
        color_sampled_ = label_to_color(label_sampled_)

        f = plt.figure()
        plt.title("Latent space embeddings with equidistant ellipses")
        z_scale = np.minimum(np.max(z_, axis=0), np.min(z_, axis=0))
        eig_mean = torch.svd(G_).S.mean().item()
        scale = 0.1 * z_scale * np.sqrt(eig_mean)
        alpha = 0.3
        for idx in range(len(z_sampled_)):
            e = PD_metric_to_ellipse(
                np.linalg.inv(G_[idx, :, :]),
                z_sampled_[idx, :],
                scale,
                fc=color_sampled_[idx, :] / 255.0,
                alpha=alpha,
            )
            plt.gca().add_artist(e)
        for label in label_unique:
            label = label.item()
            plt.scatter(
                z_[label_ == label, 0],
                z_[label_ == label, 1],
                c=color_[label_ == label] / 255,
                label=label,
            )
        plt.legend()
        plt.axis("equal")
        plt.close()

        return {
            "input@": torch.clip(x_img, min=0, max=1),
            "recon@": torch.clip(recon_img, min=0, max=1),
            "latent_space#": f,
        }


class IRAE(AE):
    def __init__(self, encoder, decoder, iso_reg=1.0, metric="identity"):
        super(IRAE, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        self.metric = metric

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encode(x)
        recon = self.decode(z)
        mse = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()

        iso_loss = relaxed_distortion_measure(
            self.encode, x.view(x.shape[0], -1), eta=None, metric=self.metric, reg="iso"
        )

        loss = mse + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "mse": mse.item(), "reg": iso_loss.item()}


class ConfAE(AE):
    def __init__(self, encoder, decoder, conf_reg=1.0, metric="identity", reg_type="conf"):
        super(ConfAE, self).__init__(encoder, decoder)
        self.conf_reg = conf_reg
        self.metric = metric
        self.reg_type = reg_type

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encode(x)
        recon = self.decode(z)
        mse = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()

        conf_loss = relaxed_distortion_measure(
            self.encode, x.view(x.shape[0], -1), eta=0.2, metric=self.metric, reg=self.reg_type
        )

        loss = mse + self.conf_reg * conf_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "mse": mse.item(), "reg": conf_loss.item()}


class GeomAE(AE):
    def __init__(self, encoder, decoder, geom_reg=1.0):
        super(GeomAE, self).__init__(encoder, decoder)
        self.geom_reg = geom_reg

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        x = x.view(x.shape[0], -1)
        z = self.encode(x)
        recon = self.decode(z)
        mse = ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()

        bs = len(x)
        x_perm = x[torch.randperm(bs)]

        eta = None
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2 * eta) - eta).unsqueeze(1).to(x).squeeze()
            x_augmented = alpha * x + (1 - alpha) * x_perm
        else:
            x_augmented = x
            z_augmented = z

        G = get_pushforwarded_Riemannian_metric(self.encode, x_augmented)
        # G = get_pullbacked_Riemannian_metric(self.decode, z_augmented)
        logdetG = torch.logdet(G)
        torch.nan_to_num(logdetG, nan=0.0, posinf=0.0, neginf=0.0)
        geom_loss = torch.var(logdetG)
        # geom_loss = torch.var(torch.log(
        #     torch.clip(torch.det(G), min=1.0e-4)
        #     ))

        loss = mse + self.geom_reg * geom_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "mse": mse.item(), "reg": geom_loss.item()}


class VAE(AE):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__(encoder, decoder)

    def encode(self, x):
        z = self.encoder(x)
        if len(z.size()) == 4:
            z = z.squeeze(2).squeeze(2)
        half_chan = int(z.shape[1] / 2)
        return z[:, :half_chan]

    def decode(self, z):
        return self.decoder(z)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu**2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)

        nll = -self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)

        loss = nll + kl_loss
        loss = loss.mean()
        nll = nll.mean()

        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            # "nll_": nll.item(),
            # "kl_loss_": kl_loss.mean(),
            # "sigma_": self.decoder.sigma.item(),
        }


class IRVAE(VAE):
    def __init__(
        self,
        encoder,
        decoder,
        iso_reg=1.0,
        metric="identity",
    ):
        super(IRVAE, self).__init__(encoder, decoder)
        self.iso_reg = iso_reg
        self.metric = metric

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)

        nll = -self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        iso_loss = relaxed_distortion_measure(
            self.encode, z_sample, eta=0.2, metric=self.metric, reg="iso"
        )

        loss = (nll + kl_loss).mean() + self.iso_reg * iso_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "iso_loss_": iso_loss.item()}


class ConfVAE(VAE):
    def __init__(
        self,
        encoder,
        decoder,
        conf_reg=1.0,
        metric="identity",
    ):
        super(ConfVAE, self).__init__(encoder, decoder)
        self.conf_reg = conf_reg
        self.metric = metric

    def train_step(self, x, optimizer, **kwargs):
        optimizer.zero_grad()
        z = self.encoder(x)
        z_sample = self.sample_latent(z)

        nll = -self.decoder.log_likelihood(x, z_sample)
        kl_loss = self.kl_loss(z)
        conf_loss = relaxed_distortion_measure(
            self.encode, z_sample, eta=0.2, metric=self.metric, reg="conf"
        )

        loss = (nll + kl_loss).mean() + self.conf_reg * conf_loss

        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "conf_loss_": conf_loss.item()}
