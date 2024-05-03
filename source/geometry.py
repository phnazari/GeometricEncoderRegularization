import torch
import copy

from utils.utils import batch_jacobian
from utils.config import Config

config = Config()


def relaxed_distortion_measure(func, z, eta=0.2, metric='identity', create_graph=True, reg="iso"):
    if metric == 'identity':
        bs = len(z)
        z_perm = z[torch.randperm(bs)]
        if eta is not None:
            alpha = (torch.rand(bs) * (1 + 2*eta) - eta).unsqueeze(1).to(z)
            z_augmented = alpha*z + (1-alpha)*z_perm
        else:
            z_augmented = z
            
        if reg in ["iso", "conf", "conf-log"]:
            v = torch.randn(z.size()).to(z)
            Jv = torch.autograd.functional.jvp(
                func, z_augmented, v=v, create_graph=create_graph)[1]
            TrG = torch.sum(Jv.view(bs, -1)**2, dim=1)
            JTJv = (torch.autograd.functional.vjp(func, z_augmented,
                    v=Jv, create_graph=create_graph)[1]).view(bs, -1)
            TrG2 = torch.sum(JTJv**2, dim=1)
            if reg == "iso":
                return TrG2.mean()/(TrG**2).mean()
            elif reg == "conf":
                # for numerical stability add clip
                return (TrG2/torch.clip(TrG, min=1.0e-6)**2).mean()
            elif reg == "conf-log":
                return torch.logsumexp(torch.log(TrG2) - 2*torch.log(TrG), dim=0)
                # return torch.log((TrG2/TrG**2).mean())
            elif reg == "conf-log-inside":
                return (torch.log(TrG2) - 2*torch.log(TrG)).mean()
            else:
                raise NotImplementedError
        elif reg in ["conf-noapprox"]:
            J = jacobian_parallel(func, z_augmented)
            JTJ = torch.einsum('nij, nik -> njk', J, J)
            TrG2 = torch.einsum('nii -> n', JTJ@JTJ)
            TrG = torch.einsum('nii -> n', JTJ)
            if reg == "conf-noapprox":
                return (TrG2/torch.clip(TrG**2, min=1.0e-6)).mean()
    else:
        raise NotImplementedError


def get_flattening_scores(G, mode='condition_number'):
    if mode == 'condition_number':
        S = torch.svd(G).S
        scores = S.max(1).values/S.min(1).values - 1  # condition number & added the -1
    elif mode == 'variance':
        G_mean = torch.mean(G, dim=0, keepdim=True)
        A = torch.inverse(G_mean)@G
        scores = torch.sum(torch.log(torch.svd(A).S)**2, dim=1)
    elif mode == 'volume_preserving':
        logdetG = torch.log(torch.clip(torch.det(G), min=1.0e-8))
        mean = torch.mean(logdetG, dim=0, keepdim=True)
        scores = ((logdetG - mean)**2)
    else:
        pass
    return scores


def jacobian_parallel(func, inputs, v=None, create_graph=True, mode="rev"):
    #batch_size, z_dim = inputs.size()
    #if v is None:
    #    v = torch.eye(z_dim).unsqueeze(0).repeat(
    #        batch_size, 1, 1).view(-1, z_dim).to(inputs)
    #inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    #jac = (
    #    torch.autograd.functional.jvp(
    #        func, inputs, v=v, create_graph=create_graph
    #    )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    #)
    #return jac

    # J = torch.autograd.functional.jacobian(func, inputs, create_graph=True, strict=True, vectorize=False)
    # return J

    def flattened_immersion(x):
        recon = func(x)
        recon = recon.view(recon.shape[0], -1)
        return recon

    J = batch_jacobian(flattened_immersion, inputs, mode).squeeze()

    return J


def get_pushforwarded_Riemannian_metric(func, z):
    J = jacobian_parallel(func, z, v=None, mode="rev")
    G = torch.einsum('nij, nkj->nik', J, J)
    return G


def get_pullbacked_Riemannian_metric(func, z, mode="fwd"):
    J = jacobian_parallel(func, z, v=None, mode="fwd")
    G = torch.einsum('nij, nik->njk', J, J)
    return G


def get_Riemannian_metric(func, z, purpose, purpose_part=None):
    if purpose not in ["vis", "reg"]:
        raise NotImplementedError
    
    if purpose_part is None:
        purpose_part = config["part_of_ae"][purpose]

    if purpose_part == "encoder":
        return get_pushforwarded_Riemannian_metric(func, z)
    elif purpose_part == "decoder":
        return get_pullbacked_Riemannian_metric(func, z)
    else:
        raise NotImplementedError
