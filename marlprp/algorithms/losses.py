import torch
import torch.nn.functional as F


def ce_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        **kwargs
    ):

    bs = logp.size(0)
    if mask is not None:
        logp[mask] = 0
        entropy[mask] = 0
        denom = mask.view(bs, -1).logical_not().sum(1) + 1e-6

    # add entropy penalty 
    loss = torch.clamp(-logp - entropy_coef * entropy, min=0)

    if mask is not None:
        loss = loss.view(bs, -1).sum(1) / denom
    else:
        loss = loss.view(bs, -1).mean(1)

    return loss



def simple_listnet_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        **kwargs
    ):
    """ListNet inspired loss. This loss assumes that the logps are ordered corresponding to the
    order the machinens sampled actions during experience collection. The loss enforces a precendence
    by weighting machines that sampled first stronger. This makes intuitive sense, because these agents
    disrupted the sampling space of succeeding machines
    """
    bs = logp.size(0)
    # (bs, num_actions)
    logp = logp.view(bs, -1)
    
    y_true = torch.ones_like(logp)
    target_dist = F.softmax(y_true, dim=-1)
    if mask is not None:
        target_dist[mask] = 0
        entropy[mask] = 0  # for masked entries, simply add no penalty

    # (bs, actions)
    ce_loss = -torch.mul(target_dist, logp)
    loss = torch.clamp(ce_loss - entropy_coef * entropy, min=0)
        
    loss = loss.sum(-1)

    return loss



def listnet_loss(
        logp: torch.Tensor, 
        entropy: torch.Tensor, 
        mask: torch.Tensor = None, 
        entropy_coef: float = 0,
        alpha: float = 0.0,
        **kwargs
    ):
    """ListNet inspired loss. This loss assumes that the logps are ordered corresponding to the
    order the machinens sampled actions during experience collection. The loss enforces a precendence
    by weighting machines that sampled first stronger. This makes intuitive sense, because these agents
    disrupted the sampling space of succeeding machines
    """
    bs, N = logp.shape
    # (bs, num_actions)
    logp = logp.view(bs, -1)
    
    y_true = torch.ones_like(logp)
    ranks = torch.arange(1, N+1, device=logp.device).view(1, N).expand_as(logp)
    weights = torch.exp(-(alpha * (ranks - 1)))
    if mask is not None:
        # TODO is this sufficient?
        weights[mask] = 0
        entropy[mask] = 0  # for masked entries, simply add no penalty
    # (bs, num_actions)
    target_dist = (y_true * weights) / weights.sum(dim=-1, keepdims=True)
    # (bs, actions)
    ce_loss = -torch.mul(target_dist, logp)
    loss = torch.clamp(ce_loss - entropy_coef * entropy, min=0)
        
    loss = loss.sum(-1)

    return loss

def calc_adv_weights(adv: torch.Tensor, temp: float = 1.0, weight_clip: float = 20.0):

    adv_mean = adv.mean()
    adv_std = adv.std()

    norm_adv = (adv - adv_mean) / (adv_std + 1e-5)

    weights = torch.exp(norm_adv / temp)

    weights = torch.minimum(weights, torch.full_like(weights, fill_value=weight_clip))
    return weights

