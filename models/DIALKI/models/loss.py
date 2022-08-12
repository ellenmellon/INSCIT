import torch
import torch.nn.functional as F


def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    if reduce:
        return (p* (rp - ry) * 2).sum() / bs
    else:
        return (p* (rp - ry) * 2).sum()


def sym_kl(input, target, reduction='batchmean'):
    input = input.float()
    target = target.float()
    left = F.kl_div(
        F.log_softmax(input, dim=-1, dtype=torch.float32),
        F.softmax(target.detach(), dim=-1, dtype=torch.float32),
        reduction=reduction,
    )

    right = F.kl_div(
            F.log_softmax(target, dim=-1, dtype=torch.float32),
            F.softmax(input.detach(), dim=-1, dtype=torch.float32),
            reduction=reduction,
        )
    loss = left + right
    return loss


def ns_sym_kl(input, target, reduction='batchmean'):
    input = input.float()
    target = target.float()
    loss = stable_kl(input, target.detach()) + \
            stable_kl(target, input.detach())
    return loss


def js(input, target, reduction='batchmean'):
    input = input.float()
    target = target.float()
    m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + \
        F.softmax(input.detach(), dim=-1, dtype=torch.float32)
    m = 0.5 * m
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), m, reduction=reduction) + \
        F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)
    return loss


def hl(input, target, reduction='batchmean'):
    input = input.float()
    target = target.float()
    si = F.softmax(target.detach(), dim=-1, dtype=torch.float32).sqrt_()
    st = F.softmax(input.detach(), dim=-1, dtype=torch.float32).sqrt_()
    loss = F.mse_loss(si, st)
    return loss


LOSS = {
    'sym_kl': sym_kl,
    'ns_sym_kl': ns_sym_kl,
    'js': js,
    'hl': hl,
}