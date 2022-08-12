# Copyright (c) Microsoft. All rights reserved.
import torch
import logging
from .loss import stable_kl

logger = logging.getLogger(__name__)

def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) *  epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


class SmartPerturbation():
    def __init__(
        self,
        epsilon=1e-6,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p='inf',
        k=1,
        norm_level=0
    ):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon 
        # eta
        self.step_size = step_size
        self.k = k
        # sigma
        self.noise_var = noise_var 
        self.norm_p = norm_p
        self.norm_level = norm_level > 0


    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == 'l2':
            if sentence_level:
                direction = grad / (torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon)
            else:
                direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + self.epsilon)
        elif self.norm_p == 'l1':
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon)
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
                eff_direction = eff_grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, eff_direction

    def forward(
        self,
        model,
        logits,
        batch,
        global_step,
        calc_logits_keys,
    ):
        # init delta
        embed = model(batch, global_step, fwd_type='get_embs')
        noise = generate_noise(embed, batch['attention_mask'], epsilon=self.noise_var)
        for step in range(0, self.k):
            adv_logits, _ = model(batch, global_step, fwd_type='inputs_embeds', inputs_embeds=embed+noise, end_task_only=True)
            adv_loss = 0
            for k in calc_logits_keys:
                adv_loss += stable_kl(adv_logits[k], logits[k].detach(), reduce=False) 

            delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True, retain_graph=False)
            norm = delta_grad.norm()
            if (torch.isnan(norm) or torch.isinf(norm)):
                return {}, -1, -1
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            noise, eff_noise = self._norm_grad(delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level)
            noise = noise.detach()
            noise.requires_grad_()

            # save memory
            del adv_logits

        adv_logits, _ = model(batch, global_step, fwd_type='inputs_embeds', inputs_embeds=embed+noise, end_task_only=True)

        for k in list(adv_logits.keys()):
            if k in calc_logits_keys:
                adv_logits[f'adv_{k}'] = adv_logits.pop(k)
            else:
                adv_logits.pop(k)

        return adv_logits, embed.detach().abs().mean(), eff_noise.detach().abs().mean()
        #adv_lc = self.loss_map[task_id]
        #adv_loss = adv_lc(logits, adv_logits, ignore_index=-1)
        #return adv_loss, embed.detach().abs().mean(), eff_noise.detach().abs().mean()
