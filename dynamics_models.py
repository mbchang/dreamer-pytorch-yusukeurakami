from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

import distributions as dt
import interfaces as itf



class BasePredictiveModel(nn.Module):
    def __init__(self, state_dim):
        nn.Module.__init__(self)
        self.state_dim = state_dim

    def forward(self, state):
        raise NotImplementedError

class BaseDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        nn.Module.__init__(self)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, state, action):
        raise NotImplementedError

class SlotTransformation(nn.Module):
    def __init__(self, slot_dim, hid_dim):
        nn.Module.__init__(self)
        self.slot_dim = slot_dim
        self.hid_dim = hid_dim

    def forward(self, slots):
        raise NotImplementedError

class PairwiseInteraction(SlotTransformation):
    def __init__(self, slot_dim, hid_dim):
        SlotTransformation.__init__(self, slot_dim, hid_dim)
        self.pairwise_encoder = nn.Linear(2*self.slot_dim, self.hid_dim)
        self.pairwise_encoder_norm = nn.LayerNorm(self.hid_dim)
        self.pairwise_attention = nn.Linear(self.hid_dim, 1)
        self.pairwise_effect = nn.Linear(self.hid_dim, self.slot_dim)

    def forward(self, slots):
        slot_pairs = []
        for ii in range(slots.shape[1]):
            slot_pairs_this_entity = []
            for jj in range(slots.shape[1]):
                if ii != jj:
                    slot_pairs_this_entity.append(torch.cat((slots[:, ii], slots[:, jj]), dim=1))
            slot_pairs_this_entity = torch.stack(slot_pairs_this_entity, dim=1)  # (B, K-1, D)
            slot_pairs.append(slot_pairs_this_entity)
        slot_pairs = torch.stack(slot_pairs, dim=1)  # (B, K, K-1, D)

        pair_encodings = F.relu(self.pairwise_encoder_norm(self.pairwise_encoder(slot_pairs)))  # (B, K, K-1, D)
        pair_attn = torch.sigmoid(self.pairwise_attention(pair_encodings))  # (B, K, K-1, 1)
        pair_effects = self.pairwise_effect(pair_encodings)  # (B, K, K-1, D)
        new_slots = (pair_attn*pair_effects).sum(dim=2)  # (B, K, D)
        return new_slots

class AttentionInteraction(SlotTransformation):
    def __init__(self, slot_dim, hid_dim):
        SlotTransformation.__init__(self, slot_dim, hid_dim)
        self.scale = self.hid_dim ** -0.5

        self.to_q = nn.Linear(self.slot_dim, self.hid_dim, bias=False)
        self.to_k = nn.Linear(self.slot_dim, self.hid_dim, bias=False)
        self.to_v = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

    def forward(self, slots):
        k, v = self.to_k(slots), self.to_v(slots)  # (B, K_context, Dext), (B, K_context, Dext)
        q = self.to_q(slots)  # (B, K_self, Dext)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale  # torch.Size([B, K_self, K_context])
        attn = torch.sigmoid(dots)  # torch.Size([B, K_self, K_context])
        new_slots = torch.einsum('bjd,bij->bid', v, attn)  # torch.Size([B, K_self, Dext])
        return new_slots


def parallelize_rnn(rnn):
    def apply_rnn(x, h):
        assert x.dim() == 3
        assert h.dim() == 3
        b, k, _ = x.shape
        x = x.reshape(b*k, -1)
        h = h.reshape(b*k, -1)
        new_x = rnn(x, h)
        new_x = new_x.view(b, k, -1)
        return new_x
    return apply_rnn


class InteractionNetwork(SlotTransformation):
    def __init__(self, slot_dim, hid_dim, interaction_type):
        SlotTransformation.__init__(self, slot_dim, hid_dim)
        interaction_types = dict(
            attention=AttentionInteraction,
            pairwise=PairwiseInteraction)
        self.interaction = interaction_types[interaction_type](slot_dim=self.slot_dim, hid_dim=self.hid_dim)
        self.self_mlp = nn.Linear(self.slot_dim, self.slot_dim)
        self.both_norm = nn.LayerNorm(self.slot_dim)
        self.both_gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.norm_gru = nn.LayerNorm(self.slot_dim)

    def forward(self, slots):
        bsize, num_slots = slots.shape[:2]
        pairwise_hidden = self.interaction(slots)  # (B, K, Ds)
        self_hidden = self.self_mlp(slots)  # (B, K, Ds)
        update = F.relu(self.both_norm(pairwise_hidden+self_hidden))
        new_slots = self.norm_gru(parallelize_rnn(self.both_gru)(update, slots))  # (B, K, Ds)
        return new_slots

class SlotPredictiveModel(BasePredictiveModel):
    def __init__(self, state_dim, hid_dim, interaction_type):
        BasePredictiveModel.__init__(self, state_dim)
        self.hid_dim = hid_dim
        self.interaction_type = interaction_type
        self.encoder = nn.Linear(self.state_dim, self.state_dim)
        self.norm_encoder  = nn.LayerNorm(self.state_dim)
        self.interaction_model = InteractionNetwork(self.state_dim, self.hid_dim, self.interaction_type)

    def forward(self, slots):
        slots = F.relu(self.norm_encoder(self.encoder(slots)))  # (B, K, Dext)
        # slots = slots.detach()
        # # assert False
        new_slots = self.interaction_model(slots)
        return new_slots


class DiscreteActionSlotDynamicsModel(BaseDynamicsModel):
    def __init__(self, state_dim, action_dim, hid_dim, interaction_type):
        BaseDynamicsModel.__init__(self, state_dim, action_dim)
        self.hid_dim = hid_dim
        self.interaction_type = interaction_type
        self.action_encoder = nn.Embedding(self.action_dim, self.state_dim)
        self.encoder = nn.Linear(2*self.state_dim, self.state_dim)
        self.norm_encoder  = nn.LayerNorm(self.state_dim)
        self.interaction_model = InteractionNetwork(self.state_dim, self.hid_dim, self.interaction_type)

    def forward(self, slots, action):
        bsize, k, _  = slots.shape
        action = self.action_encoder(action)
        action = action.unsqueeze(1).repeat(1, k, 1)
        slots = F.relu(self.norm_encoder(self.encoder(torch.cat([slots, action], dim=-1))))  # (B, K, Dext)
        new_slots = self.interaction_model(slots)
        return new_slots


class SlotDynamicsModel(BaseDynamicsModel):
    def __init__(self, state_dim, action_dim, hid_dim, interaction_type):
        BaseDynamicsModel.__init__(self, state_dim, action_dim)
        self.hid_dim = hid_dim
        self.interaction_type = interaction_type
        self.action_encoder = nn.Linear(self.action_dim, self.state_dim)
        self.encoder = nn.Linear(2*self.state_dim, self.state_dim)
        self.norm_encoder  = nn.LayerNorm(self.state_dim)
        self.interaction_model = InteractionNetwork(self.state_dim, self.hid_dim, self.interaction_type)

    def forward(self, slots, action):
        bsize, k, _  = slots.shape
        action = self.action_encoder(action)
        action = action.unsqueeze(1).repeat(1, k, 1)
        slots = F.relu(self.norm_encoder(self.encoder(torch.cat([slots, action], dim=-1))))  # (B, K, Dext)
        new_slots = self.interaction_model(slots)
        return new_slots


class RSSMHead(nn.Module):
    def __init__(self, indim, outdim):
        nn.Module.__init__(self)
        self.indim = indim
        self.outdim = outdim
        self.gaussian_head = dt.GaussianHead(self.indim, self.outdim)

    def forward(self, x):
        dist = self.gaussian_head(x)
        rssm_state = itf.RSSMState(
            belief=x,
            dist=dist,
            sample=dist.rsample())
        return rssm_state

class RSSM(nn.Module):
    def __init__(self, stoch_dim, model):
        nn.Module.__init__(self)
        self.stoch_dim = stoch_dim
        self.model = model
        self.rssm_head = RSSMHead(model.state_dim, self.stoch_dim)

    def forward(self, rssm_state):
        if isinstance(self.model, BaseDynamicsModel):
            assert isinstance(rssm_state, itf.RSSMStateAction)
            next_belief = self.model(rssm_state.belief, rssm_state.action)
        elif isinstance(self.model, BasePredictiveModel):
            assert isinstance(rssm_state, itf.RSSMState)
            next_belief = self.model(rssm_state.belief)
        else:
            assert False
        next_rssm_state = self.rssm_head(next_belief)
        return next_rssm_state
