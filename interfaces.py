from collections import namedtuple
import dataclasses
from dataclasses import dataclass
from typing import NamedTuple
import torch

Dist = namedtuple('Dist', ('dist'))


RSSMState = namedtuple('RSSMState', ('belief', 'dist', 'sample'))
RSSMStateAction = namedtuple('RSSMStateAction', ('belief', 'dist', 'sample', 'action'))

SlimRSSMState = namedtuple('SlimRSSMState', ('belief', 'sample'))
SlimRSSMStateAction = namedtuple('SlimRSSMStateAction', ('belief', 'sample', 'action'))


def add_action(rssm_state, action):
    if isinstance(rssm_state, RSSMState):
        return RSSMStateAction(belief=rssm_state.belief, action=action, dist=rssm_state.dist, sample=rssm_state.sample)
    elif isinstance(rssm_state, SlimRSSMState):
        return SlimRSSMStateAction(belief=rssm_state.belief, action=action, sample=rssm_state.sample)
    else:
        assert False


@dataclass
class BatchData:
    obs: torch.Tensor

    def to(self, device):
        self.obs = self.obs.to(device)
        return self


@dataclass
class InteractiveBatchData(BatchData):
    act: torch.Tensor

    def to(self, device):
        BatchData.to(self, device)
        self.act = self.act.to(device)
        return self


# dreamer
Overshooting = namedtuple('Overshooting', ('actions', 'nonterminals', 'rewards', 'beliefs', 'prior_states', 'posterior_means', 'posterior_std_devs', 'masks'))