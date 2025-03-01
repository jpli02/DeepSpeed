# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
The file has been adapted from two fairscale files:
 (1) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/moe_layer.py
 (2) https://github.com/facebookresearch/fairscale/blob/master/fairscale/nn/moe/top2gate.py
 Git commit hash: 34df606902a240567a0d898037ece55c2f1336cf
 We retain the following license from the original files:
"""

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union
from moe.experts import Experts


import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.utils import groups
import argparse
import time

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

TOPK_GATE_TIMER = 'topk_gate'
MOE_TIMER = 'moe'
FIRST_ALLTOALL_TIMER = '1st_a2a'
SECOND_ALLTOALL_TIMER = '2nd_a2a'

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/deepspeedai/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device),
                                                      high=torch.tensor(1.0 + epsilon,
                                                                        device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def gumbel_rsample(shape: Tuple, device: torch.device) -> Tensor:
    gumbel = gumbel_map.get(device)
    if gumbel is None:
        one = torch.tensor(1.0, device=device)
        zero = torch.tensor(0.0, device=device)
        gumbel = torch.distributions.gumbel.Gumbel(zero, one).rsample  # type: ignore
        gumbel_map[device] = gumbel
    return gumbel(shape)


from deepspeed import comm as dist

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(output, input, group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'se,sec->sec':
        return a.unsqueeze(2) * b
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


# The following functions are extracted and scripted
# because otherwise during a torch.jit.trace, the non-Tensor
# values used in the calculations get recorded as constants.
# torch.jit.script coerces them into Tensors and preserves
# their dynamic shapes. This enables ONNX export.
# We can't script the entire top1gating function because it
# includes stateful caching logic which is incompatible with ONNX.


@torch.jit.script
def _capacity(gates: Tensor, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity


@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]


@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()


def top1gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               used_token: Tensor = None,
               noisy_gate_policy: Optional[str] = None,
               drop_tokens: bool = True,
               use_rts: bool = True,
               ep_group: Union[torch.distributed.ProcessGroup, None] = None,
               use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function

    gates = F.softmax(logits, dim=1)
    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # mask only used tokens
    if used_token is not None:
        mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to(logits.device)

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        # Communicate across expert processes to pick the maximum capacity.
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        # Make sure the capacity value does not exceed the number of tokens.
        capacity = min(new_capacity, torch.tensor(mask1.size(0)).to(new_capacity.device))

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def top2gating(logits: Tensor,
               capacity_factor: float,
               min_capacity: int,
               drop_tokens: bool = True,
               ep_group: Union[torch.distributed.ProcessGroup, None] = None,
               top2_2nd_expert_sampling: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    # print(f"gates shape : {gates.shape}")
    # print(f"indices1_s shape : {indices1_s.shape}")
    num_experts = int(gates.shape[1])
    # print(f"num_experts : {num_experts}")
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    # print(f"mask1 shape : {mask1.shape}")
    



    if top2_2nd_expert_sampling:
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits += gumbel_rsample(logits.shape, device=logits.device)

    # Replace top-expert with min value
    logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts

    # gating decisions
    exp_counts = torch.sum(mask1 + mask2, dim=0).detach().to(logits.device)

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))
        mask1 *= torch.lt(locations1, capacity)
        mask2 *= torch.lt(locations2, capacity)
    else:
        # Do not drop tokens - set capacity according to current expert assignments
        new_capacity = torch.max(exp_counts)
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        capacity = new_capacity

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


def topkgating(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    drop_tokens: bool = True,
    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
    drop_policy: str = "probs",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""

    # everything is in fp32 in this function
    # get topk gates
    top_gate, top_idx = torch.topk(logits, k=k, dim=1)
    # gating decisions
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])

    # get topk mask
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_idx, top_gate)

    mask = torch.zeros_like(gates, dtype=torch.bool).scatter_(1, top_idx, 1)

    exp_counts = torch.sum(mask, dim=0).detach().to(logits.device)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts / k

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
        # update mask and locations by capacity

        if drop_policy == 'probs':
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            mask = torch.logical_and(mask, capacity_mask)
            locations = torch.cumsum(mask, dim=0) - 1

        elif drop_policy == "position":
            locations = torch.cumsum(mask, dim=0) - 1
            mask *= torch.lt(locations, capacity)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

    else:
        # Do not drop tokens - set capacity according to current expert assignments
        new_capacity = torch.max(exp_counts)
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        capacity = new_capacity

    # normalize gates
    gates_masked = gates * mask
    gates_s = torch.sum(gates_masked, dim=-1, keepdim=True)
    denom_s = torch.clamp(gates_s, min=torch.finfo(gates_masked.dtype).eps)
    gates_masked = gates_masked / denom_s

    # dispatch_mask
    locations_sc = _one_hot_to_float((locations * mask), capacity)

    combine_weights = torch.einsum("se,sec->sec", gates_masked, locations_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
    """

    # wg: torch.nn.Linear

    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 eval_capacity_factor: float = 1.0,
                 min_capacity: int = 8,
                 noisy_gate_policy: Optional[str] = None,
                 drop_tokens: bool = True,
                 use_rts: bool = True,
                 ep_group: Union[torch.distributed.ProcessGroup, None] = None,
                 top2_2nd_expert_sampling: bool = True) -> None:
        super().__init__()

        # self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.ep_group = ep_group
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.noisy_gate_policy = noisy_gate_policy
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.top2_2nd_expert_sampling = top2_2nd_expert_sampling

    def _set_ep_group(self, ep_group):
        assert self.ep_group is None, f'Attempting to override an existing ep_group'
        self.ep_group = ep_group

    def forward(self,
                logits: torch.Tensor,
                use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        # if self.wall_clock_breakdown:
        #     self.timers(TOPK_GATE_TIMER).start()

        # input_fp32 = input.float()
        # # input jittering
        # if self.noisy_gate_policy == 'Jitter' and self.training:
        #     input_fp32 = multiplicative_jitter(input_fp32, device=input.device)
        # logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)

        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, self.ep_group, use_tutel)

        if self.k == 2:
            gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, self.drop_tokens, self.ep_group, self.top2_2nd_expert_sampling)
        else:
            gate_output = topkgating(logits, self.k,
                                     self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, self.drop_tokens, self.ep_group)

        # if self.wall_clock_breakdown:
        #     self.timers(TOPK_GATE_TIMER).stop()
        #     self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output



def run_deepspeed_all(top_k, exp_num, bs, seq_len, hid_dim, use_tutel):
    # input: [sl, bs, hs]
    input = torch.rand((seq_len, hid_dim), device='cuda')
    gate = TopKGate(hid_dim, exp_num, top_k)
    logits = torch.rand((seq_len, exp_num), device='cuda')

    # Implement Algorithm 2 from GShard paper.
    d_model = input[0].shape[-1]

    # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
    # Reshape into G groups so that each group can distribute tokens equally
    # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
    reshaped_input = input[0].reshape(-1, d_model)
    
    # gpu warm up
    for _ in range(10):
        for _ in range(10): 
            l_aux, combine_weights, dispatch_mask, exp_counts = gate(logits)
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)
        
        
    if use_tutel:
        # teset tutel: mem and speed
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        
        start_time = time.time()
        for _ in range(10):
            l_aux, C, E, indices_, locations_, gates_, exp_counts = gate(logits, True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)
            _tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            _tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = _tutel_dispatcher.encode(reshaped_input)
        end_time = time.time()
    
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        # summary
        memory_used = end_memory - start_memory
        peak_memory_used = peak_memory - start_memory
        print("---------- benchmarking tutel's gating kernel ----------")
        print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
        print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
        print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
    else:
        # test gshard: huge mem use as baseline
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()
        
        for _ in range(10): 
            l_aux, combine_weights, dispatch_mask, exp_counts = gate(logits)
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)
        
        end_time = time.time()
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        # summary
        memory_used = end_memory - start_memory
        peak_memory_used = peak_memory - start_memory
        print("---------- benchmarking deepspeed gshard's gating kernel ----------")
        print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
        print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
        print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
        
    # test for expert layer
    expert_mlp = torch.nn.Sequential(torch.nn.Linear(hid_dim, hid_dim), torch.nn.Linear(hid_dim, hid_dim))
    experts_layer = Experts(expert_mlp, exp_num)
    for _ in range(10):
        expert_output = experts_layer(dispatched_input)

    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated()
    start_time = time.time()
    
    for _ in range(10): 
        expert_output = experts_layer(dispatched_input)
        
    end_time = time.time()
    torch.cuda.synchronize()
    end_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    # summary
    memory_used = end_memory - start_memory
    peak_memory_used = peak_memory - start_memory
    print("---------- benchmarking deepspeed gshard's expert mlp layer ----------")
    print(f"Execution Time: {((end_time - start_time) / 10.0) * 1000:.6f} ms")
    print(f"Memory Used: {memory_used / 1024 ** 2:.2f} MB")
    print(f"Peak Memory Used: {peak_memory_used / 1024 ** 2:.2f} MB")
    
 
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark dMoE's indices_and_padded_bins.")
    parser.add_argument("--top_k", type=int, required=True, help="Top-k experts")
    parser.add_argument("--e", type=int, required=True, help="Number of experts")
    parser.add_argument("--bs", type=int, required=True, help="Batch size")
    parser.add_argument("--s", type=int, required=True, help="Sequence length")
    parser.add_argument("--hid_dim", type=int, required=True, help="Hidden dimension")
    parser.add_argument("--use_tutel", type=bool, required=False, help="Hidden dimension")
    
    args = parser.parse_args()

    print(f"Arguments: {args}")
    run_deepspeed_all(args.top_k, args.e, args.bs, args.s, args.hid_dim, args.use_tutel)
