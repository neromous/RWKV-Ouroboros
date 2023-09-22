########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import gc, math, os
from random import randint
from typing import List, Optional

import numpy as np
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed.runtime.lr_schedules
import wandb

from torch.utils.cpp_extension import load
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# Script dir for various files
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CUDA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../cuda"))

########################################################################################################
# JIT / torch compile special handling
########################################################################################################

# Currently the features we need for torch compile, is avaliable only in
# 2.1 nightly build (and is expected to be in 2.1 official release)
#
# However because the nightly build torch.compile behaviour has been unstable
# the versioning code to check for enabling toch compile will not be used,
# until we confirm a stable version of torch.compile
from packaging import version
def is_torch_version_above(required_version):
    torch_version = version.parse(torch.__version__.split('+')[0])
    return torch_version >= version.parse(required_version)
IS_TORCH_2_1 = is_torch_version_above("2.0.9999")

# Get the JIT / torch compile option flags from the environment
RWKV_JIT_ON        = os.getenv("RWKV_JIT_ON", "1").lower() in ("1", "true", "yes")
RWKV_TORCH_COMPILE = os.getenv("RWKV_TORCH_COMPILE", f"0").lower() in ("1", "true", "yes")
RWKV_TORCH_RUN_MODE = None

# We enable JITMod*/Function when supporting torch.jit
# We use TorchCompile* when supporting torch compile
# based on the current runtime settings
if RWKV_TORCH_COMPILE:
    RWKV_TORCH_RUN_MODE = "torch-compile"

    JITModClass  = nn.Module
    JITModMethod = lambda x: x
    JITFunction  = lambda x: x

    # PS: i have tried mode="max-autotune", and mode="reduce-overhead", however they crash
    #     for now (8th July 2023). I may introduce them in the future once they are stable
    #
    #     Additionally, torch.compile has issues with the pytorch.lightning module directly
    # ---

    # We generally have 2 major options, either we use torch.compile
    # onto the key top level functions (train, val, test, predict, etc)
    # and let the compiler handle all the decision making on how to optimize
    #
    # However this was found to basically just match JIT level of performance exactly
    # ---
    # TCompileMax          = lambda x: x
    # TCompileBaseline     = lambda x: torch.compile(x, fullgraph=False)

    # Alternatively, we can perform a much more aggressive optimization on critical functions
    # that we know are compatible with torch.compile(fullgraph=True) - which provides the highest
    # level of optimization possible with torch.compile
    # ---
    TCompileMax        = lambda x: torch.compile(x, fullgraph=True)
    TCompileBaseline   = lambda x: x

    # ---
    # Because torch.compile is expected to change overtime, the two options should
    # be tested every now and then, for any performance changes
    #
    # and we should switch over to the broaded automated approach if its "faster"
    # ---

    # Used to wrap functions which are **not** torch.compile compatible
    TCompileDisable    = torch._dynamo.disable

    # The following are known warnings in the nightly build, that can be safely ignored for stable release
    #
    # `torch._inductor.utils: [WARNING] DeviceCopy in input program`
    # https://discuss.pytorch.org/t/what-can-cause-warning-devicecopy-in-input-program/175566

elif RWKV_JIT_ON:
    RWKV_TORCH_RUN_MODE = "torch-jit"
    JITModClass  = torch.jit.ScriptModule
    JITModMethod = torch.jit.script_method
    JITFunction  = torch.jit.script

    TCompileMax        = lambda x: x
    TCompileBaseline   = lambda x: x
    TCompileDisable    = lambda x: x
else:
    RWKV_TORCH_RUN_MODE = "torch-native"
    JITModClass  = nn.Module
    JITModMethod = lambda x: x
    JITFunction  = lambda x: x

    TCompileMax        = lambda x: x
    TCompileBaseline   = lambda x: x
    TCompileDisable    = lambda x: x

print(f"[RWKV.model] Running RWKV model using '{RWKV_TORCH_RUN_MODE}' with torch '{torch.__version__}'")

# ---
# Isolating out known operations that **does not work** with torch.compile
# and wrapping them within a torch._dynamo.disable, this is required to get
# the baseline torc.compile to work
# ---

@TCompileDisable
def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

@TCompileDisable
def wkv_op(time_decay, time_first, k, v, wkv_state):
    return torch.ops.rwkv.wkv(time_decay, time_first, k, v, wkv_state)

########################################################################################################
# RWKV: State Blocks
########################################################################################################

class TimeMixState:

    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:

    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    # @ TCompileMax (no difference)
    @staticmethod
    def create(N, B, C, device, dtype):
        result = BlockStateList.empty(N, B, C, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    # @ TCompileMax (no difference)
    @staticmethod
    def empty(N, B, C, device, dtype):
        wkv_states = torch.empty((N, B, C, 3),
                                 device=device,
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

class RWKV_TimeMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, dim_att):
        super().__init__()

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # fancy time_decay
            decay_speed = torch.ones(dim_att)
            for h in range(dim_att):
                decay_speed[h] = -5 + 8 * (h /
                                           (dim_att - 1))**(0.7 +
                                                            1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1
                                   for i in range(dim_att)]) * 0.5
            self.time_first = nn.Parameter(
                torch.ones(dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(
                torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(
                torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_att, bias=False)
        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)

    @JITModMethod
    @TCompileMax
    def _forward_kvsr(self, x, last_state: TimeMixState):
        # Mix x with the previous timestep to produce xk, xv, xr
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        # Enforce bf16 type for kv, as this can be mis init
        # when being called directly via inference
        if k.dtype != torch.bfloat16:
            k = k.to(torch.bfloat16)
        if v.dtype != torch.bfloat16:
            v = v.to(torch.bfloat16)

        return k, v, sr

    @JITModMethod
    @TCompileMax
    def _forward_out(self, sr, y, x_l, new_wkv_state):
        return self.output(sr * y), TimeMixState(x_l, new_wkv_state)

    @JITModMethod
    @TCompileBaseline
    def forward(self, x, last_state: TimeMixState):
        k, v, sr = self._forward_kvsr(x, last_state)

        # Enforce bf16 for self.time_first
        # as this can be mis init when being called directly via inference
        if self.time_first.dtype != torch.bfloat16:
            self.time_first = self.time_first.to(torch.bfloat16)

        # Perform the WKV op via cuda code
        y, new_wkv_state = wkv_op(self.time_decay, self.time_first,
                                  k, v, last_state.wkv_state)
        return self._forward_out(sr, y, x[:, -1], new_wkv_state)


########################################################################################################


class RWKV_ChannelMix(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(dim_ffn, n_embd, bias=False)

    @JITModMethod
    @TCompileMax
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]),
                          dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return (torch.sigmoid(self.receptance(xr)) * kv,
                ChannelMixState(x[:, -1]))


########################################################################################################
# The RWKV Model blocks
########################################################################################################

class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)
        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )
        x = x + att_out
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = x + ffn_out
        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y, token_amount, currentMask):
        # Currently (8th July 2023), save_for_backward, causes an issue with
        # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
        #
        # Due to L2Wrap being a major hotspot, we should monitor this for future support.
        # so that once its resolved, we can include the L2Wrap step in the torch.compile path
        #
        # See also:
        # - checkpointed_step
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        ctx.currentMask = currentMask
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        token_amount = ctx.token_amount
        # to encourage the logits to be close to 0
        factor = 1e-4 / token_amount
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        gy = gy * ctx.currentMask[:, None][None, :]
        return (grad_output, gy, None, None)

########################################################################################################
# Static optimized functions
########################################################################################################

# @ TCompileMax (no speed improvement)
# def F_cross_entropy_reduction_none_optimized(logits, targets):
#     return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")

########################################################################################################
# Core RWKV module
########################################################################################################
class RWKV(nn.Module):

    def __init__(self,
                 # Model file path to load from
                 load_model: str,
                 # Model size settings, which we either
                 # "auto detect", or use the user specified settings
                 n_embd: int = -1,
                 n_layer: int = -1,
                 vocab_size: int = -1,
                 # Context length size for the model
                 ctx_len: int = 2048,
                 # Context length schedule
                 ctx_len_cutoffs: List[int] = [],
                 ctx_len_warmup_steps: List[int] = [],
                 # Learning rate schedule
                 # use only target_lr_init / lr_init
                 # to configure a constant learning rate
                 lr_init: float = -1.0,
                 lr_final: float = -1.0,
                 lr_period: int = -1,
                 lr_period_type: str = 'epoch',
                 # Adam optimizer settings
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 adam_eps: float = 1.0e-08,
                 weight_decay: float = 0.01,
                 warmup_steps: int = -1,
                 # loss bias start
                 position_loss_bias: float = 1.0,
                 position_loss_bias_in_validation: bool = False,
                 # Backprop settings
                 grad_cp: bool = True,
                 bptt_learning: bool = True,
                 bptt_learning_range: int = -1,
                 bptt_truncated_learning: bool = False,
                 layerwise_lr: bool =True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 substep_cuda_cache_clear: bool = False,
                 substep_logging: bool = False,
                 torch_set_float32_matmul_precision:str = 'high',
                 accumulate_grad_batches = 1,
                 num_devices=1
                 ):

        # Lets save everything in one shot
        # (this is used for wandb logging)
        self.setup_args = locals()
        del self.setup_args["self"]
        del self.setup_args["__class__"]

        # Setup the parent class
        super().__init__()

        # Load the model, unless its the special ".//<#|=@%!$init_model$!%@=|#>//." path
        # which is reserved to be used with the `init_model.py`
        #
        # We intentionally used several filesystem illegal characters, to ensure it
        # is not accidentally used by the user for a real file
        model_weights = None
        model_keys = None
        if load_model != ".//<#|=@%!$init_model$!%@=|#>//.":
            # Print the loading event
            print(f"[RWKV.model]: Preloading model from '{load_model}'")

            # Check if the load_model path exists, and is a file
            if not os.path.isfile(load_model):
                raise ValueError(f"load_model file '{load_model}' does not exist")

            # Load the model weights
            model_weights = torch.load(load_model, map_location='cpu')

            # Get the model keys
            model_keys = list(model_weights.keys())

        # Lets compute the model various sizes, if they are not provided
        if n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            n_layer = max_block_id + 1

        if n_embd < 0:
            n_embd = model_weights['head.weight'].shape[1]

        if vocab_size < 0:
            vocab_size = model_weights['head.weight'].shape[0]

        # Save the various other params for later
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_period = lr_period
        self.lr_period_type = lr_period_type
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.bptt_learning = bptt_learning
        self.bptt_learning_range = bptt_learning_range
        self.bptt_truncated_learning = bptt_truncated_learning
        self.substep_cuda_cache_clear = substep_cuda_cache_clear
        self.substep_logging = substep_logging
        self.accumulate_grad_batches =  accumulate_grad_batches
        self.num_devices = num_devices

        # Save the position loss params
        self.position_loss_bias = position_loss_bias
        self.position_loss_bias_in_validation = position_loss_bias_in_validation

        dim_att = dim_att or n_embd
        dim_ffn = dim_ffn or n_embd * 4

        if torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(torch_set_float32_matmul_precision)

        self.emb = nn.Embedding(vocab_size, n_embd)

        load(name=f"wkv_{self.ctx_len}_bf16",
             sources=[
                os.path.join(CUDA_DIR, "wkv_op_bf16.cpp"),
                os.path.join(CUDA_DIR, "wkv_cuda_bf16.cu")
            ],
             verbose=True,
             extra_cflags=["-std=c++17", "-O3", f"-DTmax={self.ctx_len}"],
             extra_cuda_cflags=[
                 "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
                 "--use_fast_math", "-O3", "-Xptxas -O3",
                 "--extra-device-vectorization", f"-DTmax={self.ctx_len}"
             ],
             is_python_module=False)

        self.blocks = nn.ModuleList([
            Block(i, n_layer, n_embd, dim_att, dim_ffn) for i in tqdm(range(n_layer))
        ])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # load the state, and GC the original cpu copy
        if model_weights != None:
            # Print the loading event
            print(f"[RWKV.model]: Loading model weights ( L{self.n_layer}-D{self.n_embd}-V{self.vocab_size} )")

            self.load_state_dict(model_weights)
            del model_weights
            gc.collect()

        print(f"[RWKV.model]: Finished initial model load")

    def get_optimizers(self):

        #print(f"[RWKV.model][rank={self.trainer.local_rank}] Configuring optimizer ...")

        # Get the learning rate used for the optimizer
        lr_init = self.lr_init
        lr_final = self.lr_final
        # If the final learning rate is not specified, use the initial learning rate
        if lr_final < 0:
            lr_final = self.lr_init

        # Log the learning rate, and various other parameters
        #if self.trainer.local_rank == 0:
        if True:
            # Add the important notes, for informing users of common gotchas
            print((
                "#\n"
                "# RWKV lighting_trainer.py important notes \n"
                "# https://github.com/RWKV/RWKV-infctx-trainer \n"
                "#\n"
                "# - Ensure your host is not running cuda 12.0 (use either 11.8, or >=12.1), as this is known to have freeze issues\n"
                "# - The terms used in wandb / the progress bar can be confusing, see the github README.md for beter clarifications\n"
                "# - When resuming from checkpoint, the estimated time is inaccurate\n"
                "#"
            ))

            lr_init_e = "{:.3e}".format(lr_init)
            lr_final_e = "{:.3e}".format(lr_final)
            print(f"\n[RWKV.model] Configuring optimizer with\n"+
                  f"    - lr_init:  {lr_init_e} ({lr_init})\n"+
                  f"    - lr_final: {lr_final_e} ({lr_final})\n")

            # Get the setup args
            model_args = dict(self.setup_args)
            model_args["__lr_init"] = lr_init
            model_args["__lr_final"] = lr_final

            # Update WANDB
            if wandb.run is not None:
                wandb.config.update({ "model": model_args })

        # Setup layerwise learning rate
        if self.layerwise_lr:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    lr_1x.add(n)
                elif "time_decay" in n:
                    lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "lr": 3.0 * lr_init
                },
            ]
        else:
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters()],
                    "weight_decay": 0.0
                },
            ]

        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode=False,
                                         weight_decay=self.weight_decay,
                                         amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups,
                                  lr=lr_init,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.adam_eps,
                                  bias_correction=True,
                                  adam_w_mode=False,
                                  weight_decay=self.weight_decay,
                                  amsgrad=False)
        lr_scheduler = None
        if self.warmup_steps > 0:
            lr_scheduler = deepspeed.runtime.lr_schedules.WarmupLR(
                optimizer,
                warmup_min_lr=0.2 * self.lr_init,
                warmup_max_lr=self.lr_init,
                warmup_num_steps=self.warmup_steps,
                warmup_type='linear')



        return optimizer,lr_scheduler
#        return optim_groups

    # We have to compute the number of steps per epoch ourselves
    # as this value is not provided directly by pytorch lightning
    # https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-1501597319

    @property
    def deepspeed_offload(self) -> bool:
        return True

    @property
    def deepspeed_stage(self) -> int:
        return -1

    @TCompileBaseline
    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor = None,
                last_wkv_states: torch.Tensor = None):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        new_states = BlockStateList.empty(self.n_layer, B, self.n_embd,
                                          x.device, x.dtype)

        # last_shift_states can be None, when we are performing direct inference
        if last_shift_states is None:
            cur_bs_list = BlockStateList.create(
                self.n_layer, B,
                self.n_embd,
                x.device, x.dtype
            )
        else:
            cur_bs_list = BlockStateList(last_shift_states, last_wkv_states)

        # Avoid using the zip operation, as torch.compile throws an exception on it
        # with `zip not reconized as a valid function`
        # ---
        # for i, (block, last_state) in enumerate(
        #         zip(self.blocks,
        #             BlockStateList(last_shift_states, last_wkv_states))):
        # ---
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.grad_cp:
                x, new_state = deepspeed_checkpoint(
                    block, x, last_state)
            else:
                x, new_state = block(x, last_state)
            new_states[i] = new_state

        x = self.ln_out(x)

        x = self.head(x)

        return x, new_states.shift_states, new_states.wkv_states

    #
    # Custom overwrite of manual_backwards operation, to skip the "manual_backwards"
    # safety check, so we can perform manual backward operation step, while using
    # the default trainer loop. This is modified from the original code found here:
    # https://github.com/Lightning-AI/lightning/blob/37c244f94be365496def82870b22c2faf0ab889e/src/lightning/pytorch/core/module.py#L999
    #
    # ---
    #
    # This allow us to avoid disabling the "automatic_optimization" flag
    #
    # Which would have been required to do "segmented learning", or "Backpropagation Through Time"
    # where we would need to implement manual optimization as per
    # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    #
    # Otherwise an error will be thrown if we call `self.manual_backward`
    #
    # However this would mean that we would need to do a full reimplementation
    # of several features that were handled by the automatic optimization.
    # - accumulate_grad_batches
    # - gradient_clip_val
    # - logging behaviour
    # - distributed training co-ordination
    # - (And probably other features that I am not aware of)
    #
    # So this is a hacky work around, to avoid reimplementing all of the above.
    #
    # From the current code implementatiion, it seem like this is blocked only by
    # automatic_optimization flag - and has no adverse side effect otherwise
    # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule.manual_backward
    #
    # If anyone have a better idea, let me know
    # (have experimented with, reimplementing the above, but it is not trivial, unfortunately)
    #
    def manual_backward(self,model_engine, loss: torch.Tensor, *args, **kwargs):
        # loss.backward(retain_graph = True)
        model_engine.backward(loss,retain_graph = True)
        # if self._fabric:
        #     self._fabric.backward(loss, *args, **kwargs)
        # else:
        #     # self._verify_is_manual_optimization("manual_backward")
        #     self.trainer.strategy.backward(loss, None, *args, **kwargs)

    #
    # Main compute_loss function, this is called by the trainer loop
    #
    def compute_loss(self,batch, is_training_run=True,model_engine=None):
        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        ori_seq_mask = batch['attention_mask']

        # Check if attent mask is set, if not initialize it
        if ori_seq_mask is None or ori_seq_mask.ndim != 2:
            ori_seq_mask = torch.ones_like(seq[:, 1:])

        # Get the starting and ending loss bias
        loss_bias_start = self.position_loss_bias
        loss_bias_end   = 2.0 - loss_bias_start

        # total_mask_sum
        total_mask_sum = torch.sum(ori_seq_mask)

        # Skip loss bias calculation, if loss_bias_start is 1.0
        if loss_bias_start == 1.0 or (is_training_run == False and self.position_loss_bias_in_validation == False):
            seq_mask = ori_seq_mask
        else:
            # Lets get a linear multiplier for the loss bias
            # seq_mask_sum = torch.sum(ori_seq_mask)
            bias_mask = torch.linspace(loss_bias_start, loss_bias_end, int(total_mask_sum.item()), device=ori_seq_mask.device)

            # Boolean flag of seq_mask > 0
            seq_mask_index = ori_seq_mask[0] > 0

            # Apply the bias mask only to positive seq_mask values
            final_mask = torch.zeros(ori_seq_mask.shape[1], device=ori_seq_mask.device)
            final_mask[seq_mask_index] = ori_seq_mask[0][seq_mask_index] * bias_mask

            # And save it as seq_mask
            seq_mask = final_mask.unsqueeze(0)

        # Perform cutoff for training run
        if is_training_run:
            prev_step = 0

            # Avoid using the zip operation, as torch.compile throws an exception on it
            # with `zip not reconized as a valid function`
            # ---
            # for step, len_cut in zip(self.ctx_len_warmup_steps,
            #                          self.ctx_len_cutoffs):
            # ---
            for i in range(min(len(self.ctx_len_warmup_steps), len(self.ctx_len_cutoffs))):
                step = self.ctx_len_warmup_steps[i]
                len_cut = self.ctx_len_cutoffs[i]

                if prev_step <= self.global_step < step and len_cut < seq.shape[
                        1] - 1:
                    pos = randint(0, seq.shape[1] - len_cut - 1)

                    # Original
                    # seq = seq[:, pos:pos + len_cut + 1]

                    # Changed to use masking for prefix cutoff (i do not know if this makes sense)
                    seq = seq[:, :pos + len_cut + 1]
                    seq_mask = seq_mask[:, :pos + len_cut + 1]
                    # Set the attention mask to 0 for the skipped tokens
                    seq_mask[:, :pos] = 0
                    break
                prev_step = step

        do_bptt_learning = self.bptt_learning and is_training_run
        idx, targets = seq[:, :-1], seq[:, 1:]

        B, T = idx.shape
        C = self.n_embd

        # If total_mask_sum, we skip, as there is no tokens of value to learn from anyway
        if total_mask_sum == 0:
            return 0

        def checkpointed_step(idx, targets, mask, prev_loss, last_shift_states,
                              last_wkv_states, prev_steps):
            logits, new_shift_states, new_wkv_states = self(
                idx, last_shift_states, last_wkv_states)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.view(-1),
                                    reduction="none")

            submask = mask.view(-1)[:loss.shape[0]]
            submask_sum = torch.sum(submask)
            loss = torch.sum(loss * submask) / total_mask_sum

            loss = L2Wrap.apply(loss, logits, total_mask_sum, submask)
            new_steps = prev_steps + submask_sum
            new_loss = prev_loss + loss
            return new_loss, new_shift_states, new_wkv_states, new_steps

        total_loss = torch.tensor(
            0, dtype=self.emb.weight.dtype).requires_grad_()
        steps = 0
        states = BlockStateList.create(self.n_layer, B, C, seq.device,
                                       self.emb.weight.dtype)
        segment_count = math.ceil(T / self.ctx_len)

        #
        # BPTT learning, we split the sequence into segments
        # and perform a backward pass for each segment, on its own.
        #
        # Allowing us to perform backpropagation across context sizes much larger
        # then what is supported by the current GPU memory.
        #
        # This reduces the need for the checkpointing process, and mitigate
        # a known error where multiple backwards pass throws an exception.
        #
        # While not mathematically equivalent to full context size learning,
        # it makes "infctx" size training possible with deepspeed 2/3
        #
        # ---
        #
        # See the following, for more details on "Gradient computed twice" error:
        # https://github.com/microsoft/DeepSpeed/issues/988#issuecomment-1549417269
        #
        # Other possibly related issues on the topic:
        # https://github.com/microsoft/DeepSpeed/pull/677
        # https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-766366413
        #
        if do_bptt_learning:

            gradient_accumulation_steps = max(1, self.accumulate_grad_batches)
            optimizer = None
            cur_device = None

            # We use the average segment size, instead of ctx length size.
            # this helps ensure that the segment cutoffs do not make the last segment too small.
            # (eg, the last chunk having only 1 token)
            #
            # it also helps ensure the segment cutoff points are more varied, across mixed dataset sizes
            # and avoid potentially undesired training behaviour at fixed cutoff points
            # (this only applies for segmented learning)
            segment_size = min(math.ceil(T / segment_count), self.ctx_len)

            # Dummy 2D tenros of shape [1,1], are used to do "dummy checkpoint/forward/backprop" to keep everything in sync
            dummy_2d_zero = torch.tensor([[0]], dtype=torch.long, device=cur_device)

            # Get the max segment count across all GPUs, in the current batch, which is used to keep all devices are in sync
            # Once a thread has completed all its segments, it will do dummy checkpoint/forward/backprop with one token,
            # and stay in sync with the thread that are still working on their segments
            #
            # This is used to work around an undocumented behaviour for either lightning / deepspeed loss.backward multi-gpu handling
            # where the `self.manual_backward()` / `loss.backward()` call will block / freeze / hang when being too "out of sync"
            #
            # This can be viewed as a form of `fabric.barrier()` which is invoked implicitly by each `self.manual_backward()` call
            # except that it isn't exactly a `fabric.barrier()` - because it does not block immediately and instead blocks in
            # the next `self.manual_backward()` call if the previous ones are too far out of sync.
            # (its confusing, but makes sense for efficency)
            #
            # Additionally because the "code line position" and params actually matter for the 'barrier' code effect,
            # we cant work around this issue by doing dummy `self.manual_backward()` calls, in another if/else branch or loop
            #
            # Combined, this makes this issue very hard to trace and debug, as it will manifest itself as randomly "freezing"
            # when out of sync during `self.manual_backward()` calls. Without the current work around put in place
            #
            # We only do this, if we are doing bptt learning on all segments (-1), and gpu count > 1
            # otherwise we just use the segment count as it is
            if self.num_devices > 1:
                if self.bptt_learning_range <= 0:
                    # We perform forward/backward on the shared max segment count across all GPUs
                    # ---
                    # we map it to be a tensor, instead of the int directly, as this is more reliable across certain versions of torch/lightning
                    # https://discord.com/channels/992359628979568762/1148755392638234697/1148821863749931008
                    forward_segment_count  = self.trainer.strategy.reduce(torch.Tensor([segment_count]).to(torch.int), reduce_op="max")

                    # Convert to int, if its a torch tensor
                    if isinstance(forward_segment_count, torch.Tensor):
                        forward_segment_count = forward_segment_count.item()
                    # We perform as many backward pass as we need to be equal or more then bptt_learning_range
                    backward_segment_count = forward_segment_count
                else:
                    # We perform as many forward pass as we need to be equal or more then bptt_learning_range
                    # and perform an equal amount of backward pass
                    forward_segment_count  = max(segment_count, self.bptt_learning_range)
                    backward_segment_count = self.bptt_learning_range
            else:
                if self.bptt_learning_range <= 0:
                    # Since we do not need to sync GPUs here, we perform as much forward as we exactly need
                    forward_segment_count  = segment_count
                    backward_segment_count = forward_segment_count
                else:
                    # We clamp the backward segment count to the forward count, and bptt_learning_range
                    forward_segment_count  = segment_count
                    backward_segment_count = min(self.bptt_learning_range, segment_count)

            # We compute when we start the segmented learning process
            if forward_segment_count != backward_segment_count:
                start_learning_segment = max(segment_count - self.bptt_learning_range, 0);
            else:
                start_learning_segment = 0;

            # # Segment loss array to track (and reduce later)
            # # of size equal to forward_segment_count
            # segment_loss_arr = [0] * forward_segment_count

            # Lets go through and forward all the segments
            # (including dummy ones)
            for i in range(forward_segment_count):
                # Apply state truncation, if truncated learning is enabled
                # this limits the backprop process, reduces loss learning rate,
                # but save vram across extreamly large backpropagation steps
                if self.bptt_truncated_learning:
                    prv_shift_states = states.shift_states.clone().detach().requires_grad_(False)
                    prv_wkv_states = states.wkv_states.clone().detach().requires_grad_(False)
                else:
                    prv_shift_states = states.shift_states
                    prv_wkv_states = states.wkv_states

                # We use a dummy masked token 0, to do additional dummy checkpoint/forward/backprop when needed
                # for each additional call after the current "segment_count" max
                if i <= segment_count - 1:
                    cur_idx = idx[:, i * segment_size:(i + 1) * segment_size]
                    cur_tar = targets[:, i * segment_size:(i + 1) * segment_size]
                    cur_msk = seq_mask[:, i * segment_size:(i + 1) * segment_size]
                else:
                    cur_idx = dummy_2d_zero
                    cur_tar = dummy_2d_zero
                    cur_msk = dummy_2d_zero

                # Segmented learning, applies the forward/pass over each chunk seperately
                segment_loss, new_shift_states, new_wkv_states, steps = checkpointed_step(
                    cur_idx,
                    cur_tar,
                    cur_msk,
                    torch.tensor(0, dtype=self.emb.weight.dtype, device=cur_device).requires_grad_(True),
                    prv_shift_states,
                    prv_wkv_states,
                    steps,
                )
                states = BlockStateList(new_shift_states, new_wkv_states)

                # # Keep the segment loss (for backpassing in reverse)
                # segment_loss_arr[i] = segment_loss

                # Perform the backward pass accordingly, for valid segments (besides the last segment)
                # In this version, we do backward passes together the forward passes in the main segment loop
                # Instead of after all segment losses are computed
                if i >= start_learning_segment and i < start_learning_segment + backward_segment_count:
                    # The learning loss, should be normalized against the accumulation steps
                    # as we are bypassing the pytorch lightning normalization
                    # https://lightning.ai/docs/pytorch/2.0.4/common/lightning_module.html#backward
                    learning_loss = segment_loss / gradient_accumulation_steps

                    # Perform the backward pass accordingly, for valid segments (besides the last segment)
                    if i == start_learning_segment + backward_segment_count - 1:
                        # This is the last backward pass, we let the default pytorch lightning handle the backward pass
                        # and return the segment loss as part of the total loss
                        total_loss = total_loss + segment_loss
                    else:
                        # Undocumented multiple backward pass support
                        # https://github.com/Lightning-AI/lightning/blob/678f642808c54e4c490caee4df5d357301c976bb/tests/trainer/optimization/test_manual_optimization.py#L251
                        self.manual_backward(model_engine,learning_loss, optimizer)

                        # Accumulate without gradient, as we already did the backward pass
                        total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
                else:
                    # Even if its not the segments we use for backward pass, we still need to accumulate the loss
                    total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)

                # GC collect unused memory
                # gc.collect()
                # torch.cuda.empty_cache()

            # # Lets backpass the respective segments, in reverse
            # # (including dummy backpass)
            # for i in range(forward_segment_count-1, -1, -1):
            #     # Get the segment loss
            #     segment_loss = segment_loss_arr[i]
            #
            #     # Compute the backward pass for the segment
            #     if i >= start_learning_segment and i < start_learning_segment + backward_segment_count:
            #         # The learning loss, should be normalized against the accumulation steps
            #         # as we are bypassing the pytorch lightning normalization
            #         # https://lightning.ai/docs/pytorch/2.0.4/common/lightning_module.html#backward
            #         learning_loss = segment_loss / gradient_accumulation_steps
            #
            #         # Perform the backward pass accordingly, for valid segments (besides the start_learning_segment)
            #         if i > start_learning_segment:
            #             # Undocumented multiple backward pass support
            #             # https://github.com/Lightning-AI/lightning/blob/678f642808c54e4c490caee4df5d357301c976bb/tests/trainer/optimization/test_manual_optimization.py#L251
            #             self.manual_backward(learning_loss, optimizer, retain_graph=True)
            #
            #             # Accumulate without gradient, as we already did the backward pass
            #             total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
            #         else:
            #             # This is the last backward pass, we let the default pytorch lightning handle the backward pass
            #             # and return the segment loss as part of the total loss
            #             total_loss = total_loss + segment_loss
            #     else:
            #         # Even if its not the segments we use for backward pass, we still need to accumulate the loss
            #         total_loss = total_loss + segment_loss.clone().detach().requires_grad_(False)
            #
            #    # GC collect unused memory
            #    gc.collect()
            #    # torch.cuda.empty_cache()

        # Wandb logging only, if an active run exists
        # if wandb.run is not None:
        #     global_rank = self.global_rank
        #     global_device_count = self.num_devices * 1
        #     wandb.log({
        #         'substep': batch_idx * global_device_count + global_rank,
        #         'batchidx': batch_idx,
        #         'global_rank': global_rank,
        #         'real_ctx_len': T,
        #         'train/loss': total_loss,
        #         'trainer/global_step':self.global_step,
        #         'trainer/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr']
        #     })

        return total_loss

    @TCompileBaseline
    def training_step(self, batch,is_training_run=True,model_engine=None):
        total_loss = self.compute_loss(batch, is_training_run=is_training_run,model_engine=model_engine)

        #self.log('train/loss', total_loss, prog_bar=True)
        # If set - forces the above train/loss log line to always be on a new line
        if self.substep_logging:
            print(f"--->loss {total_loss}")

        if self.substep_cuda_cache_clear:
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss

    @TCompileBaseline
    def validation_step(self, batch, batch_idx):
        total_loss = self.compute_loss(batch, batch_idx, False)
        self.log('validation/loss', total_loss, prog_bar=True, sync_dist=True)
        return total_loss
