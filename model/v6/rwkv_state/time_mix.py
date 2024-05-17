import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from .lora import LoraLinear
from torch.utils.cpp_extension import load


HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

wkv6state_cuda_kernel = load(name="wkv6statefull",
                      sources=["cuda/cuda_state/wkv6state_op.cpp", f"cuda/cuda_state/wkv6state_cuda.cu"],
                      verbose=True,
                      extra_cuda_cflags=["-res-usage", "--use_fast_math",
                                         "-O3", "-Xptxas -O3",
                                         "--extra-device-vectorization",
                                         f"-D_N_={HEAD_SIZE}",
                                         f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])


class WKV_6STATE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u, s):
        with torch.no_grad():
            # Save the sizing & dtype
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            dtype = r.dtype
            ctx.dtype = dtype

            assert s.is_contiguous()
            assert w.is_contiguous()

            # Rest can be their respective types, but they are expected
            # to be consistent with each other
            assert dtype == k.dtype
            assert dtype == v.dtype
            assert dtype == u.dtype
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert u.is_contiguous()

            # Lets pre-compute the exp(-w)
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u, s.clone())

            s = s.to(dtype)

            # Output logits
            y = torch.empty((B, T, C), device=r.device, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)

            # Call the cuda kernel
            if dtype == torch.bfloat16:
                wkv6state_cuda_kernel.forward(B, T, C, H, r, k, v, ew, u, s, y)
            elif dtype == torch.float16:
                wkv6state_cuda_kernel.forward_fp16(B, T, C, H, r, k, v, ew, u, s, y)
            elif dtype == torch.float32:
                wkv6state_cuda_kernel.forward_fp32(B, T, C, H, r, k, v, ew, u, s, y)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for WKV6_CUDA")
            
            # Logits (without state)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            # Get the sizing & dtype
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            dtype = ctx.dtype

            # GY dtype
            assert gy.dtype == dtype
            assert gy.is_contiguous()
            r, k, v, ew, u, s = ctx.saved_tensors

            # Initialize all the backward pass vars required
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)

            # Perform the backward pass
            if dtype == torch.bfloat16:
                wkv6state_cuda_kernel.backward(B, T, C, H, r, k, v, ew, u, s, gy, gr, gk, gv, gw, gu, gs)
            elif dtype == torch.float16:
                wkv6state_cuda_kernel.backward_fp16(B, T, C, H, r, k, v, ew, u, s, gy, gr, gk, gv, gw, gu, gs)
            elif dtype == torch.float32:
                wkv6state_cuda_kernel.backward_fp32(B, T, C, H, r, k, v, ew, u, s, gy, gr, gk, gv, gw, gu, gs)
            else:
                raise ValueError(f"Unsupported dtype {dtype} for WKV5_CUDA")

            #gw = torch.sum(gw, 0).view(H, C//H) # FIXME - not needed, because w is a different shape now in v6?
            gu = torch.sum(gu, 0).view(H, C//H)
            return (
                # B, T, C, H,
                None, None, None, None, 
                gr, gk, gv, gw, gu, gs)


def RUN_CUDA(B, T, C, H, r, k, v, w, u, s):
    return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)


class TimeMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size
        self.n_head = args.n_head  # args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd,
                                                          D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA,
                                                          args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head,
                                                       self.head_size))
            # self.time_state = nn.Parameter(torch.zeros(self.n_head,
            #                                            self.head_size,
            #                                            self.head_size))


        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # init attention
        if args.lora.lora_on and args.lora.train_state:
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        elif "att" in args.lora.parts and args.lora.r > 0:
            self.key = LoraLinear(args, args.n_embd, args.dim_att, bias=False)
            self.value = LoraLinear(args, args.n_embd, args.dim_att, bias=False)
            self.receptance = LoraLinear(args, args.n_embd, args.dim_att, bias=False)
        else:
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)

        # init output
        self.output = nn.Linear(args.dim_att,
                                args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd,
                              args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head,
                                 args.dim_att,
                                 eps=(1e-5)*(args.head_size_divisor**2))


    def forward(self, x, last_state):
        shift_state_out = x[:, -1]

        B, T, C = x.size()
        H = self.n_head

        dxprev = torch.concat((last_state[0].unsqueeze(1), x[:, :-1]), dim=1) - x
        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww
        u = self.time_faaaa

        # logits and state
        wkv_state = last_state[1].to(r.dtype)

        x = RUN_CUDA(B, T, C, H,
                    r, k, v, w,
                    u=self.time_faaaa,
                    s=wkv_state)

        B, T, C = x.size()

        x = x.view(-1, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)

        return (x, (shift_state_out, wkv_state))
