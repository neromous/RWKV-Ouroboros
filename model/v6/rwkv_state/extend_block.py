import torch
import torch.nn as nn
from .time_mix import TimeMix
from .channel_mix import ChannelMix

class BlockState:
    def __init__(self, time_mix_state: tuple[torch.Tensor,torch.Tensor],
                 channel_mix_state: torch.Tensor):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state

class BlockStateList:
    def __init__(self, shift_states, wkv_states, encoder, decoder):
        self.wkv_states = wkv_states
        self.shift_states = shift_states
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        wkv_states = torch.empty((N, B, n_head, head_size, head_size),
                                 device=device,
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        encoder = Encoder(wkv_states.shape, shift_states.shape).to(device) # 初始化编码器
        decoder = Decoder(wkv_states.shape, shift_states.shape).to(device) # 初始化解码器
        return BlockStateList(shift_states, wkv_states, encoder, decoder)

    @staticmethod
    def merge(s1, s2, s1_part: float=0.5, s2_part: float=0.5):
        wkv_states = s1.wkv_states * s1_part + s2.wkv_states * s2_part
        shift_states = s1.shift_states * s1.s1_part + s2.shift_states * s2_part 
        return BlockStateList(shift_states, wkv_states, s1.encoder, s1.decoder) # 使用s1的编码器和解码器

    def decay(self, ratio: float=0.95):
        if ratio == 0:
            return self
        self.wkv_states = self.wkv_states * ratio
        self.shift_states = self.shift_states * ratio
        return self
    
    def __getitem__(self, layer: int):
        return BlockState(
            (self.shift_states[layer, 0], self.wkv_states[layer]),
            (self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state
    def cpu(self):
        wkv_states =  self.wkv_states.detach().cpu()
        shift_states = self.shift_states.detach().cpu()
        return BlockStateList(shift_states, wkv_states, self.encoder.cpu(), self.decoder.cpu())

    def cuda(self):
        wkv_states =  self.wkv_states.to("cuda")
        shift_states = self.shift_states.to("cuda")
        return BlockStateList(shift_states, wkv_states, self.encoder.cuda(), self.decoder.cuda())

    @classmethod
    def save(cls, item, path):
        item = item.cpu()
        data = {
            "wkv_states": item.wkv_states,
            "shift_states": item.shift_states,
            "encoder_state_dict": item.encoder.state_dict(), # 保存编码器状态字典
            "decoder_state_dict": item.decoder.state_dict() # 保存解码器状态字典
        }
        torch.save(data, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu')
        wkv_states = data['wkv_states']
        shift_states = data['shift_states']
        encoder = Encoder(wkv_states.shape, shift_states.shape) # 初始化编码器
        decoder = Decoder(wkv_states.shape, shift_states.shape) # 初始化解码器
        encoder.load_state_dict(data["encoder_state_dict"]) # 加载编码器状态字典
        decoder.load_state_dict(data["decoder_state_dict"]) # 加载解码器状态字典
        item = cls(shift_states, wkv_states, encoder, decoder)
        return item

# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, wkv_shape, shift_shape):
        super().__init__()
        self.wkv_encoder = nn.Linear(wkv_shape[-1], wkv_shape[-1]) # 可以根据需求修改网络结构
        self.shift_encoder = nn.Linear(shift_shape[-1], shift_shape[-1]) # 可以根据需求修改网络结构

    def forward(self, wkv_states, shift_states):
        wkv_states = self.wkv_encoder(wkv_states)
        shift_states = self.shift_encoder(shift_states)
        return wkv_states, shift_states

# 定义解码器网络
class Decoder(nn.Module):
    def __init__(self, wkv_shape, shift_shape):
        super().__init__()
        self.wkv_decoder = nn.Linear(wkv_shape[-1], wkv_shape[-1]) # 可以根据需求修改网络结构
        self.shift_decoder = nn.Linear(shift_shape[-1], shift_shape[-1]) # 可以根据需求修改网络结构

    def forward(self, wkv_states, shift_states):
        wkv_states = self.wkv_decoder(wkv_states)
        shift_states = self.shift_decoder(shift_states)
        return wkv_states, shift_states

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = TimeMix(args, layer_id)

        self.ffn = ChannelMix(args, layer_id)
        
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p=args.dropout)
            self.drop1 = nn.Dropout(p=args.dropout)

    def forward(self, x, last_state: BlockState):
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )

        if self.args.dropout > 0.0:
            # Handle with dropout
            x = self.drop0(x + att_out)
            ffn_out, ffn_state = self.ffn(
                self.ln2(x),
                last_state.channel_mix_state,
            )
            x = self.drop1(x + ffn_out)
        else:
            # Handle without dropout
            x = x + att_out
            ffn_out, ffn_state = self.ffn(
                self.ln2(x),
                last_state.channel_mix_state,
            )
            x = x + ffn_out

        return x, BlockState(att_state, ffn_state)


# import torch
# import torch.nn as nn
# from .time_mix import TimeMix
# from .channel_mix import ChannelMix


# class BlockState:

#     def __init__(self, time_mix_state: tuple[torch.Tensor,torch.Tensor],
#                  channel_mix_state: torch.Tensor):
#         self.time_mix_state = time_mix_state
#         self.channel_mix_state = channel_mix_state


# class BlockStateList:

#     def __init__(self, shift_states, wkv_states):
#         self.wkv_states = wkv_states
#         self.shift_states = shift_states

#     @staticmethod
#     def create(N, B, C, n_head, head_size, device, dtype):
#         result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
#         result.wkv_states[:] = 0
#         # result.wkv_states[:, :, :, -1] = -1e38
#         result.shift_states[:] = 0
#         return result

#     @staticmethod
#     def empty(N, B, C, n_head, head_size, device, dtype):
#         wkv_states = torch.empty((N, B, n_head, head_size, head_size),
#                                  device=device,
#                                  dtype=torch.float)
#         shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
#         return BlockStateList(shift_states, wkv_states)

#     @staticmethod
#     def merge(s1, s2, s1_part: float=0.5, s2_part: float=0.5):
#         wkv_states = s1.wkv_states * s1_part + s2.wkv_states * s2_part
#         shift_states = s1.shift_states * s1.s1_part + s2.shift_states * s2_part 
#         return BlockStateList(shift_states, wkv_states)

#     def decay(self, ratio: float=0.95):
#         if ratio == 0:
#             return self
#         self.wkv_states = self.wkv_states * ratio
#         self.shift_states = self.shift_states * ratio
#         return self
    
#     def __getitem__(self, layer: int):
#         return BlockState(
#             (self.shift_states[layer, 0], self.wkv_states[layer]),
#             (self.shift_states[layer, 1]))

#     def __setitem__(self, layer: int, state: BlockState):
#         self.shift_states[layer, 0] = state.time_mix_state[0]
#         self.wkv_states[layer] = state.time_mix_state[1]
#         self.shift_states[layer, 1] = state.channel_mix_state
#     def cpu(self):
#         wkv_states =  self.wkv_states.detach().cpu()
#         shift_states = self.shift_states.detach().cpu()
#         return BlockStateList(shift_states, wkv_states)

#     def cuda(self):
#         wkv_states =  self.wkv_states.to("cuda")
#         shift_states = self.shift_states.to("cuda")
#         return BlockStateList(shift_states, wkv_states)

#     @classmethod
#     def save(cls, item, path):
#         item = item.cpu()
#         data = {
#             "wkv_states": item.wkv_states,
#             "shift_states": item.shift_states
#         }
#         torch.save(data, path)

#     @classmethod
#     def load(cls, path):
#         data = torch.load(path, map_location='cpu')
#         wkv_states = data['wkv_states']
#         shift_states = data['shift_states']
#         item = cls(shift_states, wkv_states)
#         return item

# class Block(nn.Module):
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#         self.layer_id = layer_id

#         self.ln1 = nn.LayerNorm(args.n_embd)
#         self.ln2 = nn.LayerNorm(args.n_embd)

#         if self.layer_id == 0:
#             self.ln0 = nn.LayerNorm(args.n_embd)

#         self.att = TimeMix(args, layer_id)

#         self.ffn = ChannelMix(args, layer_id)
        
#         if args.dropout > 0:
#             self.drop0 = nn.Dropout(p=args.dropout)
#             self.drop1 = nn.Dropout(p=args.dropout)

#     def forward(self, x, last_state: BlockState):
#         if self.layer_id == 0:
#             x = self.ln0(x)

#         att_out, att_state = self.att(
#             self.ln1(x),
#             last_state.time_mix_state,
#         )

#         if self.args.dropout > 0.0:
#             # Handle with dropout
#             x = self.drop0(x + att_out)
#             ffn_out, ffn_state = self.ffn(
#                 self.ln2(x),
#                 last_state.channel_mix_state,
#             )
#             x = self.drop1(x + ffn_out)
#         else:
#             # Handle without dropout
#             x = x + att_out
#             ffn_out, ffn_state = self.ffn(
#                 self.ln2(x),
#                 last_state.channel_mix_state,
#             )
#             x = x + ffn_out

#         return x, BlockState(att_state, ffn_state)
