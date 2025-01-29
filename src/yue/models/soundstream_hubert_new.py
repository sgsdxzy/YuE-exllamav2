 

from typing import Sequence, Optional, Union
import sys
# sys.path.append('/aifs4su/data/zheny/fairseq/vae_v2/codec_final')
import math
import random

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.seanet import SEANetEncoder, SEANetDecoder
from quantization  import ResidualVectorQuantizer#,VectorQuantize
from transformers import  AutoModel
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoFeatureExtractor, WhisperModel
# sys.path.append('/scratch/buildlam/codeclm/jiahaopan/codec_final/RepCodec')
from RepCodec.repcodec.modules.encoder import Encoder
from RepCodec.repcodec.modules.decoder import Decoder
# sys.path.append('/data/zheny/UniAudio/codec/descriptaudiocodecs')
#descript-audio-codec/dac/model
import descriptaudiocodec.dac.model.dac  as dac2
# sys.path.append('/aifs4su/data/zheny/peiwensun/project/s3prl/s3prl/upstream/hubert/')
# from simple_expert import UpstreamExpert

def get_model_size(model):
    # 计算总参数数
    total_params = sum(p.numel() for p in model.parameters())
    
    # 假设每个参数都是32位浮点数，计算模型大小（以字节为单位）
    model_size_bytes = total_params    # 每个参数4字节
    
    # 转换为更易读的单位（例如，MB）
    model_size_mb = model_size_bytes / (1024 ** 2)
    
    return total_params, model_size_mb


class SoundStream(nn.Module):
    """ SoundStream model or EnCodec model.
    
    Args:
        n_filters (int): n_filters (int): Base width for the model.
        D (int): Intermediate representation dimension.
        target_bandwidths (Sequence[int]): Target bandwidths in K-bits/second.
        ratios (Sequence[int]): downsampling factors, whose multiplication is the hop size.
        sample_rate (int): wave sampling rate.
        bins (int): number of code words in a codebook.
        normalize (bool): audio normalization.

    """
    def __init__(
        self,
        n_filters: int = 32,
        D: int = 128,
        # target_bandwidths: Sequence[Union[int, float]] = [0.5, 1, 1.5, 2, 4, 6],
        target_bandwidths: Sequence[Union[int, float]] = [1, 1.5, 2, 4, 6],
        ratios: Sequence[int] = [8, 5, 4, 2], #  downsampling by 320
        sample_rate: int = 16000,
        bins: int = 1024,
        normalize: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.hop_length = np.prod(ratios)
        # total nb of codebooks, e.g., 6Kb/s, sr=16000 and hop_length=320 => nq = 12
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios)) # 50 Hz
        self.bits_per_codebook = int(math.log2(bins)) # 1024 => 10
        self.target_bandwidths = target_bandwidths
        self.n_q = n_q
        self.sample_rate = sample_rate

        # Encoder model
        # self.encoder = SEANetEncoder(n_filters=n_filters, dimension=D, ratios=ratios, causal=causal)
        self.encoder = dac2.Encoder(            64,ratios,D)
        # RVQ model
        self.encoder_semantic = Encoder(input_channels=768,encode_channels=768)
        self.decoder_semantic = Decoder(code_dim=768,output_channels=768,decode_channels=768)
        # out_D=D+768
        self.quantizer = ResidualVectorQuantizer(dimension=D+768, n_q=n_q, bins=bins)
        # Decoder model
        
        # self.decoder = SEANetDecoder(n_filters= n_filters, dimension=D, ratios=ratios, causal=causal)
        self.decoder_2 = dac2.Decoder(            D,1024,ratios,)

        # )
        # self.upstream = UpstreamExpert(
        #     ckpt = '/aifs4su/data/zheny/fairseq/outputs/2024-05-08/12-50-35/checkpoints2/checkpoint_8_225000_converted.pt',
        # )#.to(self.args.device)
        # self.upstream.model = self.upstream.model.to(self.device)
        c=1
        # self.upstream(wavs) 
        # self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")

        self.is_semantic= True 
        if self.is_semantic:
            # self.semantic_model = AutoModel.from_pretrained("/aifs4su/data/zheny/DiT_TTS/ckpts/yz_2")   
            # self.semantic_model = AutoModel.from_pretrained("/aifs4su/data/zheny/fairseq/outputs/2024-05-11/13-27-56/hf15")
            self.semantic_model = AutoModel.from_pretrained("./xcodec_mini_infer/semantic_ckpts/hf_1_325000")
            self.semantic_model.eval()
            # self.transform_linear = nn.Linear(1024, 768)


 
        # processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        # self.semantic_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.fc_prior = nn.Linear(D+768, D+768 )
        # self.fc_prior= nn.Linear( D, D )
        self.fc_post1= nn.Linear( D+768, 768 )
        self.fc_post2= nn.Linear( D+768,  D)

    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
    def calculate_rec_loss(self, rec, target):  
 
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()
        # rec_loss = F.mse_loss(target, rec)


        return rec_loss

    @torch.no_grad()
    def get_regress_target(self, x ):
        x= x[:,0,:]
        x = F.pad(x, (160, 160))
        target = self.semantic_model(x, output_hidden_states=True) .hidden_states
        target = torch.stack(target, dim=1)#.transpose(-1, -2)#.flatten(start_dim=1, end_dim=2)
        
        target = target.mean(1)   
        # target = target[9]
        return target

 
    def forward(self, x: torch.Tensor, bw: int):

        e_semantic_input = self.get_regress_target_whisper(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)
 
 
        e= torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)

 
        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)

        quantized_semantic = self.fc_post1(quantized.transpose(1, 2)).transpose(1, 2)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
 
        o_semantic = self.decoder_semantic(quantized_semantic )
        semantic_recon_loss = F.mse_loss(e_semantic_input.transpose(1, 2).detach(),o_semantic)

        return o, commit_loss, semantic_recon_loss,None
        # return o, commit_loss, distill_loss.mean(),None

    def encode(self, x: torch.Tensor, target_bw: Optional[int] = None) -> torch.Tensor:
        # e = self.encoder(x)
        # if target_bw is None:
        #     bw = self.target_bandwidths[-1]
        # else:
        bw = target_bw
        # codes = self.quantizer.encode(e, self.frame_rate, bw)

        
        # if e_acoustic.shape[2] != e_semantic.shape[2]:
        #     print(f"e_acoustic {e_acoustic.shape} e_semantic{e_semantic.shape}")

        e_semantic_input = self.get_regress_target(x).detach()

        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)


        if e_acoustic.shape[2] != e_semantic.shape[2]:
            # e_acoustic = self.encoder(F.pad(x[:,0,:], (160, 160)).unsqueeze(0)) 
            e_acoustic = self.encoder(torch.transpose(F.pad(x[:,0,:], (160, 160)).unsqueeze(0), 0, 1))
 
        e= torch.cat([e_acoustic, e_semantic], dim=1)

        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)

        quantized, codes, bandwidth, commit_loss  = self.quantizer(e, self.frame_rate, bw)
        return codes

    def get_embed(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.decode(codes)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.decode(codes)
        quantized_acoustic = self.fc_post2(quantized.transpose(1, 2)).transpose(1, 2)

        o = self.decoder_2(quantized_acoustic)
        return o

# test
if __name__ == '__main__':
    soundstream = SoundStream(n_filters=32, D=256)#.cuda(0)
    # get_model_size(soundstream)
    for i in range(10):
        print(f"Iter {i}: ")
        x = torch.rand(1, 1, 16000)#.cuda(0)
        o, commit_loss, distill_loss,_= soundstream(x,soundstream.target_bandwidths[-1])
        print('output', o.shape)
