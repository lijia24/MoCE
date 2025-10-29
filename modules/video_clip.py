import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.checkpoint import checkpoint
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
import pdb
import math
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CausalTimeAttention2(nn.Module):
    """
    StreamFormer-style causal temporal attention.
    Input : query (T, B, dim), key (T, B, dim), value (T, B, dim)
    Output: (T, B, dim)   # 形状不变
    """
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5

        # 独立的q, k, v线性层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, query, key, value):
        # 输入格式为 (T, B, C)
        T, B, C = query.shape
        
        # 分别计算 q, k, v
        q = self.q_proj(query)                  # (T, B, C)
        k = self.k_proj(key)                    # (T, B, C)
        v = self.v_proj(value)                  # (T, B, C)
        
        # 重塑为多头格式
        q = q.reshape(T, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)  # (B, n_heads, T, head_dim)
        k = k.reshape(T, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)  # (B, n_heads, T, head_dim)
        v = v.reshape(T, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)  # (B, n_heads, T, head_dim)

        # 标准缩放点积
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, n_heads, T, T)

        # ---- causal mask: 下三角为 0，上三角为 -inf ----
        causal_mask = torch.triu(torch.ones(T, T, device=query.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                          # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        out = out.transpose(0, 1)                   # (T, B, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # 残差连接

        out = query + out
        
        return out, None  # 返回元组格式以匹配MultiheadAttention的接口


class MixtureOfRoutingAttention(nn.Module):
    def __init__(self, dim, heads=8, k_expert=1, dropout=0.1):
        super().__init__()
        self.k = k_expert
        self.dim, self.heads = dim, heads

        # 专家：空域
        self.spatial_experts = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            for _ in range(k_expert)
        ])
        # 专家：时域
        self.temporal_experts = nn.ModuleList([
            #nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
            CausalTimeAttention2(dim)
            for _ in range(k_expert)
        ])

        # 路由网络：轻量级
        self.router = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * k_expert)   # 空、时各 k 个路由 logits
        )

        # cross-direction fusion
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm_s = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim)
        )
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, x,spatial_cls_feature=None,template_cls_feature=None):
        B, T, D = x.shape
        # 1) 路由权重
        route_logits = self.router(x.mean(dim=1))          # (B, 2*k)
        route = route_logits.softmax(-1)                   # (B, 2*k)
        w_s, w_t = route.chunk(2, dim=-1)                  # (B, k) each
        
        # 2) 空域专家加权
        if spatial_cls_feature is not None:
            spatial_cls_feature_norm=self.norm_s(spatial_cls_feature)
            x_norm = self.norm_s(x)
            #print(w_s)
            # 只选择Top-K专家
            _, top_indices = torch.topk(w_s, k=1, dim=1)  # 只选择权重最高的专家
            spatial_out = 0
            for b in range(B):
                expert_idx = top_indices[b, 0].item()
                expert = self.spatial_experts[expert_idx]
                attn_out = expert(x_norm[b:b+1], x_norm[b:b+1], spatial_cls_feature_norm[b:b+1], need_weights=False)[0]
                spatial_out = torch.cat([spatial_out, attn_out], dim=0) if b > 0 else attn_out
        else:
            x_norm = self.norm_s(x)
            # 只选择Top-K专家
            
            _, top_indices = torch.topk(w_s, k=1, dim=1)  # 只选择权重最高的专家
            spatial_out = 0
            for b in range(B):
                expert_idx = top_indices[b, 0].item()
                expert = self.spatial_experts[expert_idx]
                attn_out = expert(x_norm[b:b+1], x_norm[b:b+1], x_norm[b:b+1], need_weights=False)[0]
                spatial_out = torch.cat([spatial_out, attn_out], dim=0) if b > 0 else attn_out

        # 3) 时域专家加权
        x_norm_t = self.norm_t(x).transpose(0, 1)          # (T, B, D)
        if template_cls_feature is not None:

            template_cls_feature_norm_t=self.norm_t(template_cls_feature).transpose(0, 1)  
            # 只选择Top-K专家
            _, top_indices = torch.topk(w_t, k=1, dim=1)  # 只选择权重最高的专家
            temporal_out = 0
            for b in range(B):
                expert_idx = top_indices[b, 0].item()
                expert = self.temporal_experts[expert_idx]
                attn_out = expert(template_cls_feature_norm_t[:, b:b+1], x_norm_t[:, b:b+1], x_norm_t[:, b:b+1])[0]
                temporal_out_b = attn_out.transpose(0, 1)
                temporal_out = torch.cat([temporal_out, temporal_out_b], dim=0) if b > 0 else temporal_out_b
        else:
            # 只选择Top-K专家
            _, top_indices = torch.topk(w_t, k=1, dim=1)  # 只选择权重最高的专家
            temporal_out = 0
            for b in range(B):
                expert_idx = top_indices[b, 0].item()
                expert = self.temporal_experts[expert_idx]
                attn_out = expert(x_norm_t[:, b:b+1], x_norm_t[:, b:b+1], x_norm_t[:, b:b+1], need_weights=False)[0]
                temporal_out_b = attn_out.transpose(0, 1)
                temporal_out = torch.cat([temporal_out, temporal_out_b], dim=0) if b > 0 else temporal_out_b
        
        # 4) cross-direction fusion
        fused = self.cross(
            spatial_out, temporal_out, temporal_out, need_weights=False
        )[0]

        x = x + fused
        x = x + self.mlp(self.norm_mlp(x))
        return x



class Video_MLP_Adapter(nn.Module):
    """
    多层 MLP-Adapter  输入: (B, L, D)  输出: (B, L, D)
    hidden_factor: 降维倍数  (D // hidden_factor)
    mid_layers : 瓶颈层数（≥1）
    """
    def __init__(self, dim, hidden_factor=4, mid_layers=2, act=nn.GELU, drop=0.):
        super().__init__()
        mid_dim = dim // hidden_factor

        # 降维
        self.down = nn.Sequential(
            nn.Linear(dim, mid_dim),
            act()
        )
        # 瓶颈多层
        self.mid = nn.Sequential(
            *[nn.Sequential(nn.Linear(mid_dim, mid_dim), act(), nn.Dropout(drop))
              for _ in range(mid_layers)]
        )
        # 升维
        self.up = nn.Sequential(
            nn.Linear(mid_dim, dim),
            nn.Dropout(drop)
        )

        # 残差 & 缩放
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.zeros(1))   # 初始恒等映射

    def forward(self, x):
        """
        x: (B, L, D)
        return: (B, L, D)
        """
        res = x
        x = self.norm(x)
        x = self.down(x)        # (B, L, mid_dim)
        x = self.mid(x)         # 多层非线性
        x = self.up(x)          # (B, L, D)
        return res + self.scale * x

class VideoTransformer(nn.Module):
    
    def __init__(self, dim=512, depth=8, heads=8, model_type=['factorised']):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        
        layers = []
        moe_layers=[]
        self.video_adapter=Video_MLP_Adapter(dim)

        for _ in range(depth):
            moe_layers.append(MixtureOfRoutingAttention(dim, heads))        
            #moe_layers.append(STDecoupleBlock(dim))


        self.moe_layers=nn.ModuleList(moe_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, regularization, cls_emb=None, labels=None,spatial_cls_feature=None,template_cls_feature=None,desc_embeds=None,frame_position_embeddings=None):
        
        B, T, D = x.shape
        x_ori=x.clone()


        if desc_embeds is not None and cls_emb is not None: #Training
            selected_cls_emb = cls_emb[labels]
            selected_cls_emb = selected_cls_emb.unsqueeze(1).expand(-1, T, -1) 
            x=self.video_adapter(x)
            desc_embeds=desc_embeds.unsqueeze(1).expand(-1, T, -1)
            
            delta = x-desc_embeds
            gate = (delta * selected_cls_emb)
            gate = (gate > 0).float()
            alpha=0.1
            x=x + alpha * gate * delta
            x_pooled = x.mean(dim=1)  # (B, D)
            cls_emb_pool=cls_emb[labels]           
            

            x_ori_pool=x_ori.mean(dim=1)
            clip_score = F.cosine_similarity(F.normalize(x_pooled, p=2, dim=-1), cls_emb_pool, dim=1)

            clip_score_ori = F.cosine_similarity(F.normalize(x_ori_pool, p=2, dim=-1), cls_emb_pool, dim=1)
            # print(f"减去背景的CLIP Score: {clip_score.mean().item():.4f}") 
            # print(f"原始的的CLIP Score: {clip_score_ori.mean().item():.4f}") 
        if frame_position_embeddings is not None:
            x=x+frame_position_embeddings
        out_s,out_t=None,None
        for blk in self.moe_layers:

            if isinstance(blk, MixtureOfRoutingAttention):
                # VideoTextCrossAttention需要额外的参数
                if spatial_cls_feature is not None and template_cls_feature is not None :#训练时
                    selected_spatial_cls_feature = spatial_cls_feature[labels]
                    selected_spatial_cls_feature = selected_spatial_cls_feature.unsqueeze(1).expand(-1, T, -1) 
                    selected_template_cls_feature = template_cls_feature[labels]
                    selected_template_cls_feature = selected_template_cls_feature.unsqueeze(1).expand(-1, T, -1)
                    x = blk(x, selected_spatial_cls_feature,selected_template_cls_feature)
                else:
                    x = blk(x)
            else:
                # 其他层正常处理
                x = blk(x)
        if out_s is not None and out_t is not None:
            return self.norm(x),self.norm(out_s),self.norm(out_t)
        return self.norm(x)     # (B, 8, 512)



class video_header(nn.Module):
    def __init__(self, vid_head, interaction, clip_state_dict, num_experts=4, spe_cls_feature=None):
        super().__init__()
        self.vid_header = vid_head
        self.interaction = interaction     
        self.mse_criterion = nn.MSELoss()
        #pdb.set_trace()
        if spe_cls_feature is None:
            # finetune on k400
            self.spe_cls_feature = nn.Parameter(torch.zeros(400,clip_state_dict["text_projection"].shape[1]), requires_grad=False)
        else:
            self.spe_cls_feature = nn.Parameter(spe_cls_feature, requires_grad=False)   
            #[num_class, dim]
        self.spe_cls_feature = nn.Parameter(torch.zeros(400,clip_state_dict["text_projection"].shape[1]), requires_grad=False)
        #pdb.set_trace()
        assert vid_head in ["None", "Transf"]
        
        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

            self.transformer=VideoTransformer(dim=512, depth=4, heads=1)
           

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):

            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def agg_video_feat(self, x, gen_cls_feat = None, regularization: bool = False, labels=None,spatial_cls_feature=None,template_cls_feature=None,desc_embeds=None):
        b, t, c = x.size()
        x = x.contiguous()

        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            
            # Temporal Feature Modulation
            if gen_cls_feat is not None: #False when train
                num_esti = 5
                scale_param = 0.05

                # spatial features from clip part
                x_clip = x.mean(dim=1,keepdim=False) # [bs, dim]
                x_clip = x_clip / x_clip.norm(dim=-1,keepdim=True)
                bs,c_ = x_clip.size() 

                # estimate gen_feat and spe_feat
                gen_logits = x_clip @ gen_cls_feat.t() # [bs, gen_num_class]
                spe_logits = x_clip @ self.spe_cls_feature.t() # [bs, spe_num_class]
                
                gen_values, gen_indices = gen_logits.topk(num_esti,-1,True,True) # values, indices: [bs, num_esti]
                gen_feat_esti = torch.gather(input=gen_cls_feat.unsqueeze(0).expand(bs,-1,-1), dim=1, \
                                        index = gen_indices.unsqueeze(-1).expand(-1,-1,c_)) # [bs, num_esti, dim]
                spe_values, spe_indices = spe_logits.topk(num_esti,-1,True,True) # values, indices: [bs, num_esti]
                spe_feat_esti = torch.gather(input=self.spe_cls_feature.unsqueeze(0).expand(bs,-1,-1), dim=1, \
                                        index = spe_indices.unsqueeze(-1).expand(-1,-1,c_)) # [bs, num_esti, dim]

                gen_spe_sim = torch.bmm(gen_feat_esti,spe_feat_esti.permute(0,2,1)) # [bs,num_esti, num_esti]
                gen_spe_sim, indi = torch.max(gen_spe_sim,dim=-1)
                gen_spe_sim=1/torch.exp((1-gen_spe_sim.mean(dim=-1))/scale_param)
                gen_spe_sim=gen_spe_sim.unsqueeze(-1).unsqueeze(-1)
                

            else: 
                gen_spe_sim=torch.ones(1,device=x.device)
            
            x_original = x #torch.Size([64, 8, 512])
            seq_length = t #8
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1) #[[0, 1, 2, 3, 4, 5, 6, 7],[0, 1, 2, 3, 4, 5, 6, 7],x64
            
            frame_position_embeddings = self.frame_position_embeddings(position_ids) #torch.Size([64, 8, 512])
            #x = x + frame_position_embeddings
            
            #x = x.permute(1, 0, 2)  # NLD -> LND #torch.Size([8, 64, 512])
            x = self.transformer(x, regularization, gen_cls_feat, labels,spatial_cls_feature,template_cls_feature,desc_embeds,frame_position_embeddings) #创新点 #torch.Size([8, 64, 512])
            #x = x.permute(1, 0, 2)  # LND -> NLD torch.Size([64, 8, 512])
            x = gen_spe_sim * x.type(x_original.dtype) + x_original
        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        return x


    def get_logits(self, vid_emb, cls_emb):
        #vid_emb=torch.Size([batch, T, dim])，cls_emb=torch.Size([batch, num_cla, dim])
        
        if self.interaction == 'DP':
            vid_emb = vid_emb.mean(dim=1, keepdim=False)
            vid_emb = vid_emb / vid_emb.norm(dim=-1, keepdim=True)
            cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
            logit = vid_emb @ cls_emb.t()  
        else:
            raise NotImplementedError
        return logit
    def compute_CLIP_score(self,vid_emb, cls_emb,labels):
        # 计算x和cls_emb的CLIP score
        if cls_emb is not None:
            # 对x进行时间维度平均池化，得到(B, D)
            x_pooled = vid_emb.mean(dim=1)  # (B, D)
            selected_cls_emb = cls_emb[labels]
            
            clip_score = F.cosine_similarity(x_pooled, selected_cls_emb, dim=1)
            return clip_score
    def forward(self, vid_emb, cls_emb,desc_embeds=None, labels=None,spatial_cls_feature=None,template_cls_feature=None):
        
        if self.training:
            
            vid_emb_expert = self.agg_video_feat(vid_emb, cls_emb, regularization=True, labels=labels,spatial_cls_feature=spatial_cls_feature,template_cls_feature=template_cls_feature,desc_embeds=desc_embeds) #torch.Size([64, 8, 512])
            
            logits = self.get_logits(vid_emb_expert, cls_emb) #torch.Size([64, num_class])
            
            mse_loss = self.mse_criterion(vid_emb_expert, vid_emb)
            logits_s=0 #self.get_logits(out_s, spatial_cls_feature)
            logits_t=0# self.get_logits(out_t, template_cls_feature)
           
            sim_loss=0 #F.cosine_similarity(vid_emb_expert.mean(dim=1),desc_embeds,dim=-1).mean()
           
            return logits, mse_loss, logits_t,vid_emb_expert,cls_emb
        else:

            # Test - 尝试所有标签，选择CLIP分数最高的结果
            B = vid_emb.size(0)
            num_classes = cls_emb.size(0)
            best_logits = None
            best_clip_scores = torch.full((B,), float('-inf'), device=vid_emb.device)
            best_vid_emb = None
            
            # 为每个类别计算CLIP分数
            for class_idx in range(num_classes):
                # 为当前批次的所有样本使用相同的类别标签
                current_labels = torch.full((B,), class_idx, device=vid_emb.device, dtype=torch.long)
                
                # 使用当前标签处理视频特征
                current_vid_emb = self.agg_video_feat(vid_emb, gen_cls_feat=cls_emb, regularization=False, 
                                                    labels=current_labels, spatial_cls_feature=spatial_cls_feature,
                                                    template_cls_feature=template_cls_feature, desc_embeds=desc_embeds)
                
                # 计算当前标签的CLIP分数
                current_clip_scores = self.compute_CLIP_score(current_vid_emb, cls_emb, current_labels)
                
                # 更新每个样本的最佳结果
                mask = current_clip_scores > best_clip_scores
                best_clip_scores[mask] = current_clip_scores[mask]
                
                if best_vid_emb is None:
                    best_vid_emb = current_vid_emb.clone()
                else:
                    best_vid_emb[mask] = current_vid_emb[mask]
            
            # 使用最佳的视频特征计算最终logits
            logits = self.get_logits(best_vid_emb, cls_emb)
            
            #print(f"测试阶段最佳CLIP Score: {best_clip_scores.mean().item():.4f}")
            
            return logits

class VideoCLIP(nn.Module):
    def __init__(self, clip_model, n_seg) :
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
        self.n_seg = n_seg
        self.logit_scale = clip_model.logit_scale

    def forward(self, image):
        # CLIP encode images
        image_emb = self.encode_image(image) # [BS, T, C]
        return image_emb, self.logit_scale.exp()

    def encode_image(self, image):
        bt = image.size(0) # [BS*T, C, H, W]
        b = bt // self.n_seg
        image_emb = self.visual(image) # [BS*T, C]
        image_emb = image_emb.view(b, self.n_seg, -1) # [BS, T, C]
        return image_emb

