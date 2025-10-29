import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import torchvision
import numpy as np
from utils.utils import init_distributed_mode, epoch_saving, best_saving, AverageMeter, reduce_tensor, accuracy, gen_label, gather_labels
from utils.logger import setup_logger
import clip

from pathlib import Path
import yaml
import pprint
from dotmap import DotMap

import datetime
import shutil
from contextlib import suppress
import pdb
from modules.video_clip import video_header, VideoCLIP
from utils.Augmentation import get_augmentation
from utils.solver import _optimizer, _lr_scheduler
from modules.text_prompt import text_prompt, text_prompt_ensemble

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    # grad_output: the gradients calculated from the last layer (have same size with the output of forward())
    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )
def cmp_loss(visual_emb, text_emb, labels, epsilon=1e-8):
    """
    CMPM损失：通过KL散度对齐跨模态投影分布
    Args:
        visual_emb: 视觉特征 [B, T, D]  # 更新了输入形状说明
        text_emb: 文本特征 [num_class, D]
        labels: 真实标签 [B, ]
        epsilon: 平滑项
    Returns:
        loss: CMPM损失值
    """
    # 对visual_emb的第二维（T维度）取平均值，将形状从[B, T, D]变为[B, D]
    visual_emb = torch.mean(visual_emb, dim=1)  # [1,2](@ref)
    
    # 特征归一化
    visual_emb = F.normalize(visual_emb, p=2, dim=-1)  # [B, D]
    text_emb = F.normalize(text_emb, p=2, dim=-1)     # [num_class, D]
    
    # 计算相似度矩阵 [B, num_class]
    sim_matrix = torch.matmul(visual_emb, text_emb.T)
    
    # 创建真实的匹配分布（one-hot），但转换为平滑分布更稳定
    identity_matrix = torch.eye(text_emb.size(0)).to(visual_emb.device) # [num_class, num_class]
    match_distribution = identity_matrix[labels] # [B, num_class] one-hot of labels
    
    # 计算投影兼容性分布 (对相似度矩阵按行做softmax)
    comp_distribution = F.softmax(sim_matrix, dim=1) # [B, num_class]
    
    # 计算KL散度损失
    loss = F.kl_div(torch.log(comp_distribution + epsilon), match_distribution, reduction='batchmean')
    
    return loss
def bidirectional_cmp_loss(visual_emb, text_emb, labels, tau=1.0, epsilon=1e-8):
    """
    双向CMPM损失：同时优化视觉→文本和文本→视觉的对齐
    Args:
        visual_emb: 视觉特征 [B, T, D]
        text_emb: 文本特征 [num_class, D]
        labels: 真实标签 [B, ]
        tau: 温度参数，控制分布尖锐程度
        epsilon: 平滑项
    Returns:
        loss: 双向CMPM损失值
    """
    # 对visual_emb的时间维度取平均 [B, T, D] -> [B, D]
    visual_emb = torch.mean(visual_emb, dim=1)
    
    # 特征归一化
    visual_emb = F.normalize(visual_emb, p=2, dim=-1)  # [B, D]
    text_emb = F.normalize(text_emb, p=2, dim=-1)     # [num_class, D]
    
    # 计算双向相似度矩阵
    sim_v2t = torch.matmul(visual_emb, text_emb.T) / tau  # [B, num_class] 视觉→文本
    sim_t2v = torch.matmul(text_emb, visual_emb.T) / tau  # [num_class, B] 文本→视觉
    
    # 创建真实匹配分布
    num_classes = text_emb.size(0)
    batch_size = visual_emb.size(0)
    
    # 视觉→文本的真实分布 (one-hot)
    identity_matrix = torch.eye(num_classes).to(visual_emb.device)
    match_distribution_v2t = identity_matrix[labels]  # [B, num_classes]
    
    # 文本→视觉的真实分布 (需要转置)
    match_distribution_t2v = torch.zeros(num_classes, batch_size).to(visual_emb.device)
    for i, label in enumerate(labels):
        match_distribution_t2v[label, i] = 1.0
    
    # 计算双向兼容性分布
    comp_distribution_v2t = F.softmax(sim_v2t, dim=1)  # [B, num_classes]
    comp_distribution_t2v = F.softmax(sim_t2v, dim=1)  # [num_classes, B]
    
    # 计算双向KL散度损失
    loss_v2t = F.kl_div(
        torch.log(comp_distribution_v2t + epsilon), 
        match_distribution_v2t, 
        reduction='batchmean'
    )
    
    loss_t2v = F.kl_div(
        torch.log(comp_distribution_t2v + epsilon), 
        match_distribution_t2v, 
        reduction='batchmean'
    )
    
    # 双向损失平均（如论文公式16）
    bidirectional_loss = 0.5 * (loss_v2t + loss_t2v)
    #print(bidirectional_loss,"  " ,loss_v2t,"   ", loss_t2v)
    return bidirectional_loss
def consistency_loss(visual_emb, text_emb, labels):
    """
    跨模态一致性损失：最小化匹配样本对间的特征距离
    Args:
        visual_emb: 视觉特征 [B,T, D]
        text_emb: 文本特征 [num_class, D]
        labels: 真实标签 [B, ]
    Returns:
        loss: 一致性损失值
    """
    visual_emb = F.normalize(visual_emb, p=2, dim=-1)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    
    # 为每个视觉样本获取其对应的正文本样本特征 [B, D]
    positive_text = text_emb[labels] # Index using labels
    visual_emb_pooled = torch.mean(visual_emb, dim=1) 
    # 计算余弦相似度（或距离）
    cosine_sim = F.cosine_similarity(visual_emb_pooled, positive_text, dim=-1) # [B, ]
    # 最大化相似度等价于最小化 (1 - similarity)
    loss = 1 - torch.mean(cosine_sim)
    
    return loss


allgather = AllGather.apply

def update_dict(dict):
    new_dict = {}
    for k, v in dict.items():
        new_dict[k.replace('module.', '')] = v
    return new_dict

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', type=str, default='clip.yaml', help='global config file')
    parser.add_argument('--log_time', default='001')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')                        
    parser.add_argument("--local_rank", type=int,
                        help='local rank for DistributedDataParallel')
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )                        
    args = parser.parse_args()
    return args



def main(args):
    global best_prec1
    """ Training Program """
    init_distributed_mode(args)
    if args.distributed:
        print('[INFO] turn on distributed train', flush=True)
    else:
        print('[INFO] turn off distributed train', flush=True)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    working_dir = os.path.join(config['data']['output_path'], config['data']['dataset'], config['network']['arch'] , args.log_time)


    if dist.get_rank() == 0:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, working_dir)
        shutil.copy('train.py', working_dir)
        shutil.copy('modules/video_clip.py', working_dir)


    # build logger, print env and config
    logger = setup_logger(output=working_dir,
                          distributed_rank=dist.get_rank(),
                          name=f'MoTE')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("storing name: {}".format(working_dir))



    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        cudnn.benchmark = True
        # cudnn.deterministic = True

    # fix the seed for reproducibility
    seed = config.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    # get fp16 model and weight
    model_clip, clip_state_dict = clip.load(
        config.network.arch,
        device='cpu',jit=False,
        internal_modeling=config.network.tm,
        T=config.data.num_segments,
        dropout=config.network.drop_out,
        emb_dropout=config.network.emb_dropout,
        pretrain=config.network.init,
        joint_st = config.network.joint_st) # Must set jit=False for training  ViT-B/32

    # Data Augmentations
    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)
    logger.info('train transforms: {}'.format(transform_train.transforms))
    logger.info('val transforms: {}'.format(transform_val.transforms))

    if args.precision == "amp" or args.precision == "fp32":
        model_clip = model_clip.float()

    if config.data.dataset == 'charades':
        from datasets.charades import Video_dataset
        train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense,
            fps=config.data.fps)
        val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, test_mode=True, dense_sample=config.data.dense)            
    else:
        from datasets.video import Video_dataset
        train_data = Video_dataset(
            config.data.train_root, config.data.train_list,
            config.data.label_list, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
            transform=transform_train, dense_sample=config.data.dense,
            text_csv_file=getattr(config.data, 'train_desc_list', None),
            spatial_label_list=getattr(config.data, 'spatial_label_list', None),
            template_label_list=getattr(config.data, 'template_label_list', None))
        val_data = Video_dataset(
            config.data.val_root, config.data.val_list, config.data.label_list,
            random_shift=False, num_segments=config.data.num_segments,
            modality=config.data.modality,
            image_tmpl=config.data.image_tmpl,
            transform=transform_val, dense_sample=config.data.dense,
            text_csv_file=getattr(config.data, 'val_desc_list', None))


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)                       
    train_loader = DataLoader(train_data,
        batch_size=config.data.batch_size, num_workers=config.data.workers,
        sampler=train_sampler, drop_last=False)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)
    val_loader = DataLoader(val_data,
        batch_size=config.data.batch_size,num_workers=config.data.workers,
        sampler=val_sampler, drop_last=False)

    loss_type = config.solver.loss_type
    if loss_type == 'CE':
        print('============= Using CE Loss ==============')
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # ============= generate class features ==============
    print('============= Start encoding class features ===========')
    classes = text_prompt_ensemble(train_data.classes)
    if config.data.spatial_label_list:
        spatial_classes = text_prompt_ensemble(train_data.spatial_classes)
    if config.data.template_label_list:
        template_classes = text_prompt_ensemble(train_data.template_classes)
    
    n_class = classes[0].size(0)
    print(f"有{n_class}类")
    model_clip.cuda()
    model_clip.eval()
    def get_cls_feature(classes):
        with torch.no_grad():
            # @zmhh_h multi text prompts
            cls_feature_list = [model_clip.encode_text(classes[i].cuda(), return_token=True)[0] for i in range(len(classes))]
            for cls_feature in cls_feature_list:
                cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
            cls_feature = torch.stack(cls_feature_list, 0).mean(0)
            cls_feature /= cls_feature.norm(dim=-1, keepdim=True)
            return cls_feature
    cls_feature=get_cls_feature(classes)
    if config.data.spatial_label_list:
        spatial_cls_feature=get_cls_feature(spatial_classes)
    if config.data.template_label_list:
        template_cls_feature=get_cls_feature(template_classes)
    
    print('============= End encoding class features ===========')
    
    model = VideoCLIP(model_clip, config.data.num_segments)
    #del model_clip

    # Temporal Aggregation Module
    video_head = video_header(
        config.network.sim_header,
        config.network.interaction,
        clip_state_dict,
        config.network.num_experts,
        cls_feature)
    
    start_epoch = config.solver.start_epoch
    
    if config.pretrain:#False
        if os.path.isfile(config.pretrain):
            logger.info("=> loading checkpoint '{}'".format(config.pretrain))
            checkpoint = torch.load(config.pretrain, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            video_head.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))
    
    if config.resume:#False
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume, map_location='cpu')
            model.load_state_dict(update_dict(checkpoint['model_state_dict']))
            video_head.load_state_dict(update_dict(checkpoint['fusion_model_state_dict']))
            start_epoch = checkpoint['epoch'] + 1
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(config.evaluate, checkpoint['epoch']))
            del checkpoint
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.pretrain))

    if config.network.fix_video:#False
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad_(False)

    # ============== count trainable parameters ==============
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    
    # ============== set optimizer ==============
    optimizer = _optimizer(config, model, video_head)
    lr_scheduler = _lr_scheduler(config, optimizer)

    if args.distributed:
        model = DistributedDataParallel(model.cuda(), device_ids=[args.gpu])
        if config.network.sim_header == "None" and config.network.interaction in ['DP']:
            video_head_nomodule = video_head
        else:
            video_head = DistributedDataParallel(video_head.cuda(), device_ids=[args.gpu], find_unused_parameters=True)
            video_head_nomodule = video_head.module

    scaler = GradScaler() if args.precision == "amp" else None

    best_prec1 = 0.0
    if config.solver.evaluate:
        logger.info(("===========evaluate==========="))
        prec1 = validate(start_epoch, val_loader, device, model, video_head, config, cls_feature, logger,spatial_cls_feature,template_cls_feature)
        return
    

    for epoch in range(start_epoch, config.solver.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)        

        train(model, video_head, train_loader, optimizer, criterion, scaler,
              epoch, device, lr_scheduler, config, cls_feature, logger,model_clip,spatial_cls_feature,template_cls_feature)

        if (epoch+1) % config.logging.eval_freq == 0:
            prec1 = validate(epoch, val_loader, device, model, video_head, config, cls_feature, logger,model_clip,spatial_cls_feature,template_cls_feature)

            if dist.get_rank() == 0:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                logger.info('Testing: {}/{}'.format(prec1,best_prec1))
                logger.info('Saving:')
                filename = "{}/last_model.pt".format(working_dir)

                epoch_saving(epoch, model.module, video_head_nomodule, optimizer, filename)
                if is_best:
                    best_saving(working_dir, epoch, model.module, video_head_nomodule, optimizer)


def train(model, video_head, train_loader, optimizer, criterion, scaler,
          epoch, device, lr_scheduler, config, text_embedding, logger,model_clip,spatial_cls_feature=None,template_cls_feature=None):
    """ train a epoch """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    img_losses = AverageMeter()
    text_losses = AverageMeter()

    #model.train()
    video_head.train()
    autocast = torch.cuda.amp.autocast if args.precision == 'amp' else suppress
    end = time.time()
    # 检查数据集是否包含文本描述
    sample_batch = next(iter(train_loader))
    has_descriptions = len(sample_batch) == 3
    
    for i, batch in enumerate(train_loader):
        
        if has_descriptions:
            images, list_id, descriptions = batch
        else:
            images, list_id = batch
            descriptions=None
        desc_embeds=None
        if descriptions is not None:
            desc_embeds=model_clip.encode_text(clip.tokenize(descriptions,truncate=True).cuda())[0]
            desc_embeds=desc_embeds / desc_embeds.norm(dim=-1, keepdim=True)
        else:
            print("no description")
            desc_embeds=model_clip.encode_text(clip.tokenize(["a photo of a scene, place, background,, environment, no people, no animals,without person, nobody"]*256,truncate=True).cuda())[0]
            
            desc_embeds=desc_embeds / desc_embeds.norm(dim=-1, keepdim=True)
        if config.solver.type != 'monitor':
            if (i + 1) == 1 or (i + 1) % 10 == 0:
                lr_scheduler.step(epoch + i / len(train_loader))
        # lr_scheduler.step()

        data_time.update(time.time() - end)
        # b t 3 h w
        images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])  # b t 3 h w
        b,t,c,h,w = images.size()

        images= images.view(-1, c, h, w) 
        
        with autocast():
            if config.solver.loss_type in ['CE']:
                # image_embedding: [B, frames, 512],logit_scale是一个值
                image_embedding, logit_scale= model(images)
                if epoch==0 and i==0:
                    flops1, params1 = profile(model, inputs=(images,))
                    flops2, params2 = profile(video_head, inputs=(image_embedding, text_embedding,desc_embeds, list_id.to(device),spatial_cls_feature,template_cls_feature,))
                    logger.info(f"CLIP: {flops1/(10**9)} GFLOPs, video_head: {flops2/(10**9)} GFLOPs, Total: {flops1/(10**9)+flops2/(10**9)} GFLOPs" )
                    logger.info(f"CLIP params: {params1/(10**6)} M, video head: {params2/(10**6)} M, Total: {params1/(10**6)+params2/(10**6)} M" )
                    
                logits_exp, mse_loss, logits_t,vid_emb_expert,cls_emb = video_head(image_embedding, text_embedding,desc_embeds, list_id.to(device),spatial_cls_feature,template_cls_feature)
               
                loss_exp= criterion(logit_scale * logits_exp, list_id.to(device))

                Consistency_loss=consistency_loss(vid_emb_expert,cls_emb,list_id.to(device))
                loss_wmr =bidirectional_cmp_loss(vid_emb_expert,cls_emb,list_id.to(device))
               
                logger.info(f"loss_exp: {loss_exp},Consistency_loss:{Consistency_loss},loss_wmr:{loss_wmr}")

                loss = 2*loss_exp  + Consistency_loss +0.1*loss_wmr
            else:
                raise NotImplementedError
            
            loss = loss / config.solver.grad_accumulation_steps

        if scaler is not None:
            # back propagation
            scaler.scale(loss).backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                scaler.step(optimizer)  
                scaler.update()  
                optimizer.zero_grad()
        else:
            # back propagation
            loss.backward()
            if (i + 1) % config.solver.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        losses.update(loss.item(), logits_exp.size(0))


        batch_time.update(time.time() - end)
        end = time.time()                
        cur_iter = epoch * len(train_loader) + i
        max_iter = config.solver.epochs * len(train_loader)
        eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
        eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))        

        if i % config.logging.print_freq == 0:
            logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                             epoch, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                             lr=optimizer.param_groups[-1]['lr'])))




def validate(epoch, val_loader, device, model, video_head, config, text_embedding, logger,model_clip,spatial_cls_feature=None,template_cls_feature=None):
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    video_head.eval()
    
    with torch.no_grad():
        # 检查数据集是否包含文本描述
        sample_batch = next(iter(val_loader))
        has_descriptions = len(sample_batch) == 3
        
        for i, batch in enumerate(val_loader):
            if has_descriptions:
                image, class_id, descriptions = batch
            else:
                image, class_id = batch
                descriptions=None
            desc_embeds=None
            
            if descriptions is not None:
                desc_embeds=model_clip.encode_text(clip.tokenize(descriptions,truncate=True).cuda())[0]
                desc_embeds=desc_embeds / desc_embeds.norm(dim=-1, keepdim=True)
            else:
                print("no description")
                desc_embeds=model_clip.encode_text(clip.tokenize(["a photo of a scene, place, background,, environment, no people, no animals,without person, nobody"]*256,truncate=True).cuda())[0]
           
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            class_id = class_id.to(device)
            text_embedding = text_embedding.to(device)
            image = image.to(device).view(-1, c, h, w) # [BS*T, C, H, W]

            image_embedding = model.module.encode_image(image) # [BS, T, C]
            similarity = video_head(image_embedding, text_embedding,desc_embeds, class_id,spatial_cls_feature,template_cls_feature) # [BS, n_cls]


            prec = accuracy(similarity, class_id, topk=(1, 5))
            prec1 = reduce_tensor(prec[0])
            prec5 = reduce_tensor(prec[1])

            top1.update(prec1.item(), class_id.size(0))
            top5.update(prec5.item(), class_id.size(0))

            if i % config.logging.print_freq == 0:
                logger.info(
                    ('Test: [{0}/{1}]\t'
                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                         i, len(val_loader), top1=top1, top5=top5)))
    logger.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5)))
    return top1.avg




if __name__ == '__main__':
    args = get_parser() 
    main(args)

