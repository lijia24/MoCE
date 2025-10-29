import torch
import torch.utils.data as data
import decord
import os
import numpy as np
from numpy.random import randint
import io
import pandas as pd
import random
from PIL import Image
import math
import copy
import pdb

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[-1])


class Video_dataset(data.Dataset):
    def __init__(self, root_path, list_file, labels_file,
                 num_segments=1, modality='RGB', new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 index_bias=1, dense_sample=False, test_clips=3,
                 num_sample=1, text_csv_file=None, spatial_label_list=None, template_label_list=None):

        self.root_path = root_path
        self.list_file = list_file
        self.labels_file = labels_file
        self.spatial_label_list = spatial_label_list
        self.template_label_list = template_label_list
        self.num_segments = num_segments
        self.modality = modality
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.index_bias = index_bias
        self.sample_range = 128
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.test_clips = test_clips
        self.num_sample = num_sample
        self.text_csv_file = text_csv_file
        self.text_descriptions = None
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.num_sample > 1:
            print('=> Using repeated augmentation...')

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self._load_text_descriptions()
        # 预处理文本描述查找，提升运行时性能
        self._preprocess_text_lookup()


    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()
    
    @property
    def spatial_classes(self):
        if self.spatial_label_list is not None:
            classes_all = pd.read_csv(self.spatial_label_list)
            return classes_all.values.tolist()
        return None
    
    @property
    def template_classes(self):
        if self.template_label_list is not None:
            classes_all = pd.read_csv(self.template_label_list)
            return classes_all.values.tolist()
        return None
    
    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3: # skip remove_missin for decording "raw_video label" type dataset_config
            if not self.test_mode:
                tmp = [item for item in tmp if int(item[1]) >= 8]
       
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))
    
    def _load_text_descriptions(self):
        """Load text descriptions from CSV file if provided"""
        
        if isinstance(self.text_csv_file, str) and os.path.exists(self.text_csv_file):
            try:
                # 使用更高效的读取方式，只读取需要的列
                text_df = pd.read_csv(self.text_csv_file, usecols=['video_path', 'description'], 
                                    dtype={'video_path': 'string', 'description': 'string'})
                # 去除可能的空值和重复项
                text_df = text_df.dropna().drop_duplicates(subset=['video_path'])
                # 创建字典映射，使用更高效的方式
                self.text_descriptions = dict(zip(text_df['video_path'].values, text_df['description'].values))
                print(f'Loaded text descriptions for {len(self.text_descriptions)} videos')
                # 释放DataFrame内存
                del text_df
            except Exception as e:
                print(f'Error loading text descriptions from {self.text_csv_file}: {e}')
                self.text_descriptions = None
        else:
            self.text_descriptions = None

    def _preprocess_text_lookup(self):
        """预处理文本描述查找，创建优化的查找表"""
        if self.text_descriptions is not None:
            # 创建一个基于video_list中实际路径的优化查找表
            self.optimized_text_lookup = {}
            missing_descriptions = 0
            for record in self.video_list:
                path = record.path
                if path in self.text_descriptions:
                    self.optimized_text_lookup[path] = self.text_descriptions[path]
                else:
                    self.optimized_text_lookup[path] = ""
                    missing_descriptions += 1
            
            if missing_descriptions > 0:
                print(f'Warning: {missing_descriptions} videos missing text descriptions')
            
            # 释放原始字典以节省内存
            del self.text_descriptions
            self.text_descriptions = self.optimized_text_lookup
        else:
            self.optimized_text_lookup = None

    def _sample_indices(self, video_list):
        # uniformly smaple num_segments frames (e.g. 8x16)
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            base_offsets = np.arange(self.num_segments) * interval
            offsets = (base_offsets + start_idx) % len(video_list)
            return np.array(offsets) + self.index_bias
        else:
        # randomly smaple num_segments frames
            seg_size = float(len(video_list) - 1) / self.num_segments
            offsets = []
            for i in range(self.num_segments):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                offsets.append(random.randint(start, end))
            return np.array(offsets) + self.index_bias

    def _get_val_indices(self, video_list):
        if self.dense_sample:
            sample_pos = max(1, 1 + len(video_list) - self.sample_range)
            t_stride = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % len(video_list) for idx in range(self.num_segments)]
            return np.array(offsets) + self.index_bias
        else:
            tick = len(video_list) / float(self.num_segments)
            offsets = [int(tick * x) % len(video_list) for x in range(self.num_segments)]
            return np.array(offsets) + self.index_bias


    def _get_test_indices(self, video_list):
        if self.dense_sample:
            # multi-clip for dense sampling
            num_clips = self.test_clips
            sample_pos = max(0, len(video_list) - self.sample_range)
            interval = self.sample_range // self.num_segments
            # start frame indexes of each num_clips
            start_list = [clip_idx * math.floor(sample_pos / (num_clips -1)) for clip_idx in range(num_clips)]
            base_offsets = np.arange(self.num_segments) * interval
            offsets = []
            for start_idx in start_list:
                offsets.extend((base_offsets + start_idx) % len(video_list))
            return np.array(offsets) + self.index_bias
        else:
            # multi-clip for uniform sampling
            num_clips = self.test_clips
            tick = len(video_list) / float(self.num_segments)
            start_list = np.linspace(0, tick - 1, num=num_clips, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [
                    int(start_idx + tick * x) % len(video_list)
                    for x in range(self.num_segments)
                ]
            return np.array(offsets) + self.index_bias


    def _decord_decode(self, video_path):
        try:
            # 对于webm格式，尝试使用不同的参数
            if video_path.lower().endswith('.webm'):
                # webm格式可能需要特殊处理
                container = decord.VideoReader(video_path, ctx=decord.cpu(0))
            else:
                container = decord.VideoReader(video_path)
        except Exception as e:
            # 如果是webm格式失败，尝试使用备用方法
            if video_path.lower().endswith('.webm'):
                try:
                    # 尝试不指定ctx参数
                    container = decord.VideoReader(video_path)
                except Exception as e2:
                    print("Failed to decode webm {} with exception: {}".format(
                        video_path, e2))
                    return None
            else:
                # print("Failed to decode {} with exception: {}".format(
                #     video_path, e))
                return None
        
        return container

    def __getitem__(self, index):
        # decode frames to video_list
        if self.modality == 'video':
            _num_retries = 10
            for i_try in range(_num_retries):
                try:
                    record = copy.deepcopy(self.video_list[index])          # copy index位置的VideoRecord变量
                    
                    
                    directory = os.path.join(self.root_path, record.path)   # 取出index位置的视频路径
                    
                    # 检查文件是否存在
                    if not os.path.exists(directory):
                        # print("Video file not found: {}".format(directory))
                        index = random.randint(0, len(self.video_list) - 1)
                        continue
                    
                    video_list = self._decord_decode(directory)             # 解码视频
                except Exception as e:
                    # print("Exception during video loading: {}".format(e))
                    index = random.randint(0, len(self.video_list) - 1)
                    continue
                # video_list = self._decord_pyav(directory)
                if video_list is None:
                    # print("Failed to decode video idx {} from {}; trial {}".format(
                    #     index, directory, i_try)
                    # )
                    index = random.randint(0, len(self.video_list) - 1)
                    continue
                
                # 检查视频是否有有效帧
                try:
                    if len(video_list) == 0:
                        # print("Video has no frames: {}".format(directory))
                        index = random.randint(0, len(self.video_list) - 1)
                        continue
                except Exception as e:
                    # print("Error checking video length: {}".format(e))
                    index = random.randint(0, len(self.video_list) - 1)
                    continue
                    
                break
        else:
            record = self.video_list[index]
            video_list = os.listdir(os.path.join(self.root_path, record.path))

        if not self.test_mode: # train/val
            segment_indices = self._sample_indices(video_list) if self.random_shift else self._get_val_indices(video_list) 
        else: # test
            segment_indices = self._get_test_indices(video_list)
        
        return self.get(record, video_list, segment_indices)


    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]


    def get(self, record, video_list, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            if self.modality == 'video':
                try:
                    # 确保索引在有效范围内
                    frame_idx = max(0, min(p - 1, len(video_list) - 1))
                    frame = video_list[frame_idx]
                    
                    # 对于webm格式，可能需要特殊处理帧数据
                    if hasattr(frame, 'asnumpy'):
                        frame_array = frame.asnumpy()
                    else:
                        # 备用方法，如果asnumpy不可用
                        frame_array = np.array(frame)
                    
                    seg_imgs = [Image.fromarray(frame_array).convert('RGB')]
                except Exception as e:
                    # print("Error processing frame {}: {}".format(p-1, e))
                    # 如果帧处理失败，使用第一帧作为备用
                    try:
                        frame_array = video_list[0].asnumpy()
                        seg_imgs = [Image.fromarray(frame_array).convert('RGB')]
                    except Exception as e2:
                        # print("Error processing backup frame: {}".format(e2))
                        # 创建一个黑色图像作为最后的备用
                        seg_imgs = [Image.new('RGB', (224, 224), (0, 0, 0))]
            else:
                seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
            if p < len(video_list):
                p += 1
        
        # Get text description if available - 使用预处理的查找表提升性能
        text_description = ""
        if self.text_descriptions is not None:
            # 直接从预处理的查找表获取，避免get()方法的开销
            text_description = self.text_descriptions[record.path]  # 预处理时已确保所有路径都存在
        
        if self.num_sample > 1:
            frame_list = []
            label_list = []
            text_list = []
            for _ in range(self.num_sample):
                process_data, record_label = self.transform((images, record.label))
                frame_list.append(process_data)
                label_list.append(record_label)
                text_list.append(text_description)
            # Return text descriptions only if text_csv_file was provided
            if self.text_csv_file is not None:
                return frame_list, label_list, text_list
            else:
                return frame_list, label_list
        else:
            process_data, record_label = self.transform((images, record.label))
            # Return text description only if text_csv_file was provided
            if self.text_csv_file is not None:
                return process_data, record_label, text_description
            else:
                return process_data, record_label

    def __len__(self):
        return len(self.video_list)
