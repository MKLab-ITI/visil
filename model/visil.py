import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from model.layers import *
from model.similarities import *

        
class Feature_Extractor(nn.Module):

    def __init__(self, network='resnet50', whiteninig=False, dims=3840):
        super(Feature_Extractor, self).__init__()
        self.normalizer = VideoNormalizer()
        
        self.cnn = models.resnet50(pretrained=True)
        
        self.rpool = RMAC()
        self.layers = {'layer1': 28, 'layer2': 14, 'layer3': 6, 'layer4': 3}
        if whiteninig or dims != 3840:
            self.pca = PCA(dims)

    def extract_region_vectors(self, x):
        tensors = []
        for nm, module in self.cnn._modules.items():
            if nm not in {'avgpool', 'fc', 'classifier'}:
                x = module(x).contiguous()
                if nm in self.layers:
                    # region_vectors = self.rpool(x)
                    s = self.layers[nm]
                    region_vectors = F.max_pool2d(x, [s, s], int(np.ceil(s / 2)))
                    region_vectors = F.normalize(region_vectors, p=2, dim=1)
                    tensors.append(region_vectors)
        x = torch.cat(tensors, 1)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = F.normalize(x, p=2, dim=-1)
        return x
    
    def forward(self, x):
        x = self.normalizer(x)
        x = self.extract_region_vectors(x)
        if hasattr(self, 'pca'):
            x = self.pca(x)
        return x

    
class ViSiLHead(nn.Module):

    def __init__(self, dims=3840, attention=True, video_comperator=True, symmetric=False):
        super(ViSiLHead, self).__init__()
        if attention:
            self.attention = Attention(dims, norm=True)
        if video_comperator:
            self.video_comperator = VideoComperator()
        self.tensor_dot = TensorDot("biok,bjpk->biopj")
        self.f2f_sim = ChamferSimilarity(axes=[3, 2], symmetric=symmetric)
        self.v2v_sim = ChamferSimilarity(axes=[2, 1], symmetric=symmetric)
        self.htanh = nn.Hardtanh()
    
    def frame_to_frame_similarity(self, query, target):
        sim = self.tensor_dot(query, target)
        return self.f2f_sim(sim)
    
    def visil_output(self, sim):
        sim = sim.unsqueeze(1)
        return self.video_comperator(sim).squeeze(1)
            
    def video_to_video_similarity(self, query, target):
        sim = self.frame_to_frame_similarity(query, target)
        if hasattr(self, 'video_comperator'):
            sim = self.visil_output(sim)
            sim = self.htanh(sim)
        return self.v2v_sim(sim)
    
    def attention_weights(self, x):
        x, weights = self.attention(x)
        return x, weights

    def prepare_tensor(self, x):
        if hasattr(self, 'attention'):
            x, _ = self.attention_weights(x)
        return x

    def apply_constrain(self):
        if hasattr(self, 'att'):
            self.att.apply_contraint()
    
    def forward(self, query, target):
        if query.ndim == 3: 
            query = query.unsqueeze(0)
        if target.ndim == 3: 
            target = target.unsqueeze(0)
        return self.video_to_video_similarity(query, target)

    
class ViSiL(nn.Module):
    
    def __init__(self, network='resnet50', pretrained=False, dims=3840,
                 whiteninig=True, attention=True, video_comperator=True, symmetric=False):
        super(ViSiL, self).__init__()
        
        if pretrained and not symmetric:
            self.cnn = Feature_Extractor('resnet50', True, 3840)
            self.visil_head = ViSiLHead(3840, True, True, False)
            self.visil_head.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'http://ndd.iti.gr/visil/visil.pth'))
        elif pretrained and symmetric:
            self.cnn = Feature_Extractor('resnet50', True, 512)
            self.visil_head = ViSiLHead(512, True, True, True)
            self.visil_head.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    'http://ndd.iti.gr/visil/visil_symmetric.pth'))
        else:
            self.cnn = Feature_Extractor(network, whiteninig, dims)
            self.visil_head = ViSiLHead(dims, attention, video_comperator, symmetric)
    
    def calculate_video_similarity(self, query, target):
        return self.visil_head(query, target)

    def calculate_f2f_matrix(self, query, target):
        return self.visil_head.frame_to_frame_similarity(query, target)

    def calculate_visil_output(self, query, target):
        sim = self.visil_head.frame_to_frame_similarity(query, target)
        return self.visil_head.visil_output(sim)
        
    def extract_features(self, video_tensor):
        features = self.cnn(video_tensor)
        features = self.visil_head.prepare_tensor(features)
        return features
