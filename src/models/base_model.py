# -*- coding: utf-8 -*-

import os
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
import yaml
import sys

from EGNN import DualGraphModule
from src.utils.test_utils import VRDTrainTester
from torch.nn import functional as F

import numpy as np


num_batch =
num_ways = 
num_shots = 
num_query = 



class VFEModule(nn.Module):
  
    def __init__(self,feature_dim, use_cuda=False, mode='train'):

        super().__init__()
        
        self.os_branch = FeaturesExtractionBranch(feature_dim,use_cuda=use_cuda)
    
        self.softmax = nn.Softmax(dim=1)
        self.mode = mode

    def forward(self, subj_feats, pred_feats, obj_feats, masks,
                subj_embs, obj_embs,attention_num,subj_sem,obj_sem):
        
        
        so_scores ,mapping_subj_feature,mapping_obj_feature= self.os_branch(
            subj_feats, obj_feats, subj_embs, obj_embs, masks,pred_feats,attention_num,subj_sem,obj_sem)
        
       
        return so_scores,mapping_subj_feature,mapping_obj_feature




class FeaturesExtractionBranch(nn.Module):


    def __init__(self,feature_dim, use_cuda=True):
      
        super().__init__()
        
     
        self.fc_subj = nn.Sequential(
            nn.Linear(2048, 1024), nn.LeakyReLU())
        
        
        self.fc_obj = nn.Sequential(
            nn.Linear(2048, 1024), nn.LeakyReLU())
        
      
        self.mapping_sem_feature = nn.Sequential(
            nn.Linear(600, 200), nn.LeakyReLU(),nn.Linear(200, 100), nn.LeakyReLU())
        
        
     
        self.mapping_subj = nn.Sequential(
            nn.Linear(1024, feature_dim), nn.LeakyReLU())
        
        self.mapping_obj = nn.Sequential(
            nn.Linear(1024, feature_dim), nn.LeakyReLU())
        
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 96, 5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(96, 128, 5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(128, 100, 8), nn.LeakyReLU()
        )

      
        self.query_conv1 = nn.Conv2d(1, 32, (1, 1), padding=(1 // 2, 0))
        self.query_conv2 = nn.Conv2d(32, 64, (1, 1), padding=(1 // 2, 0))
        self.query_conv_final = nn.Conv2d(64, 1, (1, 1), stride=(1, 1))
        
        
        self.conv1 = nn.Conv2d(1, 32, (num_shots, 1), padding=(num_shots // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (num_shots, 1), padding=(num_shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (num_shots, 1), stride=(num_shots, 1))
        
        self.pred_feats_fc = nn.Sequential(
            nn.Linear(2048, 1024), nn.LeakyReLU())
        
        self.fc = nn.Linear(1024, feature_dim)
        self.drop = nn.Dropout()
        self.final_fc = nn.Linear(feature_dim+200, feature_dim)
        
    def forward(self, subj_feats, obj_feats, subj_embs, obj_embs, masks,pred_feats,attention_num,subj_sem,obj_sem):
        
        object_vis_feature = self.fc_obj(obj_feats) 
        subject_vis_feature = self.fc_subj(subj_feats) 

        no_vis_feature = self.mask_net(masks).view(masks.shape[0], -1)
        
        subject_feature = subject_vis_feature
        
        object_feature = object_vis_feature
        
        if attention_num > 1:

            fea_att_score = self.pred_feats_fc(pred_feats)
            fea_att_score = F.relu(self.conv1(fea_att_score))  
            fea_att_score = F.relu(self.conv2(fea_att_score)) 
            fea_att_score = self.drop(fea_att_score)
            fea_att_score = self.conv_final(fea_att_score)
            
            fea_att_score = fea_att_score.repeat(1,1,attention_num,1)
            fea_att_score = F.relu(fea_att_score)

            fea_att_score = fea_att_score.view(-1, 1024) 
            
        else:

            fea_att_score = self.pred_feats_fc(pred_feats)
            fea_att_score = F.relu(self.query_conv1(fea_att_score)) 
            fea_att_score = F.relu(self.query_conv2(fea_att_score)) 
            fea_att_score = self.drop(fea_att_score)
            fea_att_score = self.query_conv_final(fea_att_score) 
            fea_att_score = F.relu(fea_att_score)
            fea_att_score = fea_att_score.view(-1, 1024) 
        
        fea_att_score = self.fc(fea_att_score)
        fea_att_score = F.relu(fea_att_score)
        
        mapping_subj_feature = self.mapping_subj(subject_feature)
        mapping_obj_feature = self.mapping_obj(object_feature)
        os_feats = mapping_subj_feature - mapping_obj_feature
       
        sem_feature = torch.cat((
            subj_sem,obj_sem
        ), dim=1)
        
        sem_feature = self.mapping_sem_feature(sem_feature)
    
        tatal_featutre = os_feats.mul(fea_att_score)
        
        tatal_featutre = torch.cat((
            os_feats.mul(fea_att_score),no_vis_feature
        ), dim=1)
        
        tatal_featutre = torch.cat((
            tatal_featutre,sem_feature
        ), dim=1)
        
        tatal_featutre = self.final_fc(tatal_featutre)
        
        
        
        return tatal_featutre ,mapping_subj_feature,mapping_obj_feature


class TrainTester(VRDTrainTester):


    def __init__(self, net,net1,feature_dim,layer_num,  use_cuda=True):
       
        super().__init__(net, net1, feature_dim,layer_num,use_cuda)

        
def train_test():
   
    
        
    feature_dim = 
    layer_num = 
    alpha =     
    net = VFEModule(feature_dim,use_cuda=True)
    net1 = DualGraphModule(feature_dim,feature_dim,feature_dim,layer_num,num_ways *num_shots,num_query,num_batch,alpha)
    
    train_tester = TrainTester(
            net,
            net1,
          feature_dim,
            layer_num
            )
    best_acc = train_tester.train()
       
        
   
    
if __name__ == "__main__":
    train_test()
