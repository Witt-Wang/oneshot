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





class VFEModule(nn.Module):
  
    def __init__(self,in_feature,out_feature,class_num, use_cuda):

        super().__init__()

        self.os_branch = ObjectSubjectBranch(in_feature, out_feature, class_num, use_cuda=use_cuda)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, subj_feats, pred_feats, obj_feats, masks,
                subj_embs, obj_embs, attention_num):
        so_scores, mapping_subj_feature, mapping_obj_feature = self.os_branch(
            subj_feats, obj_feats, subj_embs, obj_embs, masks, pred_feats, attention_num)

        return so_scores, mapping_subj_feature, mapping_obj_feature




class FeaturesExtractionBranch(nn.Module):


    def __init__(self,in_feature,out_feature,class_num,use_cuda=True):

        self.in_feature = in_feature

        self.fc_subj = nn.Sequential(
            nn.Linear(in_feature, in_feature // 2), nn.LeakyReLU())

        self.fc_obj = nn.Sequential(
            nn.Linear(in_feature, in_feature // 2), nn.LeakyReLU())

        self.mapping_subj = nn.Sequential(
            nn.Linear(in_feature // 2 + class_num, out_feature), nn.LeakyReLU())

        self.mapping_obj = nn.Sequential(
            nn.Linear(in_feature // 2 + class_num, out_feature), nn.LeakyReLU())

        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 96, 5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(96, 128, 5, stride=2, padding=2), nn.LeakyReLU(),
            nn.Conv2d(128, 100, 8), nn.LeakyReLU()
        )

        self.query_conv1 = nn.Conv2d(1, 32, (1, 1), padding=(1 // 2, 0))
        self.query_conv2 = nn.Conv2d(32, 64, (1, 1), padding=(1 // 2, 0))
        self.query_conv_final = nn.Conv2d(64, 1, (1, 1), stride=(1, 1))

        self.conv1 = nn.Conv2d(1, 32, (1, 1), padding=(1 // 2, 0))
        self.conv2 = nn.Conv2d(32, 64, (1, 1), padding=(1 // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (1, 1), stride=(1, 1))

        self.pred_feats_fc = nn.Sequential(
            nn.Linear(in_feature, in_feature // 2), nn.LeakyReLU())

        self.fc = nn.Linear(in_feature // 2, out_feature)
        self.drop = nn.Dropout()
        self.final_fc = nn.Sequential(nn.Linear(out_feature + 100, out_feature), nn.LeakyReLU())
        
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

            fea_att_score = fea_att_score.repeat(1, 1, attention_num, 1)
            fea_att_score = F.relu(fea_att_score)

            fea_att_score = fea_att_score.view(-1, self.in_feature // 2)

        else:

            fea_att_score = self.pred_feats_fc(pred_feats)
            fea_att_score = F.relu(self.query_conv1(fea_att_score))
            fea_att_score = F.relu(self.query_conv2(fea_att_score))
            fea_att_score = self.drop(fea_att_score)
            fea_att_score = self.query_conv_final(fea_att_score)
            fea_att_score = F.relu(fea_att_score)

            fea_att_score = fea_att_score.view(-1, self.in_feature // 2)

        fea_att_score = self.fc(fea_att_score)
        fea_att_score = F.relu(fea_att_score)

        subject_feature = torch.cat((
            subject_feature, subj_embs
        ), dim=1)

        object_feature = torch.cat((
            object_feature, obj_embs
        ), dim=1)

        mapping_subj_feature = self.mapping_subj(subject_feature)
        mapping_obj_feature = self.mapping_obj(object_feature)
        os_feats = mapping_subj_feature - mapping_obj_feature



        tatal_featutre = torch.cat((
            os_feats, no_vis_feature
        ), dim=1)

        tatal_featutre = self.final_fc(tatal_featutre)

        tatal_featutre = tatal_featutre.mul(fea_att_score)

        return tatal_featutre, mapping_subj_feature, mapping_obj_feature


class TrainTester(VRDTrainTester):


    def __init__(self, net,net1,feature_dim,layer_num,  use_cuda=True):
       
        super().__init__(net, net1, feature_dim,layer_num,use_cuda)

        
def train_test(config):
    num_batch = config['num_batch']
    num_ways = config['num_ways']
    num_shots = config['num_shots']
    num_query = config['num_query']
    feature_dim = config['feature_dim']
    num_layers = config['num_layers']
    in_feature = config['in_feature']
    out_feature = config['out_feature']
    class_num = config['class_num']



    net = VFEModule(in_feature, out_feature, class_num, use_cuda=True)
    net1 = GraphNetwork(feature_dim, feature_dim, feature_dim, num_layers, num_ways * num_shots, num_query, num_batch,
                        0.1)
    
    train_tester = TrainTester(
            net,
            net1,
        config
            )
    train_tester.train()

    train_tester.test(partition='test')
       
        
   
    
if __name__ == "__main__":
    train_test()
