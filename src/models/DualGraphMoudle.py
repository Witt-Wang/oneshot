from torch import nn
from torchvision import models, transforms
import torch
from collections import OrderedDict
import math
#import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F 
import numpy as np


control_dim = 

class ENCODE(nn.Module):
    

    def __init__(self,support_num,query_num,tasks,features,new_node_feature,alpha = 0.1):
        
        super(ENCODE,self).__init__()
        
        self.support_num = support_num
        self.query_num = query_num
        self.tasks = tasks
        self.alpha = alpha
        self.tem_args = support_num + query_num
        self.features = features
        self.new_node_feature =new_node_feature
        
        self.fc_subj = nn.Sequential(
            nn.Linear(self.features, self.features//2), nn.LeakyReLU())
        self.fc_obj = nn.Sequential(
            nn.Linear(self.features, self.features//2), nn.LeakyReLU())
        
       
        self.fc_fusion = nn.Sequential(
            nn.Linear(self.features, new_node_feature), nn.LeakyReLU())
        self.fc_fusion_predicate =  nn.Sequential(
            nn.Linear(self.new_node_feature*2, self.new_node_feature), nn.LeakyReLU(), nn.Linear(self.new_node_feature, self.new_node_feature),nn.LeakyReLU())

    def forward(self,object_node_feat, predicate_node_feat ):

        support_feature = object_node_feat[:,:self.support_num * 2 ,:]       
        query_feature = object_node_feat[:,self.support_num * 2:,:]
       
        
        subject_support_feature = support_feature[:,:self.support_num,:]
        object_support_feature = support_feature[:,self.support_num:,:]
        
       
        subject_support = self.fc_subj(subject_support_feature.reshape([-1,self.features]) )
        object_support = self.fc_obj(object_support_feature.reshape([-1,self.features]) )
        
        support_cat = torch.cat([subject_support,object_support],1)
        scores_support = self.fc_fusion(support_cat)
        
       
        
        subject_query_feature = query_feature[:,:self.query_num,:]
        object_query_feature = query_feature[:,self.query_num:,:]
        

        subject_query = self.fc_subj(subject_query_feature.reshape([-1,self.features]) )
        object_query = self.fc_obj(object_query_feature.reshape([-1,self.features]) )

        query_cat = torch.cat([subject_query,object_query],1)
        
        scores_query = self.fc_fusion(query_cat)
        
        new_node_feat = torch.cat([scores_support.reshape([-1,self.support_num,self.new_node_feature]),
                                   scores_query.reshape([-1,self.query_num,self.new_node_feature])],1)
        
        
        
        cat_new_predicate =  torch.cat([new_node_feat,predicate_node_feat],2)
       
        predicate_node_feat =  self.fc_fusion_predicate(cat_new_predicate.reshape([-1,self.new_node_feature*2])).reshape([-1,self.tem_args,self.new_node_feature])
        

        
        
        return predicate_node_feat


class ObejectClassifier(nn.Module):
   

    def __init__(self,support_num,query_num,tasks,features):
        """Initialize model."""
        super(ObejectClassifier,self).__init__()
        
        self.support_num = support_num
        self.query_num = query_num
        self.tasks = tasks
        
        self.features = features
      
        self.num_class = 150
        
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Sequential(
            nn.Linear(self.features, self.num_class), nn.LeakyReLU())


    def forward(self,object_node_feat ):
       
        
        support_feature = object_node_feat[:,:self.support_num * 2 ,:]       
        query_feature = object_node_feat[:,self.support_num * 2:,:]
       
        
        subject_support_feature = support_feature[:,:self.support_num,:]
        subject_support_output =self.fc(subject_support_feature.reshape([-1,self.features]) ) 

        
        object_support_feature = support_feature[:,self.support_num:,:]
        object_support_output = self.fc(object_support_feature.reshape([-1,self.features]) )  
       
        
        subject_query_feature = query_feature[:,:self.query_num,:]
        subject_query_output = self.fc(subject_query_feature.reshape([-1,self.features]) ) 
        object_query_feature = query_feature[:,self.query_num:,:]
        object_query_output = self.fc(object_query_feature.reshape([-1,self.features]) )  
        
        

        return [subject_support_output,object_support_output,subject_query_output,object_query_output]

class NodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(NodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
       
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).cuda()
    
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)
        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

       
        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class EdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(EdgeUpdateNetwork, self).__init__()
        
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
            
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        
        sim_val = F.sigmoid(self.sim_network(x_ij))
        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).cuda()
        edge_feat = edge_feat * diag_mask
        
        merge_sum = torch.sum(edge_feat, -1, True)
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).cuda()
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat

    


class ObjectNodeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 1],
                 dropout=0.0):
        super(ObjectNodeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.dropout = dropout

        # layers
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):

            layer_list['conv{}'.format(l)] = nn.Conv2d(
                in_channels=self.num_features_list[l - 1] if l > 0 else self.in_features * 3,
                out_channels=self.num_features_list[l],
                kernel_size=1,
                bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0 and l == (len(self.num_features_list) - 1):
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        self.network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        
        num_tasks = node_feat.size(0)
        num_data = node_feat.size(1)
        
        
        diag_mask = 1.0 - torch.eye(num_data).unsqueeze(0).unsqueeze(0).repeat(num_tasks, 2, 1, 1).cuda()
        edge_feat = F.normalize(edge_feat * diag_mask, p=1, dim=-1)

     
        aggr_feat = torch.bmm(torch.cat(torch.split(edge_feat, 1, 1), 2).squeeze(1), node_feat)
        node_feat = torch.cat([node_feat, torch.cat(aggr_feat.split(num_data, 1), -1)], -1).transpose(1, 2)

        node_feat = self.network(node_feat.unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        return node_feat


class ObjectEdgeUpdateNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 num_features,
                 ratio=[2, 2, 1, 1],
                 separate_dissimilarity=False,
                 dropout=0.0):
        super(ObjectEdgeUpdateNetwork, self).__init__()
        # set size
        self.in_features = in_features
        self.num_features_list = [num_features * r for r in ratio]
        self.separate_dissimilarity = separate_dissimilarity
        self.dropout = dropout

        
        layer_list = OrderedDict()
        for l in range(len(self.num_features_list)):
           
            layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                       out_channels=self.num_features_list[l],
                                                       kernel_size=1,
                                                       bias=False)
            layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                            )
            layer_list['relu{}'.format(l)] = nn.LeakyReLU()

            if self.dropout > 0:
                layer_list['drop{}'.format(l)] = nn.Dropout2d(p=self.dropout)

        layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                           out_channels=1,
                                           kernel_size=1)
        self.sim_network = nn.Sequential(layer_list)

        if self.separate_dissimilarity:
            
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                
                layer_list['conv{}'.format(l)] = nn.Conv2d(in_channels=self.num_features_list[l-1] if l > 0 else self.in_features,
                                                           out_channels=self.num_features_list[l],
                                                           kernel_size=1,
                                                           bias=False)
                layer_list['norm{}'.format(l)] = nn.BatchNorm2d(num_features=self.num_features_list[l],
                                                                )
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()

                if self.dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=self.dropout)

            layer_list['conv_out'] = nn.Conv2d(in_channels=self.num_features_list[-1],
                                               out_channels=1,
                                               kernel_size=1)
            self.dsim_network = nn.Sequential(layer_list)

    def forward(self, node_feat, edge_feat):
        
        x_i = node_feat.unsqueeze(2)
        x_j = torch.transpose(x_i, 1, 2)
        x_ij = torch.abs(x_i - x_j)
        x_ij = torch.transpose(x_ij, 1, 3)

        
        sim_val = F.sigmoid(self.sim_network(x_ij))
        if self.separate_dissimilarity:
            dsim_val = F.sigmoid(self.dsim_network(x_ij))
        else:
            dsim_val = 1.0 - sim_val


        diag_mask = 1.0 - torch.eye(node_feat.size(1)).unsqueeze(0).unsqueeze(0).repeat(node_feat.size(0), 2, 1, 1).cuda()
        edge_feat = edge_feat * diag_mask
        
        merge_sum = torch.sum(edge_feat, -1, True)
        
        edge_feat = F.normalize(torch.cat([sim_val, dsim_val], 1) * edge_feat, p=1, dim=-1) * merge_sum
        
        force_edge_feat = torch.cat((torch.eye(node_feat.size(1)).unsqueeze(0), torch.zeros(node_feat.size(1), node_feat.size(1)).unsqueeze(0)), 0).unsqueeze(0).repeat(node_feat.size(0), 1, 1, 1).cuda()
        edge_feat = edge_feat + force_edge_feat
        edge_feat = edge_feat + 1e-6
        edge_feat = edge_feat / torch.sum(edge_feat, dim=1).unsqueeze(1).repeat(1, 2, 1, 1)

        return edge_feat
    
    

class GraphNetwork(nn.Module):
    def __init__(self,
                 in_features,
                 node_features,
                 edge_features,
                 num_layers,
                 num_support,
                 num_query,
                 num_tasks,
                 dropout=0.0):
        super(GraphNetwork, self).__init__()
        
        self.in_features = in_features
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.num_support = num_support
        self.num_query = num_query
        self.num_tasks = num_tasks

        
        for l in range(self.num_layers):
            
            edge2node_net = NodeUpdateNetwork(in_features=self.in_features+control_dim if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            node2edge_net = EdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)
            
            
            object_edge2node_net = ObjectNodeUpdateNetwork(in_features=self.in_features if l == 0 else self.node_features,
                                              num_features=self.node_features,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)

            object_node2edge_net = ObjectEdgeUpdateNetwork(in_features=self.node_features,
                                              num_features=self.edge_features,
                                              separate_dissimilarity=False,
                                              dropout=self.dropout if l < self.num_layers-1 else 0.0)
            
            
            ENCODE_net = ENCODE(self.num_support ,self.num_query,self.num_tasks,features= self.edge_features,
                                new_node_feature = self.node_features)

            self.add_module('edge2node_net{}'.format(l), edge2node_net)
            self.add_module('node2edge_net{}'.format(l), node2edge_net)
            
            self.add_module('object_edge2node_net{}'.format(l),object_edge2node_net)
            self.add_module('object_node2edge_net{}'.format(l), object_node2edge_net)
            
            self.add_module('ENCODE{}'.format(l), ENCODE_net)
            
            self.object_class =ObejectClassifier(self.num_support ,self.num_query,self.num_tasks,features= self.node_features)
            

    def forward(self, node_feat, edge_feat,object_node_feat,object_edge_feat):
       
        edge_feat_list = []
        object_edge_feat_list = []
        for l in range(self.num_layers):
        
            
            object_node_feat = self._modules['object_edge2node_net{}'.format(l)](object_node_feat, object_edge_feat)
            
            predicate_node_feat = self._modules['edge2node_net{}'.format(l)](node_feat, edge_feat)
            
            
            node_feat = self._modules['ENCODE{}'.format(l)](object_node_feat, predicate_node_feat)
            
            
            
            objec_edge_feat_new = self._modules['object_node2edge_net{}'.format(l)](object_node_feat, object_edge_feat)
            edge_feat = self._modules['node2edge_net{}'.format(l)](node_feat, edge_feat)
           
            edge_feat_list.append(edge_feat)
            
            
            object_edge_feat_list.append(objec_edge_feat_new)
            
            object_out = self.object_class(object_node_feat)


        return edge_feat_list,object_edge_feat_list,object_out
    
    



