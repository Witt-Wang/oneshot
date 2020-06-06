# -*- coding: utf-8 -*-
"""A class for data loading for the task of VRD."""

import json
import random
import os

import numpy as np
import torch
import yaml
import torch.utils.data as data
from os.path import join
from PIL import Image
import os

VG_now = False

class VRDDataLoader(data.Dataset):
    def __init__(self, test_mode='train',
                 num_tasks=5,
                 num_ways=5,
                 num_shots=1,
                 num_queries=1
                 ):

        super(VRDDataLoader, self).__init__()

        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries

        self._mode = test_mode



        
        self._created_masks_path = 
        self._features_path = 
        self._json_path = 
        self._orig_image_path = 
        
        self._new_features_path =
       
        
        

        self.data = self.load_dataset(self._mode, self._json_path)     
        self.transe_embedding = self.load_transe_embedding()


    def load_transe_embedding(self):
       
        

        obj_vecs_transe = np.load('/home/wwt/PR/embedding/vg_vec_wiki.npy')
        
       
        return obj_vecs_transe
    
    
    def load_dataset(self, mode, json_path='json_annos/'):

        if mode == 'testtest':
            with open(json_path + 'new_annotation_train.json', 'r') as fid:
                annotations = json.load(fid)
            with open(json_path + 'new_annotation_test.json', 'r') as fid:
                annotations += json.load(fid)
        elif mode in ('train', 'test'):
            with open(json_path + 'new_annotation_' + mode + '.json', 'r') as fid:
                annotations = json.load(fid)

        return annotations


    def obj2onehot(self, object_tag):
        """Return the embedding of an object tag."""


        if VG_now:
            zero = np.zeros(150)
        else:
            zero = np.zeros(100)
        zero[int(object_tag)] = 1
        zero_embedding = np.array(zero).flatten()
        
        final_embedding = zero_embedding
        return final_embedding
    
    def obj2emb(self, object_tag):
        """Return the embedding of an object tag."""

        embedding = self.transe_embedding[int(object_tag)].flatten()     
        final_embedding = embedding
        return final_embedding

    def get_task_batch(self, num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        self.num_tasks = num_tasks
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries

        if seed is not None:
            random.seed(seed)

      

        support_label, query_label = [], []
        for _ in range(num_ways * num_shots):
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_label.append(label)

        for _ in range(num_ways * num_queries):
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_label.append(label)

        num_class = len(self.data)

        idx_for_data = []
        idx_for_class = []

        for t_idx in range(num_tasks):
           
            task_class_list = random.sample(list(range(num_class)), num_ways)
            idx_for_class.append(task_class_list)
            for c_idx in range(num_ways):

                num_relation = len(self.data[task_class_list[c_idx]]["relationships"])
                class_data_list = random.sample(list(range(num_relation)), num_shots + num_queries)
                idx_for_data.append(class_data_list)

                for i_idx in range(num_shots):
                   
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                
                for i_idx in range(num_queries):
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)
        query_label = torch.stack([torch.from_numpy(label).float() for label in query_label], 1)
        support_all_input, query_all_input,os_label = self.get_features(idx_for_data, idx_for_class)

        return support_all_input, query_all_input,os_label, [support_label, query_label],[idx_for_class,idx_for_data]

    def get_features(self, idx_for_data, idx_for_class):
      

        num_tasks = self.num_tasks
        num_ways = self.num_ways
        num_shots = self.num_shots
        num_queries = self.num_queries

        support_subject_feature = []
        support_uni_feature = []
        support_object_feature = []
        support_masks = []
        support_subject_embeddings = []
        support_object_embeddings = []

        query_subject_feature = []
        query_uni_feature = []
        query_object_feature = []
        query_masks = []
        query_subject_embeddings = []
        query_object_embeddings = []
        
        query_sem_subject_embeddings = []
        query_sem_object_embeddings = [] 
        support_subject_sem_embeddings = [] 
        support_object_sem_embeddings = []
        
        support_object_label = []
        support_subject_label = []
        
        query_object_label = []
        query_subject_label = []
        
        support_object_new_feature = []
        support_subject_new_feature = []
        query_object_new_feature = []
        query_subject_new_feature = []
        
        support_true_name =[]
        query_true_name = []
        for t_idx in range(num_tasks):
            task_class_list = idx_for_class[t_idx]

            for c_idx in range(num_ways):

                class_data_list = idx_for_data[num_ways * t_idx + c_idx]
                for i_idx in range(num_shots):
                    name = self.data[task_class_list[c_idx]]["relationships"][class_data_list[i_idx]]["filename"]
                    subject_name = self.data[task_class_list[c_idx]]["relationships"][class_data_list[i_idx]]["subject"]
                    object_name = self.data[task_class_list[c_idx]]["relationships"][class_data_list[i_idx]]["object"]

                    
                    
                    support_name = self.data[task_class_list[c_idx]]["predicate"]
                    support_true_name.append(support_name)
                    
                    subject_new_feature = np.load(
                        self._new_features_path + 'relationship'
                        + '_subject_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    support_subject_new_feature.append(subject_new_feature)

               
                    object_new_feature = np.load(
                        self._new_features_path + 'relationship'
                        + '_object_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    support_object_new_feature.append(object_new_feature)
                    
                    
                    
                    subject_feature = np.load(
                        self._features_path + 'relationship'
                        + '_subject_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    support_subject_feature.append(subject_feature)

                    uni_feature = np.load(
                        self._new_features_path + 'relationship'
                        + '_union_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    support_uni_feature.append(uni_feature)

                    object_feature = np.load(
                        self._features_path + 'relationship'
                        + '_object_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    support_object_feature.append(object_feature)
                    
                    

                    masks = np.load(
                        self._created_masks_path + "relationship"
                        + '_binary_masks/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    support_masks.append(masks)

                    subject_id = self.data[task_class_list[c_idx]]["relationships"][class_data_list[i_idx]]["object_id"]
                    object_id = self.data[task_class_list[c_idx]]["relationships"][class_data_list[i_idx]]["subject_id"]
                    
                    support_object_label.append(np.array(subject_id))
                    support_subject_label.append(np.array(object_id))

                    subject_embeddings = self.obj2onehot(subject_id)

                    support_subject_embeddings.append(subject_embeddings)

                    object_embeddings = self.obj2onehot(object_id)
                    support_object_embeddings.append(object_embeddings)
                    
   

                    subject_sem_embeddings = self.obj2emb(subject_id)

                    support_subject_sem_embeddings.append(subject_sem_embeddings)

                    object_sem_embeddings = self.obj2emb(object_id)
                    support_object_sem_embeddings.append(object_sem_embeddings)

                    
                for i_idx in range(num_queries):
                    name = self.data[task_class_list[c_idx]]["relationships"][class_data_list[num_shots + i_idx]][
                        "filename"]

                    subject_name = \
                    self.data[task_class_list[c_idx]]["relationships"][class_data_list[num_shots + i_idx]]["subject"]
                    object_name = \
                    self.data[task_class_list[c_idx]]["relationships"][class_data_list[num_shots + i_idx]]["object"]
                    
                    query_name = self.data[task_class_list[c_idx]]["predicate"]
                    query_true_name.append(query_name)
              
                    
                    subject_new_feature = np.load(
                        self._new_features_path + "relationship"
                        + '_subject_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    query_subject_new_feature.append(subject_new_feature)

                    object_new_feature = np.load(
                        self._new_features_path + "relationship"
                        + '_object_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    query_object_new_feature.append(object_new_feature)

                    
                    subject_feature = np.load(
                        self._features_path + "relationship"
                        + '_subject_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    query_subject_feature.append(subject_feature)

                    uni_feature = np.load(
                        self._new_features_path + "relationship"
                        + '_union_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    query_uni_feature.append(uni_feature)

                    object_feature = np.load(
                        self._features_path + "relationship"
                        + '_object_boxes_pool5/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    query_object_feature.append(object_feature)

                    masks = np.load(
                        self._created_masks_path + "relationship"
                        + '_binary_masks/' + name.replace('.jpg', '').replace('.png', '') + '.npy')
                    query_masks.append(masks)

                    subject_id = self.data[task_class_list[c_idx]]["relationships"][class_data_list[num_shots + i_idx]][
                        "subject_id"]
                    object_id = self.data[task_class_list[c_idx]]["relationships"][class_data_list[num_shots + i_idx]][
                        "object_id"]
                    

                    query_object_label.append(np.array(object_id))
                    query_subject_label.append(np.array(subject_id))

                    subject_embeddings = self.obj2onehot(subject_id)
                    query_subject_embeddings.append(subject_embeddings)

                    object_embeddings = self.obj2onehot(object_id)
                    query_object_embeddings.append(object_embeddings)
                    
                    
                    subject_sem_embeddings = self.obj2emb(subject_id)
                    query_sem_subject_embeddings.append(subject_sem_embeddings)

                    object_sem_embeddings = self.obj2emb(object_id)
                    query_sem_object_embeddings.append(object_sem_embeddings)
                    
        
        support_object_label =  torch.stack([torch.from_numpy(data).float() for data in support_object_label], 0)
        support_subject_label =  torch.stack([torch.from_numpy(data).float() for data in support_subject_label], 0)
        
        query_object_label =  torch.stack([torch.from_numpy(data).float() for data in query_object_label], 0)
        query_subject_label =  torch.stack([torch.from_numpy(data).float() for data in query_subject_label], 0)

        support_subject_feature = torch.stack([torch.from_numpy(data).float() for data in support_subject_feature], 0)
        support_uni_feature = torch.stack([torch.from_numpy(data).float() for data in support_uni_feature], 0)
        support_object_feature = torch.stack([torch.from_numpy(data).float() for data in support_object_feature], 0)
        support_masks = torch.stack([torch.from_numpy(data).float() for data in support_masks], 0)
        support_subject_embeddings = torch.stack(
            [torch.from_numpy(data).float() for data in support_subject_embeddings], 0)
        support_object_embeddings = torch.stack([torch.from_numpy(data).float() for data in support_object_embeddings],
                                                0)

        query_subject_feature = torch.stack([torch.from_numpy(data).float() for data in query_subject_feature], 0)
        query_uni_feature = torch.stack([torch.from_numpy(data).float() for data in query_uni_feature], 0)
        query_object_feature = torch.stack([torch.from_numpy(data).float() for data in query_object_feature], 0)
        query_masks = torch.stack([torch.from_numpy(data).float() for data in query_masks], 0)
        query_subject_embeddings = torch.stack([torch.from_numpy(data).float() for data in query_subject_embeddings], 0)
        query_object_embeddings = torch.stack([torch.from_numpy(data).float() for data in query_object_embeddings], 0)

        query_sem_subject_embeddings = torch.stack([torch.from_numpy(data).float() for data in query_sem_subject_embeddings], 0)
        query_sem_object_embeddings = torch.stack([torch.from_numpy(data).float() for data in query_sem_object_embeddings], 0)
        support_subject_sem_embeddings = torch.stack([torch.from_numpy(data).float() for data in support_subject_sem_embeddings], 0)
        support_object_sem_embeddings = torch.stack([torch.from_numpy(data).float() for data in support_object_sem_embeddings], 0)
    
        support_subject_new_feature = torch.stack([torch.from_numpy(data).float() for data in support_subject_new_feature], 0)
        support_object_new_feature = torch.stack([torch.from_numpy(data).float() for data in support_object_new_feature], 0)
        
        query_subject_new_feature = torch.stack([torch.from_numpy(data).float() for data in query_subject_new_feature], 0)
        query_object_new_feature = torch.stack([torch.from_numpy(data).float() for data in query_object_new_feature], 0)
        
        return [support_subject_feature, support_uni_feature, 
                support_object_feature, support_masks,
                support_subject_embeddings,
                support_object_embeddings,support_subject_new_feature,
                support_object_new_feature,support_subject_sem_embeddings,support_object_sem_embeddings],[query_subject_feature,query_uni_feature,
             query_object_feature, 
             query_masks,query_subject_embeddings, 
             query_object_embeddings,
             query_subject_new_feature,
               query_object_new_feature,query_sem_subject_embeddings,query_sem_object_embeddings],[support_subject_label,support_object_label,
     query_subject_label,query_object_label,support_true_name,query_true_name]
