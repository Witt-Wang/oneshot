# -*- coding: utf-8 -*-


import os
import json
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
import sys

from src.dataload import VRDDataLoader




class VRDTrainTester():


    def __init__(self, net, net1, config, use_cuda=True):


        self.config = config
        self.use_cuda = use_cuda

        self.enc_module = net.cuda()
        self.gnn_module = net1.cuda()

        self.enc_module = nn.DataParallel(self.enc_module)
        self.gnn_module = nn.DataParallel(self.gnn_module)

        self.data_loader = VRDDataLoader("train")
        self.module_params = list(self.enc_module.parameters()) + list(self.gnn_module.parameters())
        self.optimizer = optim.Adam(params=self.module_params,
                                    lr=1e-3, weight_decay=1e-6, )


        self.edge_loss = nn.BCELoss(reduction='none')
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.node_loss = nn.CrossEntropyLoss(reduction='none')

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def train(self):

        num_ways_train = self.config['num_ways']
        num_shots_train = self.config['num_shots']
        support_shot_nums = num_shots_train
        query_shot_nums = 1
        meta_batch_size = self.config['num_batch']
        num_layers = self.config['num_layers']

        train_iteration = self.config['train_iteration']
        lr = 1e-3
        test_interval = 150
        predicate_detection = self.config['predicate_detection']

        feature_dim = self.config['feature_dim']

        val_acc = self.val_acc


        num_supports = num_ways_train * num_shots_train
        num_queries = num_ways_train * 1
        num_samples = num_supports + num_queries


        support_edge_mask = torch.zeros(meta_batch_size, num_samples, num_samples).cuda()
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask
        evaluation_mask = torch.ones(meta_batch_size, num_samples, num_samples).cuda()

        object_support_edge_mask = torch.zeros(meta_batch_size, 2 * num_samples, 2 * num_samples).cuda()
        object_query_edge_mask = 1 - object_support_edge_mask
        object_evaluation_mask = torch.ones(meta_batch_size, 2 * num_samples, 2 * num_samples).cuda()


        for iter in range(self.global_step + 1, train_iteration + 1):

            self.optimizer.zero_grad()

            self.global_step = iter

            support_all_input, query_all_input, os_label, [support_label, query_label], [idx_for_class,
                                                                                         idx_for_data] = self.data_loader.get_task_batch(
                num_tasks=meta_batch_size,
                num_ways=num_ways_train,
                num_shots=num_shots_train,
                seed=iter)

            os_support_full_label = torch.cat(
                [os_label[0].view(meta_batch_size, -1).cuda(), os_label[1].view(meta_batch_size, -1).cuda()], 1)
            os_query_full_label = torch.cat(
                [os_label[2].view(meta_batch_size, -1).cuda(), os_label[3].view(meta_batch_size, -1).cuda()], 1)

            os_full_label = torch.cat([os_support_full_label, os_query_full_label], 1)
            os_full_edge = self.label2edge(os_full_label)

            support_label = support_label.cuda()
            query_label = query_label.cuda()

            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            self.enc_module.train()
            self.gnn_module.train()


            support_predicate_input = support_all_input[1].view(meta_batch_size * num_ways_train, 1, num_shots_train,
                                                                -1).cuda()
            query_predicate_input = query_all_input[1].view(meta_batch_size * num_ways_train, 1, 1, -1).cuda()




            support_subject_emb = support_all_input[4].cuda()
            support_object_emb = support_all_input[5].cuda()
            query_subject_emb = query_all_input[4].cuda()
            query_object_emb = query_all_input[5].cuda()


            all_support_full_data, support_mapping_subj_feature, support_mapping_obj_feature = self.enc_module(
                support_all_input[0].cuda(),
                support_predicate_input,
                support_all_input[2].cuda(),
                support_all_input[3].cuda(),
                support_subject_emb, support_object_emb, support_shot_nums)

            all_query_full_data, query_mapping_subj_feature, query_mapping_obj_feature = self.enc_module(
                query_all_input[0].cuda(),
                query_predicate_input,
                query_all_input[2].cuda(),
                query_all_input[3].cuda(),
                query_subject_emb, query_object_emb, query_shot_nums)

            all_support_full_data = all_support_full_data.view(meta_batch_size, num_supports, feature_dim)
            all_query_full_data = all_query_full_data.view(meta_batch_size, num_queries, feature_dim)

            full_data = torch.cat([all_support_full_data, all_query_full_data], 1)  #

            support_subject_input = support_mapping_subj_feature.view(meta_batch_size, num_supports, feature_dim).cuda()
            support_object_input = support_mapping_obj_feature.view(meta_batch_size, num_supports, feature_dim).cuda()
            support_full_data = torch.cat([support_subject_input, support_object_input], 1)

            query_subject_input = query_mapping_subj_feature.view(meta_batch_size, num_queries, feature_dim).cuda()
            query_object_input = query_mapping_obj_feature.view(meta_batch_size, num_queries, feature_dim).cuda()
            query_full_data = torch.cat([query_subject_input, query_object_input], 1)

            feature_full_data = torch.cat([support_full_data, query_full_data], 1)

            full_logit_layers, object_full_logit_layers, object_out = self.gnn_module(node_feat=full_data,
                                                                                      edge_feat=init_edge,
                                                                                      object_node_feat=feature_full_data,
                                                                                      object_edge_feat=os_full_edge)

            support_subject_output_label = object_out[0]
            support_object_output_label = object_out[1]
            query_subject_output_label = object_out[2]
            query_object_output_label = object_out[3]

            subject_support_loss = self.criterion(object_out[0], os_label[0].cuda().long())
            object_support_loss = self.criterion(object_out[1], os_label[1].cuda().long())
            subject_query_loss = self.criterion(object_out[2], os_label[2].cuda().long())

            object_query_loss = self.criterion(object_out[3], os_label[3].cuda().long())

            object_class_loss = (
                                            subject_support_loss + object_support_loss + subject_query_loss + object_query_loss) / 4

            full_edge_loss_layers = [self.edge_loss((1 - full_logit_layer[:, 0]), (1 - full_edge[:, 0])) for
                                     full_logit_layer in full_logit_layers]

            object_full_edge_loss_layers = [self.edge_loss((1 - full_logit_layer[:, 0]), (1 - os_full_edge[:, 0])) for
                                            full_logit_layer in object_full_logit_layers]


            object_pos_query_edge_loss_layers = [torch.sum(
                full_edge_loss_layer * object_query_edge_mask * os_full_edge[:,
                                                                0] * object_evaluation_mask) / torch.sum(
                object_query_edge_mask * os_full_edge[:, 0] * object_evaluation_mask) for full_edge_loss_layer in
                                                 object_full_edge_loss_layers]
            object_neg_query_edge_loss_layers = [torch.sum(full_edge_loss_layer * object_query_edge_mask * (
                        1 - os_full_edge[:, 0]) * object_evaluation_mask) / torch.sum(
                object_query_edge_mask * (1 - os_full_edge[:, 0])) for full_edge_loss_layer in
                                                 object_full_edge_loss_layers]

            object_query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                             (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                             zip(object_pos_query_edge_loss_layers, object_neg_query_edge_loss_layers)]

            pos_query_edge_loss_layers = [
                torch.sum(full_edge_loss_layer * query_edge_mask * full_edge[:, 0] * evaluation_mask) / torch.sum(
                    query_edge_mask * full_edge[:, 0]) for full_edge_loss_layer in full_edge_loss_layers]
            neg_query_edge_loss_layers = [
                torch.sum(full_edge_loss_layer * query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) / torch.sum(
                    query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) for full_edge_loss_layer in
                full_edge_loss_layers]
            query_edge_loss_layers = [pos_query_edge_loss_layer + neg_query_edge_loss_layer for
                                      (pos_query_edge_loss_layer, neg_query_edge_loss_layer) in
                                      zip(pos_query_edge_loss_layers, neg_query_edge_loss_layers)]


            full_edge_accr_layers = [self.hit(full_logit_layer, 1 - full_edge[:, 0].long()) for full_logit_layer in
                                     full_logit_layers]
            query_edge_accr_layers = [torch.sum(full_edge_accr_layer * query_edge_mask * evaluation_mask) / torch.sum(
                query_edge_mask * evaluation_mask) for full_edge_accr_layer in full_edge_accr_layers]


            query_node_pred_layers = [torch.bmm(full_logit_layer[:, 0, num_supports:, :num_supports],
                                                self.one_hot_encode(num_ways_train, support_label.long())) for
                                      full_logit_layer in
                                      full_logit_layers]
            query_node_accr_layers = [
                torch.eq(torch.max(query_node_pred_layer, -1)[1], query_label.long()).float().mean() for
                query_node_pred_layer in query_node_pred_layers]


            query_nodes_acc = query_node_accr_layers[-1]


            total_loss_layers = query_edge_loss_layers

            total_loss = []
            for l in range(num_layers - 1):
                total_loss += [total_loss_layers[l].view(-1) * 0.5]
            total_loss += [total_loss_layers[-1].view(-1) * 1.0]
            total_loss = torch.mean(torch.cat(total_loss, 0))

            object_total_loss_layers = object_query_edge_loss_layers
            object_total_loss = []
            for l in range(num_layers - 1):
                object_total_loss += [object_total_loss_layers[l].view(-1) * 0.5]
            object_total_loss += [object_total_loss_layers[-1].view(-1) * 1.0]
            object_total_loss = torch.mean(torch.cat(object_total_loss, 0))


            total_loss = total_loss + object_total_loss + 1 * object_class_loss

            total_loss.backward()

            self.optimizer.step()

            print('train/edge_loss'+ str( query_edge_loss_layers[-1]))

            print("\t train/edge_accr" + str(query_edge_accr_layers[-1]))
            print("\t train/node_accr" + str(query_nodes_acc)+"\n")


            if self.global_step % test_interval == 0:

                val_acc = self.test(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    test_acc = self.test(partition='test')
                    self.test_acc = test_acc
                    self.val_acc = val_acc
                    is_best = 1

                print("val/best_accr"+ str(self.val_acc))
                print("\t val/best_accr" + str(self.test_acc)+"\n" )



    def test(self, partition='test'):

        num_ways_test = self.config['num_ways']
        num_shots_test = self.config['num_shots']
        test_batch_size = self.config['num_batch']

        test_iteration = 150
        support_shot_nums = num_shots_test
        query_shot_nums = 1
        val_data_loader = VRDDataLoader("test")
        log_flag = True
        best_acc = 0

        predicate_detection = self.config['predicate_detection']

        num_supports = num_ways_test * num_shots_test
        num_queries = num_ways_test * 1
        num_samples = num_supports + num_queries

        support_edge_mask = torch.zeros(test_batch_size, num_samples, num_samples).cuda()
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask
        evaluation_mask = torch.ones(test_batch_size, num_samples, num_samples).cuda()

        query_edge_losses = []
        query_edge_accrs = []
        query_node_accrs = []

        total_nodes_acc = []
        test_acc_vec = []


        feature_dim = self.config['feature_dim']


        for iter in range(test_iteration // test_batch_size):

            support_all_input, query_all_input, os_label, [support_label, query_label], [idx_for_class,
                                                                                         idx_for_data] = val_data_loader.get_task_batch(
                num_tasks=test_batch_size,
                num_ways=num_ways_test,
                num_shots=num_shots_test,
                seed=iter)

            os_support_full_label = torch.cat(
                [os_label[0].view(test_batch_size, -1).cuda(), os_label[1].view(test_batch_size, -1).cuda()], 1)
            os_query_full_label = torch.cat(
                [os_label[2].view(test_batch_size, -1).cuda(), os_label[3].view(test_batch_size, -1).cuda()], 1)

            os_full_label = torch.cat([os_support_full_label, os_query_full_label], 1)
            os_full_edge = self.label2edge(os_full_label)

            support_label = support_label.cuda()
            query_label = query_label.cuda()
            full_label = torch.cat([support_label, query_label], 1)
            full_edge = self.label2edge(full_label)

            init_edge = full_edge.clone()
            init_edge[:, :, num_supports:, :] = 0.5
            init_edge[:, :, :, num_supports:] = 0.5
            for i in range(num_queries):
                init_edge[:, 0, num_supports + i, num_supports + i] = 1.0
                init_edge[:, 1, num_supports + i, num_supports + i] = 0.0

            # set as train mode
            self.enc_module.eval()
            self.gnn_module.eval()


            support_subject_emb = support_all_input[4].cuda()
            support_object_emb = support_all_input[5].cuda()
            query_subject_emb = query_all_input[4].cuda()
            query_object_emb = query_all_input[5].cuda()


            support_predicate_input = support_all_input[1].view(test_batch_size * num_ways_test, 1, num_shots_test,
                                                                -1).cuda()
            query_predicate_input = query_all_input[1].view(test_batch_size * num_ways_test, 1, 1, -1).cuda()

            all_support_full_data, support_mapping_subj_feature, support_mapping_obj_feature = self.enc_module(
                support_all_input[0].cuda(),
                support_predicate_input,
                support_all_input[2].cuda(),
                support_all_input[3].cuda(),
                support_subject_emb, support_object_emb, support_shot_nums)

            all_query_full_data, query_mapping_subj_feature, query_mapping_obj_feature = self.enc_module(
                query_all_input[0].cuda(),
                query_predicate_input,
                query_all_input[2].cuda(),
                query_all_input[3].cuda(),
                query_subject_emb, query_object_emb, query_shot_nums)

            support_subject_input = support_mapping_subj_feature.view(test_batch_size, num_supports, feature_dim).cuda()
            support_object_input = support_mapping_obj_feature.view(test_batch_size, num_supports, feature_dim).cuda()
            support_full_data = torch.cat([support_subject_input, support_object_input], 1)

            query_subject_input = query_mapping_subj_feature.view(test_batch_size, num_queries, feature_dim).cuda()
            query_object_input = query_mapping_obj_feature.view(test_batch_size, num_queries, feature_dim).cuda()
            query_full_data = torch.cat([query_subject_input, query_object_input], 1)

            feature_full_data = torch.cat([support_full_data, query_full_data], 1)

            all_support_full_data = all_support_full_data.view(test_batch_size, num_supports, feature_dim)

            all_query_full_data = all_query_full_data.view(test_batch_size, num_queries, feature_dim)

            full_data = torch.cat([all_support_full_data, all_query_full_data], 1)

            full_logit_all, object_full_logit_layers, object_out = self.gnn_module(node_feat=full_data,
                                                                                   edge_feat=init_edge,
                                                                                   object_node_feat=feature_full_data,
                                                                                   object_edge_feat=os_full_edge)



            full_logit = full_logit_all[-1]

            full_edge_loss = self.edge_loss(1 - full_logit[:, 0], 1 - full_edge[:, 0])

            query_edge_loss = torch.sum(full_edge_loss * query_edge_mask * evaluation_mask) / torch.sum(
                query_edge_mask * evaluation_mask)

            pos_query_edge_loss = torch.sum(
                full_edge_loss * query_edge_mask * full_edge[:, 0] * evaluation_mask) / torch.sum(
                query_edge_mask * full_edge[:, 0] * evaluation_mask)
            neg_query_edge_loss = torch.sum(
                full_edge_loss * query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask) / torch.sum(
                query_edge_mask * (1 - full_edge[:, 0]) * evaluation_mask)
            query_edge_loss = pos_query_edge_loss + neg_query_edge_loss

            full_edge_accr = self.hit(full_logit, 1 - full_edge[:, 0].long())
            query_edge_accr = torch.sum(full_edge_accr * query_edge_mask * evaluation_mask) / torch.sum(
                query_edge_mask * evaluation_mask)

            query_node_pred = torch.bmm(full_logit[:, 0, num_supports:, :num_supports],
                                        self.one_hot_encode(num_ways_test,
                                                            support_label.long()))  # (num_tasks x num_quries x num_supports) * (num_tasks x num_supports x num_ways)
            query_node_accr = torch.eq(torch.max(query_node_pred, -1)[1], query_label.long()).float().mean()


            query_edge_losses += [query_edge_loss.item()]
            query_edge_accrs += [query_edge_accr.item()]
            query_node_accrs += [query_node_accr.item()]


            print('evaluation: mean=%.2f%%, /n' % (np.array(query_node_accrs).mean() * 100,))


        return np.array(total_nodes_acc).mean()

    def label2edge(self, label):

        num_samples = label.size(1)

        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        label_j = label_i.transpose(1, 2)

        edge = torch.eq(label_i, label_j).float().cuda()

        edge = edge.unsqueeze(1)
        edge = torch.cat([edge, 1 - edge], 1)
        return edge

    def hit(self, logit, label):
        pred = logit.max(1)[1]
        hit = torch.eq(pred, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].cuda()

