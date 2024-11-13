import random
import sys

import torch
import numpy as np
import torch.nn.functional as F
from NSM.Model.base_model import BaseModel
from NSM.Modules.Instruction.seq_instruction import LSTMInstruction
from NSM.Modules.Reasoning.gnn_reasoning import GNNReasoning
from NSM.Model.cross_attention_model import MultiHeadCrossAttention

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class GNNModel(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(GNNModel, self).__init__(args, num_entity, num_relation, num_word)
        self.embedding_def()
        self.share_module_def()
        self.private_module_def(args, num_entity, num_relation)
        self.loss_type = "kl"
        self.model_name = args['model_name'].lower()
        self.lambda_label = args['lambda_label']
        self.filter_label = args['filter_label']
        self.word_emb=np.load(args['data_folder']+args['word_emb_file'])
        self.to(self.device)

    def private_module_def(self, args, num_entity, num_relation):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.reasoning = GNNReasoning(args, num_entity, num_relation)
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=self.rel_features)

    def one_step(self, num_step):
        # relational_ins, attn_weight = self.instruction.get_instruction(self.relational_ins, query_mask, step=num_step)
        relational_ins = self.instruction_list[num_step]
        # attn_weight = self.attn_list[num_step]
        # self.relational_ins = relational_ins
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)





    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss
    def calc_att_loss_weight_label(self, curr_dist, teacher_dist, label_valid,weight_dist):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        tp_loss=(tp_loss.to(self.device)*weight_dist.to(self.device)).to(self.device)
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def train_batch(self, batch, middle_dist, label_valid=None):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch

        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        # loss, extras = self.calc_loss_basic(answer_dist)
        pred_dist = self.dist_history[-1]
        # main_loss = self.get_loss_new(pred_dist, answer_dist)
        # tp_loss = self.get_loss_kl(pred_dist, answer_dist)
        # (batch_size, max_local_entity)
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_num=[int(i) for i in list(answer_number)]
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # main_loss = torch.sum(tp_loss * case_valid) / pred_dist.size(0)

        main_loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pre_loss = None

        #将答案实体假如到teacher的middle_dist
        middle_dist_extend=self.get_middle_dist_extend(middle_dist,answer_dist)
        middle_dist_cat = torch.cat(middle_dist_extend, dim=1)
        middle_dist_horizon = (middle_dist_cat.transpose(1, 2)).transpose(0, 1)
        current_dist_extend = [i.unsqueeze(1) for i in self.dist_history[1:]]

        current_dist_cat = torch.cat(current_dist_extend, dim=1)
        current_dist_horizon = (current_dist_cat.transpose(1, 2)).transpose(0, 1)

        middle_att_input_emb,current_att_input_emb= self.get_entities_seq(middle_dist_extend,self.dist_history[1:],answer_num)#小心Current_dist不对，这里需要的是概率分布
        middle_att_dist,current_att_dist=self.get_middle_att_dist(middle_att_input_emb,current_att_input_emb,answer_num,q_input)

        horizon_weight = 0.0003
        horizon_loss = None
        for mid_horizon, cur_horizon in zip(middle_dist_horizon, current_dist_horizon):
            if self.filter_label:
                assert not (label_valid is None)
                tp_horizon_label_loss = self.calc_loss_label(curr_dist=cur_horizon,
                                                             teacher_dist=mid_horizon,
                                                             label_valid=label_valid)
            else:
                tp_horizon_label_loss = self.calc_loss_label(curr_dist=cur_horizon,
                                                             teacher_dist=mid_horizon,
                                                             label_valid=case_valid)
            # print(tp_horizon_label_loss)
            if horizon_loss is None:
                horizon_loss = tp_horizon_label_loss*horizon_weight
            else:
                horizon_loss += tp_horizon_label_loss*horizon_weight
        # horizon_loss_weight = 0.002
        # distill_loss = horizon_loss_weight * distill_loss

        # 推理过程中学习teacher的loss
        # att_loss_weight = 0.5
        crossA_weight = 0.6
        crossA_loss = None
        for i in range(self.num_step - 1):
            curr_dist = self.dist_history[i + 1]
            curr_att_dist = current_att_dist[i]
            # teacher_dist = middle_dist[i].detach()
            teacher_dist = middle_dist[i].squeeze(1).detach()
            teacher_att_dist = middle_att_dist[i].squeeze(1).detach()
            if self.filter_label:
                assert not (label_valid is None)
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist,
                                                     teacher_dist=teacher_dist,
                                                     label_valid=label_valid)
                tp_label_att_loss = self.calc_loss_label(curr_dist=curr_att_dist,
                                                         teacher_dist=teacher_att_dist,
                                                         label_valid=label_valid)
            else:
                # tp_label_loss = self.get_loss_new(curr_dist, teacher_dist)
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist,
                                                     teacher_dist=teacher_dist,
                                                     label_valid=case_valid)
                tp_label_att_loss = self.calc_loss_label(curr_dist=curr_att_dist,
                                                         teacher_dist=teacher_att_dist,
                                                         label_valid=case_valid)

            # print(tp_label_loss,tp_label_att_loss*att_loss_weight)
            if pre_loss is None:
                crossA_loss = tp_label_att_loss*crossA_weight
                pre_loss = tp_label_loss
            else:
                crossA_loss += tp_label_att_loss*crossA_weight
                pre_loss += tp_label_loss
        # attention最后一步的loss


        for i in range(len(middle_att_dist[self.num_step-1:])):
            curr_att_dist=(current_att_dist[self.num_step-1:])[i]
            teacher_att_dist=(middle_att_dist[self.num_step-1:])[i]

            # answer_num_int = [int(i) for i in answer_num]
            rate = torch.Tensor([[i / 2000] for i in answer_num])

            if self.filter_label:
                assert not (label_valid is None)
                tp_label_att_loss = self.calc_att_loss_weight_label(curr_dist=curr_att_dist,
                                                         teacher_dist=teacher_att_dist,
                                                         label_valid=label_valid,
                                                         weight_dist=rate)
            else:
                tp_label_att_loss = self.calc_att_loss_weight_label(curr_dist=curr_att_dist,
                                                         teacher_dist=teacher_att_dist,
                                                         label_valid=case_valid,
                                                         weight_dist=rate)

            if crossA_loss is None:
                crossA_loss = tp_label_att_loss*crossA_weight
            else:
                crossA_loss += tp_label_att_loss*crossA_weight
            # print(distill_loss)
        # pred = torch.max(pred_dist, dim=1)[1]

        distill_loss = self.fine_tune_weight_simple(pre_loss,horizon_loss,crossA_loss)


        extras = [main_loss.item(), distill_loss.item()]
        # tp_list = [h1.tolist(), f1.tolist()]
        loss = main_loss + distill_loss * self.lambda_label
        h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
        tp_list = [h1.tolist(), f1.tolist()]
        return loss, extras, pred_dist, tp_list

    def fine_tune_weight(self,pre_loss,horizon_loss,crossA_loss):
        random_rate = random.uniform(1, 2)
        ad_crossA_loss = random_rate*crossA_loss
        add_weight = float(ad_crossA_loss/(horizon_loss+ad_crossA_loss))
        horizon_loss_new = add_weight*horizon_loss
        crossA_loss_new = add_weight*ad_crossA_loss
        distill_loss = pre_loss + horizon_loss_new + crossA_loss_new
        return distill_loss

    def fine_tune_weight_simple(self,pre_loss,horizon_loss,crossA_loss):
        fine_tune_weight = 0.9
        return pre_loss + fine_tune_weight*horizon_loss +fine_tune_weight*crossA_loss

    def auto_weight(self,pre_loss,horizon_loss,crossA_loss):
        horizon_weight = pre_loss/(2*horizon_loss)
        crossA_weight = pre_loss/(2*crossA_loss)

        return pre_loss+float(horizon_weight)*horizon_loss+float(crossA_weight)*crossA_loss

    def get_middle_dist_extend(self,middle_dist,answer_dist):
        middle_dist_extend=list(middle_dist)
        middle_dist_extend.append(answer_dist.unsqueeze(1))
        return tuple(middle_dist_extend)


    def get_middle_att_dist(self,middle_att_input_emb,current_att_input_emb,answer_num,q_input):

        max_answer=max(answer_num)
        q_len=len(q_input[0])
        check=max_answer+(self.num_step-1)-q_len
        zero_50tensor = torch.zeros(50)
        # 能不能先pending，然后torch，最后transpose，这样效率应该会高
        middle_att_emb_without_last_step = torch.stack([torch.stack(_) for _ in middle_att_input_emb[:-1]]).transpose(0,1)
        current_att_emb_without_last_step = torch.stack([torch.stack(_) for _ in current_att_input_emb[:-1]]).transpose(0,1)

        for i in range(len(middle_att_input_emb[0])):
            j = self.num_step-1

            if check >= 0:
                middle_att_input_emb[j][i].extend([zero_50tensor] * (max_answer - answer_num[i]))
                current_att_input_emb[j][i].extend([zero_50tensor] * (max_answer - answer_num[i]))
            else:
                extra_range = q_len - answer_num[i] - (self.num_step - 1)
                middle_att_input_emb[j][i].extend([zero_50tensor] * extra_range)
                current_att_input_emb[j][i].extend([zero_50tensor] * extra_range)

        middle_att_emb_last_step=torch.stack([torch.stack(_) for _ in middle_att_input_emb[-1]])
        current_att_emb_last_step=torch.stack([torch.stack(_) for _ in current_att_input_emb[-1]])

        self.middle_att_emb_all_step = torch.cat([middle_att_emb_without_last_step,middle_att_emb_last_step],dim=1)
        self.current_att_emb_all_step = torch.cat([current_att_emb_without_last_step,current_att_emb_last_step],dim=1)

        question_att_input_emb=self.get_question_att_input_emb(q_input,check)

        mid_attention=MultiHeadCrossAttention(question_att_input_emb,self.middle_att_emb_all_step)
        cur_attention=MultiHeadCrossAttention(question_att_input_emb,self.current_att_emb_all_step)

        mid_att_output = mid_attention()
        cur_att_output = cur_attention()

        middle_att_output_change=mid_att_output.transpose(0,1)
        current_att_output_change = cur_att_output.transpose(0,1)
        # 直接返回张量tensor 23*20*128
        return middle_att_output_change,current_att_output_change

    def get_question_att_input_emb(self,q_input,check):
        if check>=0:
            padding_size=(0,check,0,0)
            q_padded = torch.nn.functional.pad(q_input, padding_size, mode='constant', value=6718)
        else:
            q_padded = q_input

        # 将vocab_id转换为word_emb，并变为tensor返回
        mask = (q_padded == 6718)
        q_padded_index = torch.where(mask,q_padded - 1, q_padded)

        q_att_input_emb = self.word_emb[q_padded_index]

        return (torch.from_numpy(q_att_input_emb)).float()

    def get_entities_seq(self,middle_dist_extend,curr_dist_extend,answer_num):

        # 获取除最后一步（middle_dist_extend[:-1]，curr_dist_extend[:-1]）每一步概率最大的实体下标
        mid_entites = []
        cur_entites = []

        for step in middle_dist_extend[:-1]:#step:(20,2000) 一批20个样本
            mid_entites_index = []

            for one in step:
                # 求最大概率的实体下表
                mid_max_pre,mid_entity_index = torch.max(one, dim=1) #torch.max获取张量中最大值以及索引
                mid_entites_index.append(mid_entity_index.tolist())
            mid_entites.append(mid_entites_index)

        for step in curr_dist_extend[:-1]:  # step:(20,2000) 一批20个样本
            cur_entites_index = []

            for one in step:
                # 求最大概率的实体下表
                cur_max_pre,cur_entity_index = torch.max(one,dim=0)
                cur_entites_index.append(cur_entity_index.tolist())
            cur_entites.append(cur_entites_index)

        # 获取（middle_dist_extend[-1]）最后一步答案实体有答案的下标，扩充mid_entities
        # answer_num=[]
        mid_last_entites_index=[]
        for one in middle_dist_extend[-1]:
            one=one.squeeze(dim=0)
            mid_last_entites_index_one = torch.nonzero(one == 1).tolist()
            mid_last_entites_index.append(mid_last_entites_index_one)
        mid_entites.append(mid_last_entites_index)

        # 获取（curr_dist_extend[-1]）根据概率大小排序的前answer_num个实体下标，扩充cur_entities
        cur_last_entites_index=[]
        for ct,one in enumerate(curr_dist_extend[-1]):
            indexed_list=[[index,float(i)] for index,i in enumerate(one)]
            curr_last_dist_sorted=sorted(indexed_list,key=lambda x:x[1],reverse=True)
            curr_last_dist_get=curr_last_dist_sorted[:answer_num[ct]]

            get_dist_index = [i[0] for i in curr_last_dist_get]

            cur_last_entites_index.append(get_dist_index)
        cur_entites.append(cur_last_entites_index)


        # 获取除最后一步每一步（mid_entities[:-1]，cur_entities[:-1]）实体的embedding
        mid_att_input = []
        cur_att_input = []
        for step in mid_entites[:-1]:
            step_emb = [self.local_entity_emb[index][one_index[0]] for index, one_index in enumerate(step)]
            mid_att_input.append(step_emb)

        for step in cur_entites[:-1]:
            step_emb = [self.local_entity_emb[index][one_index] for index, one_index in enumerate(step)]
            cur_att_input.append(step_emb)

        # 获取最后一步(mid_entites[-1],cur_entites[-1])实体的embedding
        last_mid_step_emb=[]
        for index,one in enumerate(mid_entites[-1]):
            last_mid_step_emb_one = []
            if one != []:
                last_mid_step_emb_one=[self.local_entity_emb[index][one_index[0]] for one_index in one]
            else:
                pass
            last_mid_step_emb.append(last_mid_step_emb_one)
        mid_att_input.append(last_mid_step_emb)

        last_cur_step_emb=[]
        for index,one in enumerate(cur_entites[-1]):
            last_cur_step_emb_one = []
            if one != []:
                    last_cur_step_emb_one=[self.local_entity_emb[index][one_index] for one_index in one]
            else:
                pass
            last_cur_step_emb.append(last_cur_step_emb_one)
        cur_att_input.append(last_cur_step_emb)


        return mid_att_input,cur_att_input

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        pred_dist = self.dist_history[-1]
        # loss, extras = self.calc_loss_basic(answer_dist)
        # tp_loss = self.get_loss_kl(pred_dist, answer_dist)
        # (batch_size, max_local_entity)
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = torch.sum(tp_loss * case_valid) / pred_dist.size(0)
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list