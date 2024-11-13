import torch
import numpy as np
from tqdm import tqdm
import random
import math
import json
import os
VERY_SMALL_NUMBER = 1e-10
from sklearn.metrics import hamming_loss, accuracy_score

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)




def f1_and_hits_new(answers, candidate2prob, eps=0.5):
    retrieved = []
    correct = 0
    cand_list = sorted(candidate2prob, key=lambda x:x[1], reverse=True)
    if len(cand_list) == 0:
        best_ans = -1
        best_ans3 = [-1]
        best_ans5 = [-1]
    else:
        best_ans = cand_list[0][0]
        best_ans3 = cand_list[0][:3]
        best_ans5 = cand_list[0][:5]

    # max_prob = cand_list[0][1]
    tp_prob = 0.0
    correct_prob = 0.0
    case_study =[]
    for c, prob in cand_list:
        retrieved.append((c, prob))
        tp_prob += prob
        if c in answers:
            correct += 1
            correct_prob += prob
            case_study.append({"T":[c,prob]})
        else:
            case_study.append({"F":[c,prob]})
        if tp_prob > eps:
            break



    y_pred = [i[0] for i in retrieved]
    # 如果预测得到的答案比正确答案多，则按正确答案个数截取
    if len(answers) < len(y_pred):
        y_pred = y_pred[:len(answers)]
    else:
        for i in range(len(answers)-len(y_pred)):
            y_pred.append(-1)
    # hamming loss 预测错误的平均比率
    hammingLoss = hamming_loss(answers, y_pred)
    # subset accuracy 子集准确率
    subsetAccuracy = accuracy_score(answers,y_pred,normalize=False)

    mrr = MRR(answers,y_pred)

    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0, 1.0, 1.0,1.0,1.0, 0, retrieved, correct_prob, case_study, 0, 0,0  # precision, recall, f1, hits, hammingLoss, subsetAccuracy, mrr
        else:
            return 0.0, 1.0, 0.0, 1.0,1.0,1.0, 1, retrieved, correct_prob, case_study ,0, 0,0 # precision, recall, f1, hits, hammingLoss, subsetAccuracy,mrr
    else:
        # hits1 answers只要大于等于1就可以计算
        hits = float(best_ans in answers)
        hits3 = 0.0
        hits5 = 0.0
        if len(answers) >=3:
            for ans in best_ans3:
                hits3 += float(ans in best_ans3)
            if len(answers)>=5:
                for ans in best_ans5:
                    hits5 += float(ans in best_ans5)

        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits,hits3,hits5, 2, retrieved , correct_prob, case_study, hammingLoss,subsetAccuracy,mrr  # precision, recall, f1, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits,hits3,hits5, 3, retrieved, correct_prob, case_study, hammingLoss, subsetAccuracy,mrr


def MRR(truth, pred):
    rr_N = 0.0
    for index in range(len(truth)):
        p = pred[index]
        if p not in truth:
            rr_N += 0.0
        else:
            i = index
            rr_N += 1 / (i + 1)  # index从0开始
    return rr_N

class Evaluator_nsm:

    def __init__(self, args, student, entity2id, relation2id, device):
        self.student = student
        self.args = args
        self.eps = args['eps']
        self.num_step = args['num_step']
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        id2entity = {idx: entity for entity, idx in entity2id.items()}
        self.id2entity = id2entity
        id2relation = {idx: relation for relation, idx in relation2id.items()}
        num_rel_ori = len(relation2id)
        if self.use_inverse_relation:
            for i in range(len(id2relation)):
                id2relation[i + num_rel_ori] = id2relation[i] + "_rev"
        if self.use_self_loop:
            id2relation[len(id2relation)] = "self_loop"
        self.id2relation = id2relation
        self.file_write = None
        self.device = device

    def write_info(self, valid_data, tp_list):
        question_list = valid_data.get_quest()
        num_step = self.num_step
        obj_list = []
        if tp_list is not None:
            # attn_list = [tp[1] for tp in tp_list]
            action_list = [tp[0] for tp in tp_list]
        for i in range(len(question_list)):
            obj_list.append({})
        for j in range(num_step):
            if tp_list is None:
                actions = None
            else:
                actions = action_list[j]
                actions = actions.cpu().numpy()
            # if attn_list is not None:
            #     attention = attn_list[j].cpu().numpy()
            for i in range(len(question_list)):
                tp_obj = obj_list[i]
                q = question_list[i]
                # real_index = self.true_batch_id[i][0]
                tp_obj['question'] = q
                # 就在这一步添加过程中的实体
                tp_obj[j] = {}
                # print(actions)
                if tp_list is not None:
                    action = actions[i]
                    rel_action = self.id2relation[action]
                    tp_obj[j]['rel_action'] = rel_action
                    tp_obj[j]['action'] = str(action)
                    # if attn_list is not None:
                    #     attention_tp = attention[i]
                    #     tp_obj[j]['attention'] = attention_tp.tolist()
        return obj_list

    def evaluate(self, valid_data, test_batch_size=20, write_info=False):
        self.student.eval()
        self.count = 0
        eps = self.eps
        id2entity = self.id2entity
        eval_loss, eval_acc, eval_max_acc = [], [], []
        f1s, hits,hits3,hits5, precisions, recalls = [], [], [], [],[],[]
        correct_probs = []
        case_studies = []
        hammingLosses = []
        subsetAccuracies = []
        mrrs = []
        valid_data.reset_batches(is_sequential=True)
        num_epoch = math.ceil(valid_data.num_data / test_batch_size)
        if write_info and self.file_write is None:
            filename = os.path.join(self.args['checkpoint_dir'],
                                    "{}_test.info".format(self.args['experiment_name']))
            self.file_write = open(filename, "w")
        case_ct = {}
        max_local_entity = valid_data.max_local_entity
        ignore_prob = (1 - eps) / max_local_entity
        for iteration in tqdm(range(num_epoch)):
            batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0, test=True)
            with torch.no_grad():
                loss, extras, pred_dist, tp_list = self.student(batch[:-1])
                pred = torch.max(pred_dist, dim=1)[1]
            local_entity, query_entities, kb_adj_mat, query_text, \
            seed_dist, true_batch_id, answer_dist, answer_list = batch
            # self.true_batch_id = true_batch_id
            if write_info:
                obj_list = self.write_info(valid_data, tp_list)
                # pred_sum = torch.sum(pred_dist, dim=1)
                # print(pred_sum)
            candidate_entities = torch.from_numpy(local_entity).type('torch.LongTensor')
            true_answers = torch.from_numpy(answer_dist).type('torch.FloatTensor')
            query_entities = torch.from_numpy(query_entities).type('torch.LongTensor')
            # acc, max_acc = cal_accuracy(pred, true_answers.cpu().numpy())
            eval_loss.append(loss.item())
            # eval_acc.append(acc)
            # eval_max_acc.append(max_acc)
            batch_size = pred_dist.size(0)
            batch_answers = answer_list
            batch_candidates = candidate_entities
            pad_ent_id = len(id2entity)
            for batch_id in range(batch_size):
                answers = batch_answers[batch_id]
                candidates = batch_candidates[batch_id, :].tolist()
                probs = pred_dist[batch_id, :].tolist()
                seed_entities = query_entities[batch_id, :].tolist()
                candidate2prob = []
                for c, p, s in zip(candidates, probs, seed_entities):
                    if s == 1.0:
                        # ignore seed entities
                        continue
                    if c == pad_ent_id:
                        continue
                    if p < ignore_prob:
                        continue
                    candidate2prob.append((c, p))
                precision, recall, f1, hit,hit3,hit5, case, retrived, correct_prob, case_study, hammingLoss, subsetAccuracy,mrr = f1_and_hits_new(answers, candidate2prob, eps)
                if write_info:
                    tp_obj = obj_list[batch_id]
                    tp_obj['precison'] = precision
                    tp_obj['recall'] = recall
                    tp_obj['f1'] = f1
                    tp_obj['hit'] = hit
                    tp_obj['cand'] = retrived
                    self.file_write.write(json.dumps(tp_obj) + "\n")
                case_ct.setdefault(case, 0)
                case_ct[case] += 1
                f1s.append(f1)
                hits.append(hit)
                hits3.append(hit3)
                hits5.append(hit5)
                precisions.append(precision)
                recalls.append(recall)
                correct_probs.append(correct_prob)
                case_studies.append(case_study)
                hammingLosses.append(hammingLoss)
                subsetAccuracies.append(subsetAccuracy)
                mrrs.append(mrr)
        print('evaluation.......')
        print('how many eval samples......', len(f1s))
        # print('avg_f1', np.mean(f1s))
        print('avg_hits', np.mean(hits))
        print('avg_hits3', np.mean(hits3))
        print('avg_hits5', np.mean(hits5))
        print('avg_precision', np.mean(precisions))
        print('avg_recall', np.mean(recalls))
        print('avg_f1', np.mean(f1s))
        print('avg_correct_probs', np.mean(correct_probs))
        print('avg_hammingLoss',np.mean(hammingLosses))
        print('avg_subsetAccuracy', np.mean(subsetAccuracies))
        print('avg_mrr', np.mean(mrrs))
        print(case_ct)

        for index,_ in enumerate(case_studies):
            question = valid_data.data[index]["question"]
            true_answer = valid_data.data[index]["answers"]
            pred_answer = [{str(i.keys())[12:13]:[valid_data.id2entity[list(i.values())[0][0]], list(i.values())[0][1]]} for i in _]
            with open("{}case_study.txt".format(self.args['checkpoint_dir']), "a") as file:
                file.writelines("question: {}\ntrue_answer: {}\npred_answer: {}\n\n".format(question,true_answer,pred_answer))

        if write_info:
            self.file_write.close()
            self.file_write = None
        return np.mean(f1s), np.mean(hits)
