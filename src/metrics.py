from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix, classification_report
import torch
from snownlp import SnowNLP

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):  # -------
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den

def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))

def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    #print("ceafe-----------01")
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = np.stack(linear_assignment(-scores), axis=1)  # --------- 为啥这样就是对的
    # matching = linear_assignment(-scores)
    # print("ceafe-----------02")
    # matching = np.asarray(matching)
    # print("matching's shape: ", matching.shape)
    # print("matching[:, 0]: ", matching[:, 0])
    # print("matching[:, 1]: ", matching[:, 1])
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    #print("ceafe-ok------------")
    return similarity, len(clusters), similarity, len(gold_clusters)

def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

def get_metaphor_mention_prf(prediction_clusters, gold_clusters):
    """计算隐喻链中mention识别的结果"""
    # 只计算隐喻句的mention识别结果
    meta_num = 0
    right_mention_num = 0
    gold_mention_num, pre_mention_num = 0, 0
    for example_num, clusters in prediction_clusters.items():
        gold_mention = [tuple(span) for clu in gold_clusters[example_num] for span in clu]
        # gold_mention = set(gold_mention)
        # gold_mention_num += len(gold_mention)
        gold_mention_num += len(set(gold_mention))
        mem_pre_mention = []  # 避免重复记录pre mention
        pre_mention = [span for clu in clusters for span in clu]
        # pre_mention = set(pre_mention)
        right_span = 0
        for span in pre_mention:
            if span not in mem_pre_mention:
                if span in gold_mention:
                    right_span += 1
                mem_pre_mention.append(span)
        right_mention_num += right_span
        # pre_mention_num += len(pre_mention)      #  --------要计算非隐喻句的结果
        pre_mention_num += len(mem_pre_mention)
        if len(gold_mention) == 0:  # 此句为非隐喻句，跳过
            continue
        else:
            meta_num += 1

    print("right mention num:", right_mention_num, "   all pre mention num:", pre_mention_num,
          "   gold_mention_num:", gold_mention_num)
    if pre_mention_num == 0:
        precision = 0
    else:
        precision = right_mention_num / pre_mention_num
    if gold_mention_num == 0:
        recall = 0
    else:
        recall = right_mention_num / gold_mention_num
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    # print("p_1, r_1, f_1:", precision, recall, f1)

    return precision, recall, f1


def get_meta_fuzzy_prf(prediction_clusters, gold_clusters, fuzzy_raito):
    """
    用模糊匹配来计算隐喻链中mention识别的结果
    :param prediction_clusters:
    :param gold_clusters:
    :param fuzzy_raito: 模糊匹配概率   当前span与gold span交集长度/ gold span length
    :return:
    """
    precision, recall, f1 = 0, 0, 0
    meta_num = 0
    all_pre_mention_num, all_gold_mention_num, all_right_span_num = 0, 0, 0
    for example_num, clusters in prediction_clusters.items():
        gold_mention = [tuple(span) for clu in gold_clusters[example_num] for span in clu]
        gold_mention_set = set(gold_mention)
        pre_mention = [span for clu in clusters for span in clu]
        pre_mention_set = set(pre_mention)
        all_pre_mention_num += len(pre_mention_set)       #  --------要计算非隐喻句的结果
        if len(gold_mention) == 0:
            continue
        all_gold_mention_num += len(gold_mention_set)
        # print("gold_mention_set:", gold_mention_set, "  pre_mention_set:", pre_mention_set)
        right_span = 0
        for span in pre_mention_set:
            start = span[0]
            end = span[1]
            span_length = end - start + 1
            for gold_span in gold_mention:
                gold_length = gold_span[1] - gold_span[0] + 1
                if start > gold_span[1] or end < gold_span[0]:
                    continue
                elif start <= gold_span[0] and end >= gold_span[1]:   # span包括了gold span
                    same_ratio_r = span_length / gold_length
                    same_ratio_p = span_length / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_raito:
                        right_span += 1
                        break
                elif start <= gold_span[0] and end < gold_span[1]:   # span_start 与 gold span在前有交集
                    # if gold_length == 1:
                    #     print("span:", span)
                    #     print("gold span:", gold_span)
                    #     right_span += 1
                    #     break
                    same_ratio_r = (end - gold_span[0] + 1) / gold_length
                    same_ratio_p = (end - gold_span[0] + 1) / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_raito:
                        right_span += 1
                        break
                elif gold_span[0] < start <= gold_span[1]:
                    if end <= gold_span[1]:    # gold span 包含了 span
                        same_ratio_r = span_length / gold_length
                        same_ratio_p = span_length / span_length
                        same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                        if same_ratio_f >= fuzzy_raito:
                            right_span += 1
                            break
                    else:     # end > gold span end   向后有交集
                        same_ratio_r = (gold_span[1] - start + 1) / gold_length
                        same_ratio_p = (gold_span[1] - start + 1) / span_length
                        same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                        if same_ratio_f >= fuzzy_raito:
                            right_span += 1
                            break
        if len(pre_mention) < right_span:
            print("pre mention num:", len(pre_mention), " right span:", right_span)

        all_right_span_num += right_span
        # print("i:", example_num, " right span:", right_span)

    # print("fuzzy p1, r1, f1:", precision, " ", recall, " ", f1)

    if all_pre_mention_num == 0:
        precision = 0
    else:
        precision = all_right_span_num / all_pre_mention_num

    if all_right_span_num == 0:
        recall = 0
    else:
        if all_right_span_num > all_gold_mention_num:
            recall = 1
        else:
            recall = all_right_span_num / all_gold_mention_num
    # print("fuzzy right num:", all_right_span_num)
    print("\nall_right_span_num:", all_right_span_num, "  all_pre_mention_num:", all_pre_mention_num,
          "  all_gold_mention_num:", all_gold_mention_num)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    # print("fuzzy p2, r2, f2:", precision, " ", recall, " ", f1)
    return precision, recall, f1

def get_meta_fuzzy_prf_type(prediction_clusters, gold_clusters, fuzzy_raito, doc_gold_spans_type, meta_span_types, doc_top_spans):
    """
    用模糊匹配来计算隐喻链中mention识别的结果 加上type(one span to classify)
    :param prediction_clusters:
    :param gold_clusters:
    :param fuzzy_raito: 模糊匹配概率   当前span与gold span交集长度/ gold span length
    :return:
    """
    precision, recall, f1 = 0, 0, 0
    meta_num = 0
    all_pre_mention_num, all_gold_mention_num, all_right_span_num = 0, 0, 0
    for example_num, clusters in prediction_clusters.items():
        gold_mention = [tuple(span) for clu in gold_clusters[example_num] for span in clu]
        gold_mention_set = set(gold_mention)
        pre_mention = [span for clu in clusters for span in clu]
        pre_mention_set = set(pre_mention)
        all_pre_mention_num += len(pre_mention_set)       #  --------要计算非隐喻句的结果
        if len(gold_mention) == 0:
            continue
        all_gold_mention_num += len(gold_mention_set)
        # print("gold_mention_set:", gold_mention_set, "  pre_mention_set:", pre_mention_set)

        gold_spans_type = doc_gold_spans_type[example_num]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = 0 if g_span_type[1] == 0 else 1

        top_spans_type = meta_span_types[example_num]
        top_spans = doc_top_spans[example_num]
        top_span_type_dict = {}
        for j, top_span in enumerate(top_spans):
            top_span_type_dict[top_span] = 0 if int(top_spans_type[j]) == 0 else 1

        right_span = 0
        for span in pre_mention_set:
            start = span[0]
            end = span[1]
            span_length = end - start + 1
            span_p_type = top_span_type_dict[span]
            for gold_span in gold_mention:
                gold_length = gold_span[1] - gold_span[0] + 1
                if start > gold_span[1] or end < gold_span[0]:
                    continue
                elif start <= gold_span[0] and end >= gold_span[1]:   # span包括了gold span
                    same_ratio_r = span_length / gold_length
                    same_ratio_p = span_length / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_raito:
                        span_g_type = gold_span_2_type[gold_span]
                        if span_p_type == span_g_type:
                            right_span += 1
                        break
                elif start <= gold_span[0] and end < gold_span[1]:   # span_start 与 gold span在前有交集
                    # if gold_length == 1:
                    #     print("span:", span)
                    #     print("gold span:", gold_span)
                    #     right_span += 1
                    #     break
                    same_ratio_r = (end - gold_span[0] + 1) / gold_length
                    same_ratio_p = (end - gold_span[0] + 1) / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_raito:
                        span_g_type = gold_span_2_type[gold_span]
                        if span_p_type == span_g_type:
                            right_span += 1
                        break
                elif gold_span[0] < start <= gold_span[1]:
                    if end <= gold_span[1]:    # gold span 包含了 span
                        same_ratio_r = span_length / gold_length
                        same_ratio_p = span_length / span_length
                        same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                        if same_ratio_f >= fuzzy_raito:
                            span_g_type = gold_span_2_type[gold_span]
                            if span_p_type == span_g_type:
                                right_span += 1
                            break
                    else:     # end > gold span end   向后有交集
                        same_ratio_r = (gold_span[1] - start + 1) / gold_length
                        same_ratio_p = (gold_span[1] - start + 1) / span_length
                        same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                        if same_ratio_f >= fuzzy_raito:
                            span_g_type = gold_span_2_type[gold_span]
                            if span_p_type == span_g_type:
                                right_span += 1
                            break
        if len(pre_mention) < right_span:
            print("pre mention num:", len(pre_mention), " right span:", right_span)

        all_right_span_num += right_span
        # print("i:", example_num, " right span:", right_span)

    # print("fuzzy p1, r1, f1:", precision, " ", recall, " ", f1)

    if all_pre_mention_num == 0:
        precision = 0
    else:
        precision = all_right_span_num / all_pre_mention_num

    if all_right_span_num == 0:
        recall = 0
    else:
        if all_right_span_num > all_gold_mention_num:
            recall = 1
        else:
            recall = all_right_span_num / all_gold_mention_num
    # print("fuzzy right num:", all_right_span_num)
    print("\nall_right_span_num:", all_right_span_num, "  all_pre_mention_num:", all_pre_mention_num,
          "  all_gold_mention_num:", all_gold_mention_num)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    # print("fuzzy p2, r2, f2:", precision, " ", recall, " ", f1)
    return precision, recall, f1


def get_metaphor_mention_prf_type(prediction_clusters, gold_clusters, doc_gold_spans_type, doc_predict_span_2_type):
    """
    span pair type识别的结果
    计算所有隐喻链中mention识别的结果
    同时看span边界和span类型，类型只分为两类：本体和喻体
    """
    meta_num = 0
    right_mention_num = 0
    gold_mention_num, pre_mention_num = 0, 0
    for example_num, clusters in prediction_clusters.items():
        gold_mention = [tuple(span) for clu in gold_clusters[example_num] for span in clu]
        # gold_mention = set(gold_mention)
        # gold_mention_num += len(gold_mention)
        gold_mention_num += len(set(gold_mention))
        mem_pre_mention = []  # 避免重复记录pre mention
        pre_mention = [span for clu in clusters for span in clu]
        # pre_mention = set(pre_mention)

        gold_spans_type = doc_gold_spans_type[example_num]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = 0 if g_span_type[1] == 0 else 1

        predict_span_2_type = doc_predict_span_2_type[example_num]

        right_span = 0
        for span in pre_mention:
            if span not in mem_pre_mention:
                span_p_type = 0 if predict_span_2_type[span]== 0 else 1
                if span in gold_mention:
                    span_g_type = gold_span_2_type[span]
                    if span_p_type == span_g_type:
                        right_mention_num += 1
                mem_pre_mention.append(span)
        
        # right_mention_num += right_span
        # pre_mention_num += len(pre_mention)      #  --------要计算非隐喻句的结果
        pre_mention_num += len(mem_pre_mention)
        if len(gold_mention) == 0:  # 此句为非隐喻句，跳过
            continue
        else:
            meta_num += 1

    print("\nright mention num:", right_mention_num, "  all pre mention num:", pre_mention_num,
          "   gold_mention_num:", gold_mention_num)
    if pre_mention_num == 0:
        precision = 0
    else:
        precision = right_mention_num / pre_mention_num
    if gold_mention_num == 0:
        recall = 0
    else:
        recall = right_mention_num / gold_mention_num
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    # print("p_1, r_1, f_1:", precision, recall, f1)

    return precision, recall, f1

def get_metaphor_mention_prf_type_fuzzy(prediction_clusters, gold_clusters, fuzzy_raito, doc_gold_spans_type, doc_predict_span_2_type):
    """
    span pair type识别的结果 
    计算所有隐喻链中mention识别的结果 fuzzy
    同时看span边界和span类型，类型只分为两类：本体和喻体
    """
    meta_num = 0
    right_mention_num = 0
    gold_mention_num, pre_mention_num = 0, 0
    for example_num, clusters in prediction_clusters.items():
        gold_mention = [tuple(span) for clu in gold_clusters[example_num] for span in clu]
        gold_mention_num += len(set(gold_mention))
        pre_mention = [span for clu in clusters for span in clu]
        pre_mention_set = set(pre_mention)

        gold_spans_type = doc_gold_spans_type[example_num]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = 0 if g_span_type[1]==0 else 1

        predict_span_2_type = doc_predict_span_2_type[example_num]

        right_span = 0
        for span in pre_mention_set:
            start = span[0]
            end = span[1]
            span_length = end - start + 1
            span_p_type = predict_span_2_type[span]
            if span_p_type == 0:  # 本体
                span_p_type_change = 0
            else:  # 喻体
                span_p_type_change = 1

            for gold_span in gold_mention:
                gold_length = gold_span[1] - gold_span[0] + 1
                gold_span_type = gold_span_2_type[gold_span]
                if start > gold_span[1] or end < gold_span[0]:
                    continue
                elif start <= gold_span[0] and end >= gold_span[1]:  # span包括了gold span
                    same_ratio_r = span_length / gold_length
                    same_ratio_p = span_length / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_raito:
                        if span_p_type_change == gold_span_type:
                            right_span += 1
                        break
                elif start <= gold_span[0] and end < gold_span[1]:  # span_start 与 gold span在前有交集

                    same_ratio_r = (end - gold_span[0] + 1) / gold_length
                    same_ratio_p = (end - gold_span[0] + 1) / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_raito:
                        if span_p_type_change == gold_span_type:
                            right_span += 1
                        break
                elif gold_span[0] < start <= gold_span[1]:
                    if end <= gold_span[1]:  # gold span 包含了 span
                        same_ratio_r = span_length / gold_length
                        same_ratio_p = span_length / span_length
                        same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                        if same_ratio_f >= fuzzy_raito:
                            if span_p_type_change == gold_span_type:
                                right_span += 1
                            break
                    else:  # end > gold span end   向后有交集
                        same_ratio_r = (gold_span[1] - start + 1) / gold_length
                        same_ratio_p = (gold_span[1] - start + 1) / span_length
                        same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                        if same_ratio_f >= fuzzy_raito:
                            if span_p_type_change == gold_span_type:
                                right_span += 1
                            break


        right_mention_num += right_span
        pre_mention_num += len(pre_mention_set)      #  --------要计算非隐喻句的结果
        # pre_mention_num += len(mem_pre_mention)
        if len(gold_mention) == 0:  # 此句为非隐喻句，跳过
            continue
        else:
            meta_num += 1

    print("\nright mention num:", right_mention_num, "   all pre mention num:", pre_mention_num,
          "   gold_mention_num:", gold_mention_num)
    if pre_mention_num == 0:
        precision = 0
    else:
        precision = right_mention_num / pre_mention_num
    if gold_mention_num == 0:
        recall = 0
    else:
        recall = right_mention_num / gold_mention_num
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    # print("p_1, r_1, f_1:", precision, recall, f1)

    return precision, recall, f1



def get_metaphor_mention_prf_one_type(prediction_clusters, gold_clusters, doc_gold_spans_type, meta_span_types, doc_top_spans):
    """
    one span classify
    计算隐喻链中mention识别的结果
    """

    meta_num = 0
    right_mention_num = 0
    gold_mention_num, pre_mention_num = 0, 0
    source_g_span_num, source_p_span_num, source_right_span_num = 0, 0, 0   #  喻体
    target_g_span_num, target_p_span_num, target_right_span_num = 0, 0, 0    # 本体
    type_span_predict, type_span_gold = [], []
    for example_num, clusters in prediction_clusters.items():
        gold_mention = [tuple(span) for clu in gold_clusters[example_num] for span in clu]
        # gold_mention = set(gold_mention)
        # gold_mention_num += len(gold_mention)
        gold_mention_num += len(set(gold_mention))
        mem_pre_mention = []  # 避免重复记录pre mention
        pre_mention = [span for clu in clusters for span in clu]
        # pre_mention = set(pre_mention)

        gold_spans_type = doc_gold_spans_type[example_num]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = 0 if g_span_type[1] == 0 else 1
                if g_span_type[1] == 0:
                    target_g_span_num += 1
                else:
                    source_g_span_num += 1

        top_spans_type = meta_span_types[example_num]
        top_spans = doc_top_spans[example_num]
        top_span_type_dict = {}
        for j, top_span in enumerate(top_spans):
            top_span_type_dict[top_span] = 0 if int(top_spans_type[j]) == 0 else 1

        right_span = 0
        for span in pre_mention:
            if span not in mem_pre_mention:
                
                if span in gold_mention:
                    if gold_span_2_type[span] == top_span_type_dict[span]:
                        right_span += 1
                        if gold_span_2_type[span] == 0:
                            target_right_span_num += 1
                        else:
                            source_right_span_num += 1
                    type_span_gold.append(gold_span_2_type[span])
                else:
                    type_span_gold.append(2)
                    
                if top_span_type_dict[span] == 0:
                    target_p_span_num += 1
                else:
                    source_p_span_num += 1
                type_span_predict.append(top_span_type_dict[span])
                
                mem_pre_mention.append(span)
        right_mention_num += right_span
        # pre_mention_num += len(pre_mention)      #  --------要计算非隐喻句的结果
        pre_mention_num += len(mem_pre_mention)
        if len(gold_mention) == 0:  # 此句为非隐喻句，跳过
            continue
        else:
            meta_num += 1

    print("\nright mention num:", right_mention_num, "   all pre mention num:", pre_mention_num,
          "   gold_mention_num:", gold_mention_num)
    if pre_mention_num == 0:
        precision = 0
    else:
        precision = right_mention_num / pre_mention_num
    if gold_mention_num == 0:
        recall = 0
    else:
        recall = right_mention_num / gold_mention_num
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    # print("p_1, r_1, f_1:", precision, recall, f1)
    
    target_precision = 0 if target_p_span_num == 0 else target_right_span_num / target_p_span_num
    target_recall = 0 if target_g_span_num == 0 else target_right_span_num / target_g_span_num
    target_f1 = 0 if target_precision + target_recall == 0 else ( 2 * target_precision * target_recall) / (target_precision + target_recall)
    
    # print("\ntarget_p_span_num:", target_p_span_num, " target_g_span_num:", target_g_span_num, " target_right_span_num:", target_right_span_num)
    # print("target p:", target_precision, " r:", target_recall, " f1:", target_f1)
    
    source_precision = 0 if source_p_span_num == 0 else source_right_span_num / source_p_span_num
    source_recall = 0 if source_g_span_num == 0 else source_right_span_num / source_g_span_num
    source_f1 = 0 if source_precision + source_recall == 0 else ( 2 * source_precision * source_recall) / (source_precision + source_recall)
    
    # print("\nsource_p_span_num:", source_p_span_num, " source_g_span_num:", source_g_span_num, " source_right_span_num:", source_right_span_num)
    # print("source p:", source_precision, " r:", source_recall, " f1:", source_f1)
    
    # print("\nspan type 分类结果：\n", classification_report(type_span_gold, type_span_predict, digits=4))

    return precision, recall, f1

def get_metaphor_mention_prf_one_type_2(prediction_clusters, gold_clusters, doc_gold_spans_type, meta_span_types):
    """
    one span classify
    计算隐喻链中mention识别的结果
    """

    meta_num = 0
    right_mention_num = 0
    gold_mention_num, pre_mention_num = 0, 0
    source_g_span_num, source_p_span_num, source_right_span_num = 0, 0, 0  # 喻体
    target_g_span_num, target_p_span_num, target_right_span_num = 0, 0, 0  # 本体
    type_span_predict, type_span_gold = [], []
    for example_num, clusters in prediction_clusters.items():
        gold_mention = [tuple(span) for clu in gold_clusters[example_num] for span in clu]
        # gold_mention = set(gold_mention)
        # gold_mention_num += len(gold_mention)
        gold_mention_num += len(set(gold_mention))
        mem_pre_mention = []  # 避免重复记录pre mention
        pre_mention = [span for clu in clusters for span in clu]
        # pre_mention = set(pre_mention)

        gold_spans_type = doc_gold_spans_type[example_num]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = 0 if g_span_type[1] == 0 else 1
                if g_span_type[1] == 0:
                    target_g_span_num += 1
                else:
                    source_g_span_num += 1

        predict_span_2_type = meta_span_types[example_num]

        right_span = 0
        for span in pre_mention:
            if span not in mem_pre_mention:

                if span in gold_mention:
                    p_type = 0 if predict_span_2_type[span] == 0 else 1
                    if gold_span_2_type[span] == p_type:
                        right_span += 1
                        if gold_span_2_type[span] == 0:
                            target_right_span_num += 1
                        else:
                            source_right_span_num += 1
                    type_span_gold.append(gold_span_2_type[span])
                else:
                    type_span_gold.append(2)

                if top_span_type_dict[span] == 0:
                    target_p_span_num += 1
                else:
                    source_p_span_num += 1
                type_span_predict.append(top_span_type_dict[span])

                mem_pre_mention.append(span)
        right_mention_num += right_span
        # pre_mention_num += len(pre_mention)      #  --------要计算非隐喻句的结果
        pre_mention_num += len(mem_pre_mention)
        if len(gold_mention) == 0:  # 此句为非隐喻句，跳过
            continue
        else:
            meta_num += 1

    print("\nright mention num:", right_mention_num, "   all pre mention num:", pre_mention_num,
          "   gold_mention_num:", gold_mention_num)
    if pre_mention_num == 0:
        precision = 0
    else:
        precision = right_mention_num / pre_mention_num
    if gold_mention_num == 0:
        recall = 0
    else:
        recall = right_mention_num / gold_mention_num
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    # print("p_1, r_1, f_1:", precision, recall, f1)

    target_precision = 0 if target_p_span_num == 0 else target_right_span_num / target_p_span_num
    target_recall = 0 if target_g_span_num == 0 else target_right_span_num / target_g_span_num
    target_f1 = 0 if target_precision + target_recall == 0 else (2 * target_precision * target_recall) / (
                target_precision + target_recall)

    print("\ntarget_p_span_num:", target_p_span_num, " target_g_span_num:", target_g_span_num,
          " target_right_span_num:", target_right_span_num)
    print("target p:", target_precision, " r:", target_recall, " f1:", target_f1)

    source_precision = 0 if source_p_span_num == 0 else source_right_span_num / source_p_span_num
    source_recall = 0 if source_g_span_num == 0 else source_right_span_num / source_g_span_num
    source_f1 = 0 if source_precision + source_recall == 0 else (2 * source_precision * source_recall) / (
                source_precision + source_recall)

    print("\nsource_p_span_num:", source_p_span_num, " source_g_span_num:", source_g_span_num,
          " source_right_span_num:", source_right_span_num)
    print("source p:", source_precision, " r:", source_recall, " f1:", source_f1)

    print("\nspan type 分类结果：\n", classification_report(type_span_gold, type_span_predict, digits=4))

    return precision, recall, f1



def compute_fuzzy_ratio(mention, all_gold_mentions, fuzzy_ratio = 0.5):
    """
    计算当前mention与所有的gold mention是否满足模糊匹配
    """
    start = mention[0]
    end = mention[1]
    span_length = end - start + 1
    for g_mention in all_gold_mentions:
        gold_length = g_mention[1] - g_mention[0] + 1
        if start > g_mention[1] or end < g_mention[0]:
            continue
        elif start <= g_mention[0] and end >= g_mention[1]:  # span包括了gold span
            same_ratio_r = span_length / gold_length
            same_ratio_p = span_length / span_length
            same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
            if same_ratio_f >= fuzzy_ratio:
                return 1

        elif start <= g_mention[0] and end < g_mention[1]:  # span_start 与 gold span在前有交集
            same_ratio_r = (end - g_mention[0] + 1) / gold_length
            same_ratio_p = (end - g_mention[0] + 1) / span_length
            same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
            if same_ratio_f >= fuzzy_ratio:
                return 1

        elif g_mention[0] < start <= g_mention[1]:
            if end <= g_mention[1]:  # gold span 包含了 span
                same_ratio_r = span_length / gold_length
                same_ratio_p = span_length / span_length
                same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                if same_ratio_f >= fuzzy_ratio:
                    return 1
            else:  # end > gold span end   向后有交集
                same_ratio_r = (g_mention[1] - start + 1) / gold_length
                same_ratio_p = (g_mention[1] - start + 1) / span_length
                same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                if same_ratio_f >= fuzzy_ratio:
                    return 1

    return 0

def find_fuzzy_gold_clu(mention, gold_clusters, fuzzy_ratio = 0.5):
    """
    根据mention返回gold cluster中满足模糊匹配值的所在gold clu
    """
    start = mention[0]
    end = mention[1]
    span_length = end - start + 1
    for g_clu in gold_clusters:
        for g_mention in g_clu:
            gold_length = g_mention[1] - g_mention[0] + 1
            if start > g_mention[1] or end < g_mention[0]:    # mention 在当前gold mention 之外（没有任何交集）
                continue
            elif start <= g_mention[0] and end >= g_mention[1]:  # span包括了gold span
                same_ratio_r = span_length / gold_length
                same_ratio_p = span_length / span_length
                same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                if same_ratio_f >= fuzzy_ratio:
                    return g_clu

            elif start <= g_mention[0] and end < g_mention[1]:  # span_start 与 gold span在前有交集
                same_ratio_r = (end - g_mention[0] + 1) / gold_length
                same_ratio_p = (end - g_mention[0] + 1) / span_length
                same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                if same_ratio_f >= fuzzy_ratio:
                    return g_clu

            elif g_mention[0] < start <= g_mention[1]:
                if end <= g_mention[1]:  # gold span 包含了 span
                    same_ratio_r = span_length / gold_length
                    same_ratio_p = span_length / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_ratio:
                        return g_clu
                else:  # end > gold span end   向后有交集
                    same_ratio_r = (g_mention[1] - start + 1) / gold_length
                    same_ratio_p = (g_mention[1] - start + 1) / span_length
                    same_ratio_f = 2 * same_ratio_r * same_ratio_p / (same_ratio_r + same_ratio_p)
                    if same_ratio_f >= fuzzy_ratio:
                        return g_clu

    return []

def Meta_evaluete_all_doc(pre_clusters_doc, gold_clusters_doc, mention_to_gold_doc):
    """计算所有句子隐喻链的预测结果"""

    right_clu_num, gold_clu_num, pre_clu_num = 0, 0, 0 
    for key, gold_cluster in gold_clusters_doc.items():
        gold_mention = [mention for clu in gold_cluster for mention in clu]
        pre_clusters = pre_clusters_doc[key]
        mention_to_gold = mention_to_gold_doc[key]
        
        pre_clu_num += len(pre_clusters)   #  --------要计算非隐喻句的结果
        if len(gold_cluster) == 0:   # 只计算metaphor sentence的预测结果
            continue
        else:
            gold_clu_num += len(gold_cluster)
        # print("pre_clusters:", pre_clusters, "\ngold clusters:", gold_cluster)
        # print("mention to gold:", mention_to_gold)
        
        for p_clu in pre_clusters:
            
            p_clu_set = set(p_clu)
            for p_men in p_clu:  # 遍历当前预测链中的mention
                mention_flag = 0   # 看当前的词是否能判断正确链
                if p_men in gold_mention:  # 如果在gold clu中
                    g_clusters = mention_to_gold[p_men]  # 所在的gold cluster,找出其他同一个链的mention
                    for g_clu in g_clusters:
                        g_clu_set = set(g_clu)
                        if len(p_clu_set & g_clu_set) >= 2:
                            right_clu_num += 1
                            mention_flag = 1
                            # print("right pre clu:", p_clu, " gold clu:", g_clu)
                            break
                if mention_flag == 1:
                    break
                    
    print("right clu num:", right_clu_num, " gold clu num:", gold_clu_num, " pre clu num:", pre_clu_num)
    if gold_clu_num == 0:
        r = 0
    else:
        r = right_clu_num / gold_clu_num
        
    if pre_clu_num == 0:
        p = 0
    else:
        p = right_clu_num / pre_clu_num

    f = 0 if r + p == 0 else 2 * p * r / (p + r)
    # print("accuracy right link num:", right_clu_num, " pre clu num:", pre_clu_num)
    # print("p, r, f:", p, r, f)

    return p, r, f

def Meta_evaluete_all_doc_fuzzy(pre_clusters_doc, gold_clusters_doc, mention_to_gold_doc):
    # 计算所有句子隐喻链的预测结果

    right_clu_num, gold_clu_num, pre_clu_num = 0, 0, 0
    for key, gold_clusters in gold_clusters_doc.items():
        gold_mentions = [mention for clu in gold_clusters for mention in clu]
        pre_clusters = pre_clusters_doc[key]
        mention_to_gold = mention_to_gold_doc[key]
        # print("gold clu:", gold_cluster)
        # print("pre clu:", pre_clusters)
        pre_clu_num += len(pre_clusters)    #  --------要计算非隐喻句的结果

        if len(gold_clusters) == 0:   # 只计算metaphor sentence的预测结果
            continue
        else:
            gold_clu_num += len(gold_clusters)

        right_clu_t = 0
        for p_clu in pre_clusters:
            if len(p_clu) == 1:  # 单链
                if p_clu[0] in gold_mentions:  # 还没加模糊处理
                    right_clu_num += 1
                    right_clu_t += 1
                elif compute_fuzzy_ratio(p_clu[0], gold_mentions) == 1:  # 加模糊处理
                    right_clu_num += 1
                    right_clu_t += 1

            elif len(p_clu) > 1:
                p_clu_set = set(p_clu)
                for p_men in p_clu:  # 遍历当前预测链中的mention
                    if p_men in gold_mentions:  # 如果在gold clu中
                        mention_flag = 0
                        g_clusters = mention_to_gold[p_men]  # 所在的gold cluster,找出其他同一个链的mention
                        for g_clu in g_clusters:
                            g_clu_set = set(g_clu)
                            if len(p_clu_set & g_clu_set) >= 2:
                                # print("no type right g clu:", g_clu, " p_clu:", p_clu)
                                right_clu_num += 1
                                right_clu_t += 1
                                mention_flag = 1
                                break
                        if mention_flag == 1:
                            break

                    elif compute_fuzzy_ratio(p_men, gold_mentions) == 1:
                        pre_in_gold_clu = find_fuzzy_gold_clu(p_men, gold_clusters)
                        if len(pre_in_gold_clu) > 1:  # 当前预测的mention所在的gold cluster
                            # print("fuzzy predict mention in gold clusters > 1:", pre_in_gold_clu)
                            p_gold_mention = [mention for mention in pre_in_gold_clu]
                            pre_in_gold_num = 0
                            for p_mention in p_clu:   # 查看当前预测链中其他mention是否模糊匹配值达到要求
                                if compute_fuzzy_ratio(p_mention, p_gold_mention) == 1:
                                    pre_in_gold_num += 1
                            if pre_in_gold_num >= 2:
                                # print("pre_in_gold_clu:", pre_in_gold_clu)
                                # print("p_clu:", p_clu)
                                right_clu_num += 1
                                right_clu_t += 1
                                # print("fuzzy right pre clu:", p_clu, " pre in gold clusters:", pre_in_gold_clu)
                                break
        if right_clu_t > len(pre_clusters):
            print("right_clu_t:", right_clu_t, " len(pre_clusters):", len(pre_clusters))
    print("\nright clu num:", right_clu_num, " pre clu num:", pre_clu_num, " gold_clu_num:", gold_clu_num)
    if gold_clu_num == 0:
        r = 0
    else:
        if right_clu_num >= gold_clu_num:
            r = 1
        else:
            r = right_clu_num / gold_clu_num
    if pre_clu_num == 0:
        p = 0
    else:
        p = right_clu_num / pre_clu_num
    f = 0 if r + p == 0 else 2 * p * r / (p + r)

    # print("p, r, f:", p, r, f)

    return p, r, f

def Meta_evaluete_all_doc_type(pre_clusters_doc, gold_clusters_doc, mention_to_gold_doc, doc_gold_spans_type, doc_predict_span_2_type, doc_sentences):
    """
    span pair type识别的结果
    计算所有句子隐喻链的预测结果同时也要看预测链中的span type是否正确
    """

    right_clu_num, gold_clu_num, pre_clu_num = 0, 0, 0
    g_right_clu_num, unright_p_clu_num, un_in_gold_p_clu = 0, 0, 0 # check gold clu
    all_g_type, all_p_type = [], []
    for key, gold_cluster in gold_clusters_doc.items():
        gold_mention = [mention for clu in gold_cluster for mention in clu]
        pre_clusters = pre_clusters_doc[key]
        mention_to_gold = mention_to_gold_doc[key]
        
        pre_clu_num += len(pre_clusters)   #  --------要计算非隐喻句的结果
        
        if len(gold_cluster) == 0:   # 只计算metaphor sentence的预测结果
            continue
        else:
            gold_clu_num += len(gold_cluster)
        # print("pre_clusters:", pre_clusters, "\ngold clusters:", gold_cluster)
        # print("mention to gold:", mention_to_gold)
        
        sentence = doc_sentences[key]
        
        predict_span_2_type = doc_predict_span_2_type[key]

        gold_spans_type = doc_gold_spans_type[key]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            g_span_type_change = 0 if g_span_type[1] == 0 else 1    # 变为二分类来判断
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = g_span_type_change   # g_span_type[1]
            all_g_type.append(g_span_type[1])

            if g_span in predict_span_2_type:
                all_p_type.append(predict_span_2_type[g_span])
            else:
                all_p_type.append(5)
       
        for p_clu in pre_clusters:
            # 默认预测链中只有两个span
            span_a = p_clu[0]
            span_b = p_clu[1]
            span_a_type = 0 if predict_span_2_type[span_a] == 0 else 1
            span_b_type = 0 if predict_span_2_type[span_b] == 0 else 1
            if p_clu in gold_cluster or (span_b, span_a) in gold_cluster:
                if span_a_type == gold_span_2_type[span_a] and span_b_type == gold_span_2_type[span_b]:
                    right_clu_num += 1
                else:
                    unright_p_clu_num += 1
                    # print("\ngold_cluster:", gold_cluster)
                    # print("pre_clusters:", pre_clusters)
                    # print("gold_span_2_type :", gold_span_2_type)
                    # print("predict_span_2_type:", predict_span_2_type)

                    # print("not right p clu:", p_clu)
            else:
                un_in_gold_p_clu += 1
           
        
        not_find_gold_clu_str = "not_find_gold_clu_str :"
        find_gold_clu_str = "find_gold_clu_str :"
        not_find_g_clu = []
        for g_clu in gold_cluster:
            span_a = g_clu[0]
            span_b = g_clu[1]
            span_a_g_type = gold_span_2_type[span_a]
            span_b_g_type = gold_span_2_type[span_b]
            span_a_str = ''.join(sentence[span_a[0]:span_a[1]+1])
            span_b_str = ''.join(sentence[span_b[0]:span_b[1]+1])
            g_clu_flag = 0
            if g_clu in pre_clusters or (span_b, span_a) in pre_clusters:
                span_a_p_type = 0 if predict_span_2_type[span_a] == 0 else 1
                span_b_p_type = 0 if predict_span_2_type[span_b] == 0 else 1
                if span_a_g_type == span_a_p_type and span_b_g_type == span_b_p_type:
                    g_right_clu_num += 1
                    g_clu_flag = 1
            if g_clu_flag == 0:
                not_find_gold_clu_str += '[' + span_a_str + str(span_a) + ',' + span_b_str + str(span_b) +'] '
                not_find_g_clu.append(g_clu)
            else:
                find_gold_clu_str += '[' + span_a_str + str(span_a) + ',' + span_b_str + str(span_b) +'] '
        # if len(not_find_g_clu) > 0:
        #     print("\nsentence ", key, ''.join(sentence), "gold cluster:", gold_cluster)
        #     print(not_find_gold_clu_str)
        #     print(find_gold_clu_str)
                
    print("\nright clu num:", right_clu_num, " gold clu num:", gold_clu_num, " pre clu num:", pre_clu_num)
    print("g_right_clu_num check:", g_right_clu_num, " unright p clu:", unright_p_clu_num, " un_in_gold_p_clu:", un_in_gold_p_clu)
    
    # print(" cluster of gold span type:", classification_report(all_p_type, all_g_type, digits=4))
    if gold_clu_num == 0:
        r = 0
    else:
        r = right_clu_num / gold_clu_num
    if pre_clu_num == 0:
        p = 0
    else:
        p = right_clu_num / pre_clu_num

    f = 0 if r + p == 0 else 2 * p * r / (p + r)
    # print("accuracy right link num:", right_clu_num, " pre clu num:", pre_clu_num)
    # print("p, r, f:", p, r, f)

    return p, r, f

def Meta_evaluete_all_doc_one_type(pre_clusters_doc, gold_clusters_doc, mention_to_gold_doc, doc_gold_spans_type, meta_span_types, doc_top_spans):
    """
    one type classify
    计算所有句子隐喻链的预测结果
    """

    right_clu_num, gold_clu_num, pre_clu_num = 0, 0, 0
    for key, gold_cluster in gold_clusters_doc.items():
        gold_mention = [mention for clu in gold_cluster for mention in clu]
        pre_clusters = pre_clusters_doc[key]
        mention_to_gold = mention_to_gold_doc[key]

        pre_clu_num += len(pre_clusters)   #  --------要计算非隐喻句的结果
        if len(gold_cluster) == 0:   # 只计算metaphor sentence的预测结果
            continue
        else:
            gold_clu_num += len(gold_cluster)
        # print("pre_clusters:", pre_clusters, "\ngold clusters:", gold_cluster)
        # print("mention to gold:", mention_to_gold)

        gold_spans_type = doc_gold_spans_type[key]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = 0 if g_span_type[1] == 0 else 1

        top_spans_type = meta_span_types[key]
        top_spans = doc_top_spans[key]
        top_span_type_dict = {}
        for j, top_span in enumerate(top_spans):
            top_span_type_dict[top_span] = 0 if int(top_spans_type[j]) == 0 else 1

        for p_clu in pre_clusters:
            span_a = p_clu[0]
            span_b = p_clu[1]
            span_a_p_type = top_span_type_dict[span_a]
            span_b_p_type = top_span_type_dict[span_b]

            if span_a in gold_mention:
                g_clusters = mention_to_gold[span_a]  # 所在的gold cluster,找出其他同一个链的mention
                span_a_g_type = gold_span_2_type[span_a]
                for g_clu in g_clusters:
                    if span_b in g_clu:
                        span_b_g_type = gold_span_2_type[span_b]
                        if span_a_p_type == span_a_g_type and span_b_p_type == span_b_g_type:
                            right_clu_num += 1
                            break
            elif span_b in gold_mention:
                g_clusters = mention_to_gold[span_b]  # 所在的gold cluster,找出其他同一个链的mention
                span_b_g_type = gold_span_2_type[span_b]
                for g_clu in g_clusters:
                    if span_a in g_clu:
                        span_a_g_type = gold_span_2_type[span_a]
                        if span_a_p_type == span_a_g_type and span_b_p_type == span_b_g_type:
                            right_clu_num += 1
                            break

    print("\nright clu num:", right_clu_num, " gold clu num:", gold_clu_num, " pre clu num:", pre_clu_num)
    if gold_clu_num == 0:
        r = 0
    else:
        r = right_clu_num / gold_clu_num
    if pre_clu_num == 0:
        p = 0
    else:
        p = right_clu_num / pre_clu_num

    f = 0 if r + p == 0 else 2 * p * r / (p + r)
    # print("accuracy right link num:", right_clu_num, " pre clu num:", pre_clu_num)
    # print("p, r, f:", p, r, f)

    return p, r, f



def Meta_evaluete_all_doc_fuzzy_one_type(pre_clusters_doc, gold_clusters_doc, mention_to_gold_doc, doc_gold_spans_type, meta_span_types, doc_top_spans):
    """
    计算所有句子隐喻链的预测结果， 同时看 span type ---》fuzzy result
    """

    right_clu_num, gold_clu_num, pre_clu_num = 0, 0, 0
    for key, gold_clusters in gold_clusters_doc.items():
        gold_mentions = [mention for clu in gold_clusters for mention in clu]
        pre_clusters = pre_clusters_doc[key]
        mention_to_gold = mention_to_gold_doc[key]
        # print("gold clu:", gold_cluster)
        # print("pre clu:", pre_clusters)
        pre_clu_num += len(pre_clusters)    #  --------要计算非隐喻句的结果

        if len(gold_clusters) == 0:   # 只计算metaphor sentence的预测结果
            continue
        else:
            gold_clu_num += len(gold_clusters)

        gold_spans_type = doc_gold_spans_type[key]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = 0 if g_span_type[1] == 0 else 1

        top_spans_type = meta_span_types[key]
        top_spans = doc_top_spans[key]
        top_span_type_dict = {}
        for j, top_span in enumerate(top_spans):
            top_span_type_dict[top_span] = 0 if int(top_spans_type[j]) == 0 else 1

        right_clu_t = 0    # 看当前判断的预测链数量是否有不对
        for p_clu in pre_clusters:
            span_a = p_clu[0]
            span_b = p_clu[1]
            span_a_p_type = top_span_type_dict[span_a]
            span_b_p_type = top_span_type_dict[span_b]

            if span_a in gold_mentions:
                g_clusters = mention_to_gold[span_a]  # 所在的gold cluster,找出其他同一个链的mention
                span_a_g_type = gold_span_2_type[span_a]
                for g_clu in g_clusters:
                    if span_b in g_clu:
                        span_b_g_type = gold_span_2_type[span_b]
                        if span_a_p_type == span_a_g_type and span_b_p_type == span_b_g_type:
                            right_clu_num += 1
                            right_clu_t += 1
                            break
            elif span_b in gold_mentions:
                g_clusters = mention_to_gold[span_b]  # 所在的gold cluster,找出其他同一个链的mention
                span_b_g_type = gold_span_2_type[span_b]
                for g_clu in g_clusters:
                    if span_a in g_clu:
                        span_a_g_type = gold_span_2_type[span_a]
                        if span_a_p_type == span_a_g_type and span_b_p_type == span_b_g_type:
                            right_clu_num += 1
                            right_clu_t += 1
                            break

            elif compute_fuzzy_ratio(span_a, gold_mentions) == 1 or compute_fuzzy_ratio(span_b, gold_mentions) == 1:
                if compute_fuzzy_ratio(span_a, gold_mentions) == 1:
                    pre_in_gold_clu = find_fuzzy_gold_clu(span_a, gold_clusters)
                else:
                    pre_in_gold_clu = find_fuzzy_gold_clu(span_b, gold_clusters)

                if len(pre_in_gold_clu) > 0:  # 当前预测的mention所在的gold cluster >0? or >1?
                    # print("pre_in_gold_clu > 0:", pre_in_gold_clu)
                    p_gold_mention = [mention for mention in pre_in_gold_clu]
                    pre_in_gold_num = 0
                    for p_mention in p_clu:  # 查看当前预测链中其他mention是否模糊匹配值达到要求
                        if compute_fuzzy_ratio(p_mention, p_gold_mention) == 1:
                            pre_in_gold_num += 1
                    if pre_in_gold_num >= 2:
                        g_a_type = gold_span_2_type[p_gold_mention[0]]  # 不知道具体那个span对应着span a或b
                        g_b_type = gold_span_2_type[p_gold_mention[1]]
                        # print("span a:", span_a, " predict a type:", span_a_p_type)
                        # print("span b:", span_b, " predict b type:", span_b_p_type)
                        # print("gold span a:", p_gold_mention[0], " gold type a:", g_a_type)
                        # print("gold span b:", p_gold_mention[1], " gold type b:", g_b_type)
                        
                        right_clu_flag = 0
                        if g_a_type == span_a_p_type:
                            if g_b_type == span_b_p_type:
                                right_clu_flag = 1
                        elif g_a_type == span_b_p_type:
                            if g_b_type == span_a_p_type:
                                right_clu_flag = 1
                        if right_clu_flag == 1:
                            right_clu_num += 1
                            right_clu_t += 1
                        # print("fuzzy right pre clu:", p_clu, " pre in gold clusters:", pre_in_gold_clu)
                        # break

        if right_clu_t > len(pre_clusters):
            print("right_clu_t:", right_clu_t, " len(pre_clusters):", len(pre_clusters))
    print("\nright clu num:", right_clu_num, " pre clu num:", pre_clu_num, " gold_clu_num:", gold_clu_num)
    if gold_clu_num == 0:
        r = 0
    else:
        if right_clu_num >= gold_clu_num:
            r = 1
        else:
            r = right_clu_num / gold_clu_num
    if pre_clu_num == 0:
        p = 0
    else:
        p = right_clu_num / pre_clu_num
    f = 0 if r + p == 0 else 2 * p * r / (p + r)

    # print("p, r, f:", p, r, f)

    return p, r, f

def Meta_evaluete_all_doc_type_fuzzy(pre_clusters_doc, gold_clusters_doc, mention_to_gold_doc, doc_gold_spans_type,
                               doc_predict_span_2_type):
    """
    span pair type fuzzy 识别的结果
    计算所有句子隐喻链的预测结果同时也要看预测链中的span type是否正确
    """

    right_clu_num, gold_clu_num, pre_clu_num = 0, 0, 0
    for key, gold_cluster in gold_clusters_doc.items():
        gold_mention = [mention for clu in gold_cluster for mention in clu]
        pre_clusters = pre_clusters_doc[key]
        mention_to_gold = mention_to_gold_doc[key]

        pre_clu_num += len(pre_clusters)  # --------要计算非隐喻句的结果

        if len(gold_cluster) == 0:  # 只计算metaphor sentence的预测结果
            continue
        else:
            gold_clu_num += len(gold_cluster)
        # print("pre_clusters:", pre_clusters, "\ngold clusters:", gold_cluster)
        # print("mention to gold:", mention_to_gold)

        gold_spans_type = doc_gold_spans_type[key]
        gold_span_2_type = {}
        for g_span_type in gold_spans_type:
            g_span = tuple(g_span_type[0])
            g_span_type_change = 0 if g_span_type[1] == 0 else 1  # 变为二分类来判断
            if g_span not in gold_span_2_type:
                gold_span_2_type[g_span] = g_span_type_change  # g_span_type[1]

        predict_span_2_type = doc_predict_span_2_type[key]

        for p_clu in pre_clusters:
            # 默认预测链中只有两个span
            span_a = p_clu[0]
            span_b = p_clu[1]
            span_a_p_type = 0 if predict_span_2_type[span_a] == 0 else 1
            span_b_p_type = 0 if predict_span_2_type[span_b] == 0 else 1
            if span_a in gold_mention:
                g_clusters = mention_to_gold[span_a]
                for g_clu in g_clusters:
                    if span_b in g_clu:
                        if span_a_p_type == gold_span_2_type[span_a] and span_b_p_type == gold_span_2_type[span_b]:
                            right_clu_num += 1
                        break
            elif span_b in gold_mention:
                g_clusters = mention_to_gold[span_b]
                for g_clu in g_clusters:
                    if span_a in g_clu:
                        if span_a_p_type == gold_span_2_type[span_a] and span_b_p_type == gold_span_2_type[span_b]:
                            right_clu_num += 1
                        break

            elif compute_fuzzy_ratio(span_a, gold_mention) == 1 or compute_fuzzy_ratio(span_b, gold_mention) == 1:
                if compute_fuzzy_ratio(span_a, gold_mention) == 1:
                    pre_in_gold_clu = find_fuzzy_gold_clu(span_a, gold_cluster)
                else:
                    pre_in_gold_clu = find_fuzzy_gold_clu(span_b, gold_cluster)

                if len(pre_in_gold_clu) > 0:  # 当前预测的mention所在的gold cluster >0? or >1?
                    # print("pre_in_gold_clu > 0:", pre_in_gold_clu)
                    p_gold_mention = [mention for mention in pre_in_gold_clu]
                    pre_in_gold_num = 0
                    for p_mention in p_clu:  # 查看当前预测链中其他mention是否模糊匹配值达到要求
                        if compute_fuzzy_ratio(p_mention, p_gold_mention) == 1:
                            pre_in_gold_num += 1
                    if pre_in_gold_num >= 2:
                        g_a_type = gold_span_2_type[p_gold_mention[0]]  # 不知道具体那个span对应着span a或b
                        g_b_type = gold_span_2_type[p_gold_mention[1]]
                        # print("span a:", span_a, " predict a type:", span_a_p_type)
                        # print("span b:", span_b, " predict b type:", span_b_p_type)
                        # print("gold span a:", p_gold_mention[0], " gold type a:", g_a_type)
                        # print("gold span b:", p_gold_mention[1], " gold type b:", g_b_type)
                        right_clu_flag = 0
                        if g_a_type == span_a_p_type:
                            if g_b_type == span_b_p_type:
                                right_clu_flag = 1
                        elif g_a_type == span_b_p_type:
                            if g_b_type == span_a_p_type:
                                right_clu_flag = 1
                        if right_clu_flag == 1:
                            right_clu_num += 1

    print("\nright clu num:", right_clu_num, " gold clu num:", gold_clu_num, " pre clu num:", pre_clu_num)
    if gold_clu_num == 0:
        r = 0
    else:
        r = right_clu_num / gold_clu_num
    if pre_clu_num == 0:
        p = 0
    else:
        p = right_clu_num / pre_clu_num

    f = 0 if r + p == 0 else 2 * p * r / (p + r)
    # print("accuracy right link num:", right_clu_num, " pre clu num:", pre_clu_num)
    # print("p, r, f:", p, r, f)

    return p, r, f


def Meta_span_type_result(doc_top_spans, doc_top_spans_type, doc_pre_clusters, doc_gold_spans_type,
                          doc_gold_clusters, doc_meta_v_span, doc_unmeta_v_span):

    right_span_type_num, gold_mention_num, pre_mention_num = 0, 0, 0
    top_span_num, v_check_num, v_pre_check = 0, 0, 0
    all_true_span_type, all_predict_span_type = [], []      # predict cluster 中 span type识别结果
    true_span_type_1, predict_span_type_1 = [], []       # gold cluster span 中 span type识别结果
    v_meta_right_num, v_meta_gold_num, v_meta_pre_num = 0, 0, 0
    for i, top_spans in doc_top_spans.items():
        # print("top spans:", top_spans)
        # doc_pre_span_type, doc_gold_span_type = [], []
        top_span_num += len(top_spans)
        top_spans_type = doc_top_spans_type[i]
        gold_spans_type = doc_gold_spans_type[i]

        gold_clusters = doc_gold_clusters[i]
        gold_mentions = [tuple(mention) for clu in gold_clusters for mention in clu]   # 有重复的mention
        # gold_mentions = set(gold_mentions)
        pre_clusters = doc_pre_clusters[i]
        pre_mentions = [mention for clu in pre_clusters for mention in clu]
        # pre_mentions = set(pre_mentions)
        # print("gold clusters:", gold_clusters)
        # print("predict clusters:", pre_clusters)
        # print("gold mentions:", gold_mentions)
        # print("predict mentions:", pre_mentions)

        # if len(gold_mentions) == 0:
        #     continue
        # else:
        gold_mention_num += len(gold_mentions)
        pre_mention_num += len(pre_mentions)

        meta_v_span = doc_meta_v_span[i]
        unmeta_v_span = doc_unmeta_v_span[i]
        all_v_span = meta_v_span + unmeta_v_span
        all_v_span = [tuple(span) for span in all_v_span]
        

        top_span_type_dict = {}
        for j, top_span in enumerate(top_spans):
            top_span_type_dict[top_span] = int(top_spans_type[j])

        gold_span_type_dict = {}
        v_gold_span, gold_span_check = [], []    # 防止答案中重复计算同一个隐喻动词（在多个链中出现）
        for one_span_type in gold_spans_type:
            gold_span = tuple(one_span_type[0])
            true_type_1 = int(one_span_type[1])
            gold_span_type_dict[gold_span] = true_type_1

            if int(one_span_type[1]) == 3 and one_span_type[0] not in v_gold_span:  # 记录隐喻动词在答案中出现的次数
                v_meta_gold_num += 1
                v_gold_span.append(one_span_type[0])

            if gold_span not in gold_span_check:
                true_span_type_1.append(true_type_1)
                gold_span_check.append(gold_span)
                if gold_span in top_span_type_dict:
                    pre_type_1 = top_span_type_dict[gold_span]
                else:
                    pre_type_1 = 5
                predict_span_type_1.append(pre_type_1)
            
        
        v_pre_span, v_right_span = [], []
        pre_spans_all_check, gold_span_all_check = [], []
        for pre_mention in pre_mentions:
            predict_type = top_span_type_dict[tuple(pre_mention)]

            if pre_mention in gold_mentions:
                true_type = gold_span_type_dict[tuple(pre_mention)]
            else:
                true_type = 5     # None

            if pre_mention in all_v_span and pre_mention not in v_pre_span:
                v_meta_pre_num += 1       # 预测为隐喻的动词
                v_pre_span.append(pre_mention)

            if true_type == 3 and pre_mention not in v_right_span:
                v_meta_right_num += 1         # 预测正确（在gold span中）的隐喻动词
                v_right_span.append(pre_mention)

            if predict_type == true_type:
                right_span_type_num += 1

            if pre_mention not in pre_spans_all_check:
                all_predict_span_type.append(predict_type)
                all_true_span_type.append(true_type)
                pre_spans_all_check.append(pre_mention)


    if pre_mention_num == 0:
        p = 0
    else:
        p = right_span_type_num / pre_mention_num
    if gold_mention_num == 0:
        r = 0
    else:
        r = right_span_type_num / gold_mention_num

    f = 0 if r + p == 0 else 2 * p * r / (p + r)

    print("length of true span type:", len(true_span_type_1), "length of pre span type:", len(predict_span_type_1))
    print("混淆矩阵结果：\n", confusion_matrix(true_span_type_1, predict_span_type_1))
    # print("predict mention num:", pre_mention_num, "   gold mention num:", gold_mention_num)
    print("v_meta_right_num:", v_meta_right_num, " v_meta_gold_num:", v_meta_gold_num, " v_meta_pre_num:", v_meta_pre_num)
    
    if v_meta_gold_num == 0:
        v_recall = 0
    else:
        v_recall = v_meta_right_num / v_meta_gold_num

    if v_meta_pre_num == 0:
        v_precision = 0
    else:
        v_precision = v_meta_right_num / v_meta_pre_num

    f_v = 0 if v_recall + v_precision == 0 else 2 * v_recall * v_precision / (v_recall + v_precision)
    print("v_p, v_r, v_f:", v_precision, " ", v_recall, " ", f_v, "\n")
    print("\ndifferent span type detection result:\n", classification_report(true_span_type_1, predict_span_type_1,
                                                                             digits=4))

    print("all predict span type result:\n", classification_report(all_true_span_type, all_predict_span_type,
          digits=4))
    return p, r, f

# span pair type
def span_pair_metaphor_link(predict_clusters, gold_clusters, top_span_starts, top_span_ends,
                            all_span_pair_types, top_antecedents):
    """
    对每句话的预测结果得到预测链和gold cluster 中span type结果
    """
    # print("predict_clusters:", predict_clusters)
    # print("gold_clusters:", gold_clusters)
    predict_span_2_span_dict = {}
    for clu in predict_clusters:
        span_a = clu[0]
        span_b = clu[1]

        if span_a not in predict_span_2_span_dict:
            predict_span_2_span_dict[span_a] = [span_b]
        else:
            predict_span_2_span_dict[span_a].append(span_b)
        if span_b not in predict_span_2_span_dict:
            predict_span_2_span_dict[span_b] = [span_a]
        else:
            predict_span_2_span_dict[span_b].append(span_a)

    gold_span_2_span_dict = {}
    for clu in gold_clusters:
        span_a = clu[0]
        span_b = clu[1]

        if span_a not in gold_span_2_span_dict:
            gold_span_2_span_dict[span_a] = [span_b]
        else:
            gold_span_2_span_dict[span_a].append(span_b)

        if span_b not in gold_span_2_span_dict:
            gold_span_2_span_dict[span_b] = [span_a]
        else:
            gold_span_2_span_dict[span_b].append(span_a)

    predict_span_2_type = {}
    gold_span_2_type = {}
    check_gold_span_type = []
    top_span_2_gold_span = {}    # top_span_2_gold_span只保存在gold cluster中的span
    for i, start in enumerate(top_span_starts):
        end = top_span_ends[i]
        top_span = (int(start), int(end))
        span_pair_types = all_span_pair_types[i]
        ants = top_antecedents[i]
        for j, ant in enumerate(ants):
            ant_span = (int(top_span_starts[ant]), int(top_span_ends[ant]))
            _, top_span_type = torch.max(span_pair_types[j][:5], 0)  # 返回下标
            _, ant_span_type = torch.max(span_pair_types[j][5:], 0)
            top_span_type = int(top_span_type)
            ant_span_type = int(ant_span_type)
            if top_span in predict_span_2_span_dict:
                predict_ant_list = predict_span_2_span_dict[top_span]
                if ant_span in predict_ant_list:
                    # predict cluster一定会被包含其中
                    # if top_span in predict_span_2_type:
                    #     print("predict span before type:", predict_span_2_type[top_span], " now type:", top_span_type)
                    predict_span_2_type[top_span] = top_span_type
                    predict_span_2_type[ant_span] = ant_span_type

            if top_span in gold_span_2_span_dict:
                gold_ant_list = gold_span_2_span_dict[top_span]
                if ant_span in gold_ant_list:
                    # gold_span_2_type先只保存出现在预测中的gold cluster type结果
                    gold_span_2_type[top_span] = top_span_type
                    gold_span_2_type[ant_span] = ant_span_type
                else:
                    # 当前的隐喻链不在gold cluster中，则比较单独的span是否是gold span，存对应预测的type
                    # top span是在gold cluster中的
                    if top_span not in top_span_2_gold_span:
                        top_span_2_gold_span[top_span] = {}    # 存top span对应预测各个type的次数，最后取次数最多的type
                        top_span_2_gold_span[top_span][top_span_type] = 1
                    else:
                        if top_span_type not in top_span_2_gold_span[top_span]:
                            top_span_2_gold_span[top_span][top_span_type] = 1
                        else:
                            top_span_2_gold_span[top_span][top_span_type] += 1
                    check_gold_span_type.append([top_span, top_span_type])

                    if ant_span in gold_span_2_span_dict:
                        # 判断ant span是否是gold span
                        if ant_span not in top_span_2_gold_span:
                            top_span_2_gold_span[ant_span] = {}
                            top_span_2_gold_span[ant_span][ant_span_type] = 1
                        else:
                            if ant_span_type not in top_span_2_gold_span[ant_span]:
                                top_span_2_gold_span[ant_span][ant_span_type] = 1
                            else:
                                top_span_2_gold_span[ant_span][ant_span_type] += 1

                        check_gold_span_type.append([ant_span, ant_span_type])


    for gold_clu in gold_clusters:
        for span in gold_clu:
            if span not in gold_span_2_type and span in top_span_2_gold_span:
                max_type_value, max_type_count = 0, 0
                for type_value, type_count in top_span_2_gold_span[span].items():
                    if type_count > max_type_count:
                        max_type_count = type_count
                        max_type_value = type_value
                    gold_span_2_type[span] = max_type_value

    # print("predict_span_2_type:", predict_span_2_type)
    # print("gold_span_2_type:", gold_span_2_type)
    # print("check_gold_span_type:", check_gold_span_type)

    return predict_span_2_type, gold_span_2_type


def Meta_span_pair_type_result(doc_predict_span_2_type, doc_pre_clusters,
                               doc_gold_spans_type, doc_gold_clusters, doc_sentences,
                               doc_meta_v_span, doc_unmeta_v_span):
    # doc_gold_span_2_type, 
    right_span_type_num, gold_mention_num, pre_mention_num = 0, 0, 0
    v_check_num, v_pre_check = 0, 0
    all_true_span_type, all_predict_span_type = [], []   # predict
    true_spans_type, predict_spans_type = [], []        # gold
    v_meta_right_num, v_meta_gold_num, v_meta_pre_num = 0, 0, 0
    for i, predict_clusters in doc_pre_clusters.items():
        sentence = doc_sentences[i]
        one_sen_right_span_num = 0
        predict_span_2_type = doc_predict_span_2_type[i]      # 在隐喻链中的span的type
        # print("predict_span_2_type:", predict_span_2_type)
        # print("predict_clusters:", predict_clusters)
        
        # gold_span_2_type = doc_gold_span_2_type[i]

        gold_spans_type = doc_gold_spans_type[i]      # gold span预测的type
        gold_clusters = doc_gold_clusters[i]
        
        gold_mentions = [tuple(mention) for clu in gold_clusters for mention in clu]

        pre_mentions = [mention for clu in predict_clusters for mention in clu]

        gold_mention_num += len(gold_mentions)      # 是否要去除重复的span数量？？？
        pre_mention_num += len(pre_mentions)

        meta_v_span = doc_meta_v_span[i]
        unmeta_v_span = doc_unmeta_v_span[i]
        all_v_span = meta_v_span + unmeta_v_span
        all_v_span = [tuple(span) for span in all_v_span]
        # if i < 20:
        #     print("all v span:", all_v_span)


        not_find_v_span_list = []
        gold_span_type_dict = {}
        v_gold_span, gold_span_check = [], []  # 防止答案中重复计算同一个隐喻动词（在多个链中出现）
        for one_span_type in gold_spans_type:
            gold_span = tuple(one_span_type[0])
            true_type_1 = int(one_span_type[1])
            gold_span_type_dict[gold_span] = true_type_1
            if int(one_span_type[1]) == 3 and one_span_type[0] not in v_gold_span:  # 记录隐喻动词在答案中出现的次数
                v_meta_gold_num += 1
                v_gold_span.append(one_span_type[0])

            if gold_span not in gold_span_check:
                true_spans_type.append(true_type_1)   # -
                gold_span_check.append(gold_span)

                # if gold_span in predict_span_2_type:    # gold span 在 predict cluster中
                #     pre_type_1 = predict_span_2_type[gold_span]
                # elif gold_span in gold_span_2_type:
                if gold_span in predict_span_2_type:      # 只看gold span的span type预测结果
                    pre_type_1 = predict_span_2_type[gold_span]
                else:
                    pre_type_1 = 5
                predict_spans_type.append(pre_type_1)  # -

                # see not found v gold span
                if true_type_1 == 3 and gold_span not in predict_span_2_type:
                    span_str = ''.join(sentence[gold_span[0]: gold_span[1]+1])
                    not_find_v_span_list.append(span_str+str(gold_span))
                    v_check_num += 1
        # print see v span
        # if len(not_find_v_span_list) > 0:
        #     print("example ", i, " sentence-", ''.join(sentence[1:-1]), " not find v span are:", not_find_v_span_list)

        v_pre_span, v_right_span = [], []
        pre_spans_all_check, gold_span_all_check = [], []
        other_type_v_span = []
        for pre_mention in pre_mentions:
            predict_type = predict_span_2_type[tuple(pre_mention)]

            if pre_mention in all_v_span and pre_mention not in v_pre_span:
                v_meta_pre_num += 1       # 预测为隐喻的动词
                v_pre_span.append(pre_mention)

            if pre_mention in gold_mentions:
                true_type = gold_span_type_dict[tuple(pre_mention)]
            else:
                true_type = 5  # None

            # if predict_type == true_type and predict_type == 3 and pre_mention not in v_right_span:
            if true_type == 3 and pre_mention not in v_right_span:
                v_meta_right_num += 1         # 预测正确（在gold span中）的隐喻动词
                v_right_span.append(pre_mention)

            if predict_type == true_type:
                right_span_type_num += 1
                # one_sen_right_span_num += 1

            if pre_mention not in pre_spans_all_check:
                all_predict_span_type.append(predict_type)
                all_true_span_type.append(true_type)
                pre_spans_all_check.append(pre_mention)
        # print("one sen right span num:", one_sen_right_span_num)

    if pre_mention_num == 0:
        p = 0
    else:
        p = right_span_type_num / pre_mention_num

    if gold_mention_num == 0:
        r = 0
    else:
        r = right_span_type_num / gold_mention_num

    f = 0 if r + p == 0 else 2 * p * r / (p + r)

    # print("length of true span type:", len(true_spans_type), "length of pre span type:", len(predict_spans_type))
    print("混淆矩阵结果：\n", confusion_matrix(true_spans_type, predict_spans_type))
    # print("predict mention num:", pre_mention_num, "   gold mention num:", gold_mention_num)
    print("v gold not find check:", v_check_num, "  v_check num:", v_pre_check)
    print("v_meta_right_num:", v_meta_right_num, " v_meta_gold_num:", v_meta_gold_num, " v_meta_pre_num:", v_meta_pre_num)
    if v_meta_gold_num == 0:
        v_recall = 0
    else:
        v_recall = v_meta_right_num / v_meta_gold_num

    if v_meta_pre_num == 0:
        v_precision = 0
    else:
        v_precision = v_meta_right_num / v_meta_pre_num

    f_v = 0 if v_recall + v_precision == 0 else 2 * v_recall * v_precision / (v_recall + v_precision)
    print("v_p, v_r, v_f:", v_precision, " ", v_recall, " ", f_v, "\n")
    print("\ndifferent span type detection result(in gold):\n", classification_report(true_spans_type, predict_spans_type,
                                                                             digits=4))

    print("\nall predict span type result:\n", classification_report(all_true_span_type, all_predict_span_type,
          digits=4))
    return p, r, f


def get_meta_type(span_a_type, span_b_type):
    if span_a_type == "本体":
        if span_b_type == "喻体":
            return "名词隐喻"
        elif span_b_type == "喻体属性":
            return "形容词隐喻"               # "形容词隐喻"
        elif span_b_type == "喻体部件":
            return "形容词隐喻"
        elif span_b_type == "喻体动作":
            return "动词隐喻"
        else:
            return "其他隐喻"
    elif span_b_type == "本体":
        if span_a_type == "喻体":
            return "名词隐喻"
        elif span_a_type == "喻体属性":
            return "形容词隐喻"                # "形容词隐喻"
        elif span_a_type == "喻体部件":
            return "形容词隐喻"
        elif span_a_type == "喻体动作":
            return "动词隐喻"
        else:
            return "其他隐喻"

def different_metaphor_result(doc_pre_clusters, doc_gold_clusters_type, doc_top_spans, doc_top_spans_type,
                              doc_sentences, doc_gold_spans_type):
    """
    查看不同类型的隐喻识别情况：名词隐喻、明喻、动词隐喻、形容词隐喻     # A is B
    """
    is_gold_meta_num, is_pre_meta_num, is_all_meta_num = 0, 0, 0
    simile_gold_meta_num, simile_pre_meta_num, simile_all_meta_num = 0, 0, 0
    v_gold_meta_num, v_pre_meta_num, v_all_meta_num = 0, 0, 0
    adj_gold_meta_num, adj_pre_meta_num, adj_all_meta_num = 0, 0, 0
    type_int_2_str = {0: "本体", 1: "喻体", 2: "喻体属性", 3: "喻体动作", 4: "喻体部件"}
    xiang_sen, other_noun_meta = [], []
    noun_gold_meta_num, noun_pre_meta_num, noun_all_meta_num = 0, 0, 0
    other_noun_num = 0
    for key, gold_clusters_type in doc_gold_clusters_type.items():
        if len(gold_clusters_type) == 0:
            continue
        sentence = doc_sentences[key]
        sen_str = ''.join(sentence[1:-1])
        top_spans = doc_top_spans[key]
        top_spans_type = doc_top_spans_type[key]
        # print("top_spans_type:", top_spans_type)
        top_span_type_dict = {}
        for i, top_span in enumerate(top_spans):
            top_span_type_dict[top_span] = int(top_spans_type[i])

        gold_mention_type = doc_gold_spans_type[key]
        gold_mention_2_type = {}
        for g_mention_type in gold_mention_type:
            g_span = tuple(g_mention_type[0])
            if g_span not in gold_mention_2_type:
                gold_mention_2_type[g_span] = g_mention_type[1]

        pre_mention_2_clu = {}
        pre_clusters = doc_pre_clusters[key]
        for p_clu in pre_clusters:
            for mention in p_clu:
                pre_mention_2_clu[mention] = p_clu

            span_a = p_clu[0]
            span_b = p_clu[1]
            span_a_str = ''.join(sentence[span_a[0]:span_a[1] + 1])
            span_b_str = ''.join(sentence[span_b[0]:span_b[1] + 1])
            span_a_type = type_int_2_str[top_span_type_dict[p_clu[0]]]
            span_b_type = type_int_2_str[top_span_type_dict[p_clu[1]]]
            relation_type = get_meta_type(span_a_type, span_b_type)

            if relation_type == "动词隐喻":
                v_all_meta_num += 1
            elif relation_type == "形容词隐喻":
                adj_all_meta_num += 1
                # print("的 sentence：", " i:", i, "--", ''.join(sentence[1:-1]))
            else:  # 名词隐喻、明喻

                if span_a[1] < span_b[0]:
                    mid_str = sentence[span_a[1]+1:span_b[0]]
                elif span_b[1] < span_a[0]:
                    mid_str = sentence[span_b[1]+1:span_a[0]]
                else:  # 嵌套关系
                    mid_str = ''
                    print("span a:", span_a_str, " span b:", span_b_str)
                    print("other mid str for span a:", span_a, " span b:", span_b)
                if "像" in mid_str or "如" in mid_str or "仿佛" in mid_str:
                # if ("像" in sentence or "如" in sentence or "仿佛" in sen_str) and "如果" not in sen_str:
                    # if sentence not in xiang_sen:
                        # xiang_sen.append(sentence)
                        simile_all_meta_num += 1
                        # print("像 sentence：", " i:", i, "--", ''.join(sentence[1:-1]))
                # elif "是" in mid_str:
                #     is_all_meta_num += 1
                #     print("\nmid str:", mid_str, " span a:", span_a_str, "span b:", span_b_str)
                #     print("是 sentence：", " i:", i, "--", ''.join(sentence[1:-1]))
                else:
                    other_noun_meta.append([span_a_str+str(span_a), span_b_str+str(span_b)])
                    noun_all_meta_num += 1


        for g_clu_type in gold_clusters_type:
            t_clu_type = 0
            if len(g_clu_type) == 3:
                clu_type = g_clu_type[2]
                if clu_type == "名词隐喻":
                    noun_gold_meta_num += 1
                    t_clu_type = 1
                elif clu_type == "明喻":
                    simile_gold_meta_num += 1
                    t_clu_type = 2
                elif clu_type == "动词隐喻":
                    v_gold_meta_num += 1
                    t_clu_type = 3
                elif clu_type == "形容词隐喻":
                    adj_gold_meta_num += 1
                    t_clu_type = 4

                if tuple(g_clu_type[0]) in pre_mention_2_clu:
                    predict_clu = pre_mention_2_clu[tuple(g_clu_type[0])]   # 找出gold span所在预测链
                    if tuple(g_clu_type[1]) in predict_clu:
                        g_span_a_type = gold_mention_2_type[tuple(g_clu_type[0])]
                        g_span_b_type = gold_mention_2_type[tuple(g_clu_type[1])]
                        p_span_a_type = top_span_type_dict[tuple(g_clu_type[0])]
                        p_span_b_type = top_span_type_dict[tuple(g_clu_type[1])]

                        if g_span_a_type == p_span_a_type and g_span_b_type == p_span_b_type:

                            if t_clu_type == 1:   # 判断当前链是什么类型
                                # print("gold clu:", g_clu_type, " predict clusters:", predict_clu)
                                noun_pre_meta_num += 1
                            elif t_clu_type == 2:
                                simile_pre_meta_num += 1
                            elif t_clu_type == 3:
                                v_pre_meta_num += 1
                            elif t_clu_type == 4:
                                adj_pre_meta_num += 1

            else:
                other_noun_num += 1

    print("noun_gold_meta_num",  noun_gold_meta_num, " noun_pre_meta_num:", noun_pre_meta_num, " noun_all_meta_num:",
          noun_all_meta_num)
    print("simile_gold_meta_num",  simile_gold_meta_num, " simile_pre_meta_num:", simile_pre_meta_num,
          " simile_all_meta_num:", simile_all_meta_num)
    print("v_gold_meta_num", v_gold_meta_num, " v_pre_meta_num:", v_pre_meta_num, " v_all_meta_num:", v_all_meta_num)
    print("adj_gold_meta_num", adj_gold_meta_num, " adj_pre_meta_num:", adj_pre_meta_num, "adj_all_meta_num",
          adj_all_meta_num, "\n")

    # print("other noun num:", other_noun_num, " other meta num:", other_meta_num)
    # print("other noun meta length:", len(other_noun_meta), " other noun meta are:\n", other_noun_meta)

    # if is_gold_meta_num == 0:
    #     is_recall = 0
    # else:
    #     is_recall = is_pre_meta_num / is_gold_meta_num
    # if is_all_meta_num == 0:
    #     is_precision = 0
    # else:
    #     is_precision = is_pre_meta_num / is_all_meta_num
    # f_is = 0 if (is_recall + is_precision) == 0 else 2 * is_recall * is_precision / (is_recall + is_precision)

    if noun_gold_meta_num == 0:
        noun_recall = 0
    else:
        noun_recall = noun_pre_meta_num / noun_gold_meta_num
    if noun_all_meta_num == 0:
        noun_precision = 0
    else:
        noun_precision = noun_pre_meta_num / noun_all_meta_num
    f_noun = 0 if (noun_recall + noun_precision) == 0 else 2 * noun_recall * noun_precision / (
                noun_recall + noun_precision)

    if simile_gold_meta_num == 0:
        simile_recall = 0
    else:
        simile_recall = simile_pre_meta_num / simile_gold_meta_num
    if simile_all_meta_num == 0:
        simile_precision = 0
    else:
        simile_precision = simile_pre_meta_num / simile_all_meta_num
    f_simile = 0 if simile_recall + simile_precision == 0 else 2 * simile_recall * simile_precision / (simile_recall + simile_precision)

    if v_gold_meta_num == 0:
        v_recall = 0
    else:
        v_recall = v_pre_meta_num / v_gold_meta_num
    if v_all_meta_num == 0:
        v_precision = 0
    else:
        v_precision = v_pre_meta_num / v_all_meta_num
    f_v = 0 if v_recall + v_precision == 0 else 2 * v_recall * v_precision / (v_recall + v_precision)

    if adj_gold_meta_num == 0:
        adj_recall = 0
    else:
        adj_recall = adj_pre_meta_num / adj_gold_meta_num
    if adj_all_meta_num == 0:
        adj_precision = 0
    else:
        adj_precision = adj_pre_meta_num / adj_all_meta_num
    f_adj = 0 if adj_recall + adj_precision == 0 else 2 * adj_recall * adj_precision / (adj_recall + adj_precision)

    print("noun_recall, precision and f1:", noun_recall, " ", noun_precision, " ", f_noun)
    print("simile_recall, precision and f1:", simile_recall, " ", simile_precision, " ", f_simile)
    print("v_recall, precision and f1:", v_recall, " ", v_precision, " ", f_v)
    print("adj_recall, precision and f1:", adj_recall, " ", adj_precision, " ", f_adj)
    
def compute_v_meta(doc_pre_clusters, doc_top_spans, doc_top_spans_type, doc_gold_spans_type):

    right_v_span_num, pre_v_span_num, gold_v_span_num = 0, 0, 0
    for i, top_spans in doc_top_spans.items():
        pre_clusters = doc_pre_clusters[i]
        pre_mentions = [mention for pre_clu in pre_clusters for mention in pre_clu]
        top_spans = doc_top_spans[i]
        top_spans_type = doc_top_spans_type[i]

        top_span_type_dict = {}
        for j, top_span in enumerate(top_spans):
            top_span_type_dict[top_span] = int(top_spans_type[j])

        gold_spans_type = doc_gold_spans_type[i]
        gold_span_type_dict = {}
        v_gold_span = []  # 防止答案中重复计算同一个隐喻动词（在多个链中出现）
        for one_span_type in gold_spans_type:
            gold_span_type_dict[tuple(one_span_type[0])] = int(one_span_type[1])
            if int(one_span_type[1]) == 3 and one_span_type[0] not in v_gold_span:  # 记录隐喻动词在答案中一共出现的次数
                v_gold_span.append(tuple(one_span_type[0]))
                gold_v_span_num += 1

        pre_v_span = []
        for pre_span in pre_mentions:
            span_type = top_span_type_dict[pre_span]
            if span_type == 3 and pre_span not in pre_v_span:
                pre_v_span_num += 1
            if pre_span in gold_span_type_dict:
                if gold_span_type_dict[pre_span] == 3:
                        right_v_span_num += 1

    print("right_v_span_num:", right_v_span_num, " pre_v_span_num:", pre_v_span_num, " gold_v_span_num:",
          gold_v_span_num)

    if pre_v_span_num == 0:
        v_precision = 0
    else:
        v_precision = right_v_span_num / pre_v_span_num

    if gold_v_span_num == 0:
        v_recall = 0
    else:
        v_recall = right_v_span_num / gold_v_span_num

    v_f1 = 0 if (v_recall + v_precision) == 0 else 2 * v_recall * v_precision / (v_recall + v_precision)

    return v_precision, v_recall, v_f1


def different_span_pair_metaphor_result(doc_pre_clusters, doc_gold_clusters_type, doc_sentences,
                                        doc_predict_span_2_type, label_true, doc_gold_spans_type):
    """
    span pair
    查看不同类型的隐喻识别情况：名词隐喻、明喻、动词隐喻、形容词隐喻       # A is B（名词是隐喻）
    """
    noun_gold_meta_num, noun_pre_meta_num, noun_all_meta_num = 0, 0, 0            # gold num 答案的数量
    simile_gold_meta_num, simile_pre_meta_num, simile_all_meta_num = 0, 0, 0     # pre num 预测正确的数量
    v_gold_meta_num, v_pre_meta_num, v_all_meta_num = 0, 0, 0                    # all num 预测的数量
    adj_gold_meta_num, adj_pre_meta_num, adj_all_meta_num = 0, 0, 0
    type_int_2_str = {0: "本体", 1: "喻体", 2: "喻体属性", 3: "喻体动作", 4: "喻体部件"}
    xiang_sen, other_noun_meta = [], []
    noun_meta_num, other_noun_num = 0, 0
    for key, gold_clusters_type in doc_gold_clusters_type.items():
        sen_label = label_true[key]
        if sen_label == 0:
            continue
        # if len(gold_clusters_type) == 0:
        #     continue
        sentence = doc_sentences[key]
        sen_str = ''.join(sentence[1:-1])
        
        gold_mention_type = doc_gold_spans_type[key]
        gold_mention_2_type = {}
        for g_mention_type in gold_mention_type:
            g_span = tuple(g_mention_type[0])
            if g_span not in gold_mention_2_type:
                gold_mention_2_type[g_span] = g_mention_type[1]
        
        predict_span_2_type = doc_predict_span_2_type[key]
        # gold_span_2_type = doc_gold_span_2_type[key]
        pre_mention_2_clu = {}
        pre_clusters = doc_pre_clusters[key]
        # 预测的链识别结果
        for p_clu in pre_clusters:
            for mention in p_clu:
                pre_mention_2_clu[mention] = p_clu

            span_a = p_clu[0]
            span_b = p_clu[1]
            span_a_str = ''.join(sentence[span_a[0]:span_a[1] + 1])
            span_b_str = ''.join(sentence[span_b[0]:span_b[1] + 1])
            span_a_type = type_int_2_str[predict_span_2_type[p_clu[0]]]
            span_b_type = type_int_2_str[predict_span_2_type[p_clu[1]]]
            relation_type = get_meta_type(span_a_type, span_b_type)

            if relation_type == "动词隐喻":
                v_all_meta_num += 1
            elif relation_type == "形容词隐喻":
                adj_all_meta_num += 1
                # print("adj sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
            else:  # 名词隐喻、明隐喻

                if span_a[1] < span_b[0]:
                    mid_str = sentence[span_a[1] + 1:span_b[0]]
                elif span_b[1] < span_a[0]:
                    mid_str = sentence[span_b[1] + 1:span_a[0]]
                else:  # 嵌套关系
                    mid_str = ''
                    # print("span a:", span_a_str, " span b:", span_b_str)
                    # print("other mid str for span a:", span_a, " span b:", span_b)
                if "像" in mid_str or "如" in mid_str or "仿佛" in mid_str:
                # if ("像" in sentence or "如" in sentence or "仿佛" in sen_str) and "如果" not in sen_str:
                    # if sentence not in xiang_sen:
                    #     xiang_sen.append(sentence)
                        simile_all_meta_num += 1
                        # print("像 sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
                else:
                    noun_all_meta_num += 1
                """
                elif "是" in mid_str:
                    is_all_meta_num += 1
                    # print("\nmid str:", mid_str, " span a:", span_a_str, "span b:", span_b_str)
                    # print("是 sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
                elif ("的" in mid_str or "之" in mid_str) and len(mid_str) == 1:
                    adj_all_meta_num += 1
                    # print("的 sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
                else:
                    # other_noun_meta_num += 1
                    other_noun_meta.append([span_a_str + str(span_a), span_b_str + str(span_b)])
                """
        # 看答案的链
        for g_clu_type in gold_clusters_type:
            t_clu_type = 0
            if len(g_clu_type) == 3:
                clu_type = g_clu_type[2]
                if clu_type == "名词隐喻":
                    noun_gold_meta_num += 1
                    t_clu_type = 1
                elif clu_type == "明喻":
                    simile_gold_meta_num += 1
                    t_clu_type = 2
                elif clu_type == "动词隐喻":
                    v_gold_meta_num += 1
                    t_clu_type = 3
                elif clu_type == "形容词隐喻":
                    adj_gold_meta_num += 1
                    t_clu_type = 4

                if tuple(g_clu_type[0]) in pre_mention_2_clu:
                    predict_clu = pre_mention_2_clu[tuple(g_clu_type[0])]  # 找出gold span所在预测链
                    
                    if tuple(g_clu_type[1]) in predict_clu :
                        g_span_a_type = gold_mention_2_type[tuple(g_clu_type[0])]
                        g_span_b_type = gold_mention_2_type[tuple(g_clu_type[1])]
                        p_span_a_type = predict_span_2_type[tuple(g_clu_type[0])]
                        p_span_b_type = predict_span_2_type[tuple(g_clu_type[1])]
                        
                        if g_span_a_type == p_span_a_type and g_span_b_type == p_span_b_type:                       
                            if t_clu_type == 1:  # 判断当前链是什么类型
                                # print("gold clu:", g_clu_type, " predict clusters:", predict_clu)
                                noun_pre_meta_num += 1
                            elif t_clu_type == 2:
                                simile_pre_meta_num += 1
                            elif t_clu_type == 3:
                                v_pre_meta_num += 1
                            elif t_clu_type == 4:
                                adj_pre_meta_num += 1

            else:
                other_noun_num += 1

    print("noun_gold_meta_num", noun_gold_meta_num, " noun_pre_meta_num:", noun_pre_meta_num, " noun_all_meta_num:",
          noun_all_meta_num)
    print("simile_gold_meta_num", simile_gold_meta_num, " simile_pre_meta_num:", simile_pre_meta_num,
          " simile_all_meta_num:", simile_all_meta_num)
    print("v_gold_meta_num", v_gold_meta_num, " v_pre_meta_num:", v_pre_meta_num, " v_all_meta_num:", v_all_meta_num)
    print("adj_gold_meta_num", adj_gold_meta_num, " adj_pre_meta_num:", adj_pre_meta_num, "adj_all_meta_num",
          adj_all_meta_num, "\n")

    # print("other noun num:", other_noun_num, " other meta num:", other_meta_num)
    # print("other noun meta length:", len(other_noun_meta), " other noun meta are:\n", other_noun_meta)

    if noun_gold_meta_num == 0:
        noun_recall = 0
    else:
        noun_recall = noun_pre_meta_num / noun_gold_meta_num
    if noun_all_meta_num == 0:
        noun_precision = 0
    else:
        noun_precision = noun_pre_meta_num / noun_all_meta_num
    f_noun = 0 if (noun_recall + noun_precision) == 0 else 2 * noun_recall * noun_precision / (noun_recall + noun_precision)

    if simile_gold_meta_num == 0:
        simile_recall = 0
    else:
        simile_recall = simile_pre_meta_num / simile_gold_meta_num
    if simile_all_meta_num == 0:
        simile_precision = 0
    else:
        simile_precision = simile_pre_meta_num / simile_all_meta_num
    f_simile = 0 if simile_recall + simile_precision == 0 else 2 * simile_recall * simile_precision / (
                simile_recall + simile_precision)

    if v_gold_meta_num == 0:
        v_recall = 0
    else:
        v_recall = v_pre_meta_num / v_gold_meta_num
    if v_all_meta_num == 0:
        v_precision = 0
    else:
        v_precision = v_pre_meta_num / v_all_meta_num
    f_v = 0 if v_recall + v_precision == 0 else 2 * v_recall * v_precision / (v_recall + v_precision)

    if adj_gold_meta_num == 0:
        adj_recall = 0
    else:
        adj_recall = adj_pre_meta_num / adj_gold_meta_num
    if adj_all_meta_num == 0:
        adj_precision = 0
    else:
        adj_precision = adj_pre_meta_num / adj_all_meta_num
    f_adj = 0 if adj_recall + adj_precision == 0 else 2 * adj_recall * adj_precision / (adj_recall + adj_precision)

    print("noun_recall, precision and f1:", noun_recall, " ", noun_precision, " ", f_noun)
    print("simile_recall, precision and f1:", simile_recall, " ", simile_precision, " ", f_simile)
    print("v_recall, precision and f1:", v_recall, " ", v_precision, " ", f_v)
    print("adj_recall, precision and f1:", adj_recall, " ", adj_precision, " ", f_adj)

def match_head(head_span, predict_spans):
    """找出预测链中是否有head部分与gold span的head匹配的span"""
    head_start = head_span[0]
    head_end = head_span[1]
    for p_span in predict_spans:
        p_start = p_span[0]
        p_end = p_span[0]
        if p_start <= head_start and head_end <= p_end:
            return p_span
    
    return []

def different_span_pair_metaphor_result_head(doc_pre_clusters, doc_gold_clusters_type, doc_sentences,
                                        doc_predict_span_2_type, label_true, doc_gold_spans_type):
    """
    查看不同类型的隐喻识别情况：名词隐喻、明喻、动词隐喻、形容词隐喻       # A is B（名词是隐喻）
    """
    noun_gold_meta_num, noun_pre_meta_num, noun_all_meta_num = 0, 0, 0            # gold num 答案的数量
    simile_gold_meta_num, simile_pre_meta_num, simile_all_meta_num = 0, 0, 0     # pre num 预测正确的数量
    v_gold_meta_num, v_pre_meta_num, v_all_meta_num = 0, 0, 0                    # all num 预测的数量
    adj_gold_meta_num, adj_pre_meta_num, adj_all_meta_num = 0, 0, 0
    type_int_2_str = {0: "本体", 1: "喻体", 2: "喻体属性", 3: "喻体动作", 4: "喻体部件"}
    xiang_sen, other_noun_meta = [], []
    noun_meta_num, other_noun_num = 0, 0
    for key, gold_clusters_type in doc_gold_clusters_type.items():
        sen_label = label_true[key]
        if sen_label == 0:
            continue
        sentence = doc_sentences[key]
        sen_str = ''.join(sentence[1:-1])
        
        gold_mention_type = doc_gold_spans_type[key]
        gold_mention_2_type = {}
        for g_mention_type in gold_mention_type:
            g_span = tuple(g_mention_type[0])
            if g_span not in gold_mention_2_type:
                gold_mention_2_type[g_span] = g_mention_type[1]
        
        predict_span_2_type = doc_predict_span_2_type[key]
        # gold_span_2_type = doc_gold_span_2_type[key]
        pre_mention_2_clu = {}
        pre_clusters = doc_pre_clusters[key]
        predict_spans = []
        # 预测的链识别结果
        for p_clu in pre_clusters:
            for mention in p_clu:
                pre_mention_2_clu[mention] = p_clu

            span_a = p_clu[0]
            span_b = p_clu[1]
            span_a_str = ''.join(sentence[span_a[0]:span_a[1] + 1])
            span_b_str = ''.join(sentence[span_b[0]:span_b[1] + 1])
            span_a_type = type_int_2_str[predict_span_2_type[p_clu[0]]]
            span_b_type = type_int_2_str[predict_span_2_type[p_clu[1]]]
            relation_type = get_meta_type(span_a_type, span_b_type)
            
            if span_a not in predict_spans:
                predict_spans.append(span_a)
            if span_b not in predict_spans:
                predict_spans.append(span_b)

            if relation_type == "动词隐喻":
                v_all_meta_num += 1
            elif relation_type == "形容词隐喻":
                adj_all_meta_num += 1
                # print("adj sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
            else:  # 名词隐喻、明隐喻

                if span_a[1] < span_b[0]:
                    mid_str = sentence[span_a[1] + 1:span_b[0]]
                elif span_b[1] < span_a[0]:
                    mid_str = sentence[span_b[1] + 1:span_a[0]]
                else:  # 嵌套关系
                    mid_str = ''
                if "像" in mid_str or "如" in mid_str or "仿佛" in mid_str:
                # if ("像" in sentence or "如" in sentence or "仿佛" in sen_str) and "如果" not in sen_str:
                    # if sentence not in xiang_sen:
                    #     xiang_sen.append(sentence)
                        simile_all_meta_num += 1
                        # print("像 sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
                else:
                    noun_all_meta_num += 1
               
        # 看答案的链
        for g_clu_type in gold_clusters_type:
            span_a = tuple(g_clu_type[0])
            span_a_str = ''.join(sentence[span_a[0]:span_a[1] + 1])
            span_a_length = span_a[1] - span_a[0] + 1
            if len(span_a_str) != span_a_length:
                print("sentence ", key, " ", sentence)
                print("span a had more token:", span_a)
            
            a_str_cut_list = SnowNLP(span_a_str).words
            if len(a_str_cut_list) > 1 and span_a_length > 2:  # 不止一个词组成
                final_start, j = 0, 0
                while j < len(a_str_cut_list) - 1:
                    final_start += len(a_str_cut_list[j])
                    j += 1
                a_new_start = span_a[0] + final_start
                if a_new_start <= span_a[1]:
                    new_span_a = [a_new_start, span_a[1]]
                else:
                    print("sentence:", key)
                    print("span a wrong!", span_a)
                    print("span a cut:", a_str_cut_list)
            else:
                new_span_a = span_a   # head span a
            new_span_a = tuple(new_span_a)
            
            span_b = tuple(g_clu_type[1])
            span_b_str = ''.join(sentence[span_b[0]:span_b[1] + 1])
            span_b_length = span_b[1] - span_b[0] + 1

            if len(span_b_str) != span_b_length:
                print("sentence ", key, " ", sentence)
                print("span b had more token:", span_b)

            b_str_cut_list = SnowNLP(span_b_str).words
            if len(b_str_cut_list) > 1 and span_b_length > 2:
                final_start, j = 0, 0
                while j < len(b_str_cut_list) - 1:
                    final_start += len(b_str_cut_list[j])
                    j += 1
                b_new_start = span_b[0] + final_start
                if b_new_start <= span_b[1]:
                    new_span_b = [b_new_start, span_b[1]]
                else:
                    print("sentence:", key)
                    print("span b wrong!", span_b)
                    print("span b cut:", b_str_cut_list)
            else:
                new_span_b = span_b    # head span b    
            new_span_b = tuple(new_span_b)
            
            t_clu_type = 0
            if len(g_clu_type) == 3:
                clu_type = g_clu_type[2]
                if clu_type == "名词隐喻":
                    noun_gold_meta_num += 1
                    t_clu_type = 1
                elif clu_type == "明喻":
                    simile_gold_meta_num += 1
                    t_clu_type = 2
                elif clu_type == "动词隐喻":
                    v_gold_meta_num += 1
                    t_clu_type = 3
                elif clu_type == "形容词隐喻":
                    adj_gold_meta_num += 1
                    t_clu_type = 4   
                g_clu_flag = 0
                if tuple(g_clu_type[0]) in pre_mention_2_clu:
                    predict_clu = pre_mention_2_clu[tuple(g_clu_type[0])]  # 找出gold span所在预测链
                    g_span_a_type = gold_mention_2_type[tuple(g_clu_type[0])]
                    g_span_b_type = gold_mention_2_type[tuple(g_clu_type[1])]
                    p_span_a_type = predict_span_2_type[tuple(g_clu_type[0])]
                    if tuple(g_clu_type[1]) in predict_clu:
                        p_span_b_type = predict_span_2_type[tuple(g_clu_type[1])]
                        g_clu_flag = 1
                    elif len(match_head(new_span_b, predict_spans)) > 0:
                        p_right_span = match_head(new_span_b, predict_spans)
                        p_span_b_type = predict_span_2_type[p_right_span]
                        g_clu_flag = 1
                        
                elif len(match_head(new_span_a, predict_spans)) > 0:
                    p_right_span = match_head(new_span_a, predict_spans)
                    predict_clu = pre_mention_2_clu[p_right_span]
                    g_clu_flag = 1
                    g_span_a_type = gold_mention_2_type[tuple(g_clu_type[0])]
                    g_span_b_type = gold_mention_2_type[tuple(g_clu_type[1])]
                    p_span_a_type = predict_span_2_type[p_right_span]
                    if tuple(g_clu_type[1]) in predict_clu:
                        p_span_b_type = predict_span_2_type[tuple(g_clu_type[1])]
                        g_clu_flag = 1
                    elif len(match_head(new_span_b, predict_spans)):
                        p_right_span = match_head(new_span_b, predict_spans)
                        p_span_b_type = predict_span_2_type[p_right_span]
                        g_clu_flag = 1
                    
                if g_clu_flag == 1:
                    if g_span_a_type == p_span_a_type and g_span_b_type == p_span_b_type:                       
                        if t_clu_type == 1:  # 判断当前链是什么类型
                            noun_pre_meta_num += 1
                        elif t_clu_type == 2:
                            simile_pre_meta_num += 1
                        elif t_clu_type == 3:
                            v_pre_meta_num += 1
                        elif t_clu_type == 4:
                            adj_pre_meta_num += 1


    print("noun_gold_meta_num", noun_gold_meta_num, " noun_pre_meta_num:", noun_pre_meta_num, " noun_all_meta_num:",
          noun_all_meta_num)
    print("simile_gold_meta_num", simile_gold_meta_num, " simile_pre_meta_num:", simile_pre_meta_num,
          " simile_all_meta_num:", simile_all_meta_num)
    print("v_gold_meta_num", v_gold_meta_num, " v_pre_meta_num:", v_pre_meta_num, " v_all_meta_num:", v_all_meta_num)
    print("adj_gold_meta_num", adj_gold_meta_num, " adj_pre_meta_num:", adj_pre_meta_num, "adj_all_meta_num",
          adj_all_meta_num, "\n")

    # print("other noun num:", other_noun_num, " other meta num:", other_meta_num)
    # print("other noun meta length:", len(other_noun_meta), " other noun meta are:\n", other_noun_meta)

    noun_recall = 0 if noun_gold_meta_num == 0 else noun_pre_meta_num / noun_gold_meta_num
    noun_precision = 0 if noun_all_meta_num == 0 else noun_pre_meta_num / noun_all_meta_num
    f_noun = 0 if (noun_recall + noun_precision) == 0 else 2 * noun_recall * noun_precision / (noun_recall + noun_precision)

    simile_recall = 0 if simile_gold_meta_num == 0 else simile_pre_meta_num / simile_gold_meta_num
    simile_precision = 0 if simile_all_meta_num == 0 else simile_pre_meta_num / simile_all_meta_num
    f_simile = 0 if simile_recall + simile_precision == 0 else 2 * simile_recall * simile_precision / (
                simile_recall + simile_precision)

    v_recall = 0 if v_gold_meta_num == 0 else v_pre_meta_num / v_gold_meta_num
    v_precision = 0 if v_all_meta_num == 0 else v_pre_meta_num / v_all_meta_num
    f_v = 0 if v_recall + v_precision == 0 else 2 * v_recall * v_precision / (v_recall + v_precision)

    adj_recall = 0 if adj_gold_meta_num == 0 else adj_pre_meta_num / adj_gold_meta_num
    adj_precision = 0 if adj_all_meta_num == 0 else adj_pre_meta_num / adj_all_meta_num
    f_adj = 0 if adj_recall + adj_precision == 0 else 2 * adj_recall * adj_precision / (adj_recall + adj_precision)

    print("noun_recall, precision and f1:", noun_recall, " ", noun_precision, " ", f_noun)
    print("simile_recall, precision and f1:", simile_recall, " ", simile_precision, " ", f_simile)
    print("v_recall, precision and f1:", v_recall, " ", v_precision, " ", f_v)
    print("adj_recall, precision and f1:", adj_recall, " ", adj_precision, " ", f_adj)

# ---------test data
def span_pair_metaphor_link_test(predict_clusters, top_span_starts, top_span_ends, all_span_pair_types,
                                 top_antecedents):
    """
    对每句话的预测结果得到预测链和gold cluster 中span type结果
    """
    # print("predict_clusters:", predict_clusters)
    predict_span_2_span_dict = {}
    for clu in predict_clusters:
        span_a = clu[0]
        span_b = clu[1]

        if span_a not in predict_span_2_span_dict:
            predict_span_2_span_dict[span_a] = [span_b]
        else:
            predict_span_2_span_dict[span_a].append(span_b)
        if span_b not in predict_span_2_span_dict:
            predict_span_2_span_dict[span_b] = [span_a]
        else:
            predict_span_2_span_dict[span_b].append(span_a)

    predict_span_2_type = {}
    for i, start in enumerate(top_span_starts):
        end = top_span_ends[i]
        top_span = (int(start), int(end))
        span_pair_types = all_span_pair_types[i]
        ants = top_antecedents[i]
        for j, ant in enumerate(ants):
            ant_span = (int(top_span_starts[ant]), int(top_span_ends[ant]))
            max_type_value, top_span_type = torch.max(span_pair_types[j][:5], 0)  # 返回下标
            max_ant_type_value, ant_span_type = torch.max(span_pair_types[j][5:], 0)
            top_span_type = int(top_span_type)
            ant_span_type = int(ant_span_type)
            if top_span in predict_span_2_span_dict:
                predict_ant_list = predict_span_2_span_dict[top_span]
                if ant_span in predict_ant_list:
                    # predict cluster一定会被包含其中
                    # if top_span in predict_span_2_type:
                    #     print("predict span before type:", predict_span_2_type[top_span], " now type:", top_span_type)
                    predict_span_2_type[top_span] = [top_span_type, max_type_value]
                    predict_span_2_type[ant_span] = [ant_span_type, max_ant_type_value]

    # print("predict_span_2_type:", predict_span_2_type)

    return predict_span_2_type

def get_metaphor_link_type(sentence, span_a, span_a_type, span_b, span_b_type):
    """
    根据两个span及其type判断这个隐喻链的类型
    :return: metaphor type
    """
    sen_str = ''.join(sentence[1:-1])
    relation_type = get_meta_type(span_a_type, span_b_type)
    xiang_sen = []
    if relation_type == "动词隐喻":
        return "动词隐喻"
    elif relation_type == "形容词隐喻":
        return "形容词隐喻"
    else:  # 名词隐喻、其他隐喻  根据这两个span之间的字符串内容来判断

        if span_a[1] < span_b[0]:
            mid_str = sentence[span_a[1] + 1:span_b[0]]
        elif span_b[1] < span_a[0]:
            mid_str = sentence[span_b[1] + 1:span_a[0]]
        else:  # 嵌套关系
            mid_str = ''
            # print("other mid str for span a:", span_a, " span b:", span_b)
        # if "像" in mid_str or "如" in mid_str:
        if ("像" in sentence or "如" in sentence or "仿佛" in sen_str) and "如果" not in sen_str:
            if sentence not in xiang_sen:
                xiang_sen.append(sentence)
                return "明喻"
                # print("像 sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
        elif "是" in mid_str:
            return "AisB型隐喻"
            # print("\nmid str:", mid_str, " span a:", span_a_str, "span b:", span_b_str)
            # print("是 sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
        elif ("的" in mid_str or "之" in mid_str) and len(mid_str) == 1:
            return "形容词隐喻"
            # print("的 sentence：", " i:", key, "--", ''.join(sentence[1:-1]))
        else:
            return "普通隐喻"

def eval_span_type_predict(doc_gold_clusters, doc_gold_spans_type, doc_gold_span_2_p_type):
    # 针对只训练了span分类器的评价函数
    all_gold_span_num, all_right_span_num = 0, 0
    for example_num, gold_clusters in doc_gold_clusters.items():
        gold_span_types = doc_gold_spans_type[example_num]
        gold_span_2_p_type = doc_gold_span_2_p_type[example_num]

        for span, g_type in gold_span_types.items():
            if g_type == gold_span_2_p_type[span]:
                all_right_span_num += 1
            all_gold_span_num += 1

    print("all_right_span_num:", all_right_span_num, " all_gold_span_num:", all_gold_span_num)
    recall = all_right_span_num / all_gold_span_num

    return recall