#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import logging
import random
import torch
import torch.optim as optim
from tqdm import tqdm, trange
import numpy as np

from bert.tokenization import BertTokenizer
import utils
from metaphor import CorefModel
import metrics
from metrics import Meta_evaluete_all_doc, Meta_evaluete_all_doc_fuzzy, Meta_span_type_result, \
    different_metaphor_result, compute_v_meta, get_metaphor_mention_prf, get_meta_fuzzy_prf, span_pair_metaphor_link, \
    Meta_span_pair_type_result, different_span_pair_metaphor_result, span_pair_metaphor_link_test, get_metaphor_link_type, get_metaphor_mention_prf_type, Meta_evaluete_all_doc_type, get_metaphor_mention_prf_one_type, Meta_evaluete_all_doc_one_type, get_meta_fuzzy_prf_type, get_metaphor_mention_prf_type_fuzzy, Meta_evaluete_all_doc_fuzzy_one_type, Meta_evaluete_all_doc_type_fuzzy

from datetime import datetime

# from TFIDF import Collection
import argparse

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from transformers import RobertaTokenizer, RobertaModel, BertConfig
from sklearn.metrics import classification_report


def train_coref(config):
    """
    指代消解模型训练
    :param config: 配置参数
    :return: None
    """
    model = CorefModel.from_pretrained(config["pretrained_model"], coref_task_config=config)
    # print("model:\n", model)
    model.to(device)

    examples = model.get_train_example()
    train_steps = config["num_epochs"] * config["num_docs"]

    param_optimizer = list(model.named_parameters())
    print("需要学习的参数：{}".format(len(param_optimizer)))

    bert_params = list(map(id, model.bert.parameters()))
    span_type_net_params = list(map(id, model.span_types_net.parameters()))
    task_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    # task_params = filter(lambda p: id(p) not in bert_params and id(p) not in span_type_net_params, model.parameters())

    # 优化器
    optimizer = optim.Adam([
        {'params': task_params},
        {'params': model.bert.parameters(), 'lr': config['bert_learning_rate']}],
        # {'params': model.span_types_net.parameters(), 'lr': config['span_type_learning_rate']}],
        lr=config['task_learning_rate'],
        eps=config['adam_eps'])
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=int(train_steps * 0.1))

    logger.info("********** Running training ****************")
    logger.info("  Num train examples = %d", len(examples))
    logger.info("  Num epoch = %d", config["num_epochs"])
    logger.info("  Num train step = %d", train_steps)

    fh = logging.FileHandler(os.path.join(config["data_dir"], 'train.log'), mode="w")
    fh.setFormatter(logging.Formatter(format))
    logger.addHandler(fh)

    model.train()
    global_step = 0
    start_time = time.time()
    accumulated_loss = 0.0

    epoch_f1_eval = {}  # 记录epoch的f1值
    best_f1 = -1
    num_eva, epoch_i = 0, 0
    for _ in trange(int(config["num_epochs"]), desc="Epoch"):
        random.shuffle(examples)
        for step, example in enumerate(tqdm(examples, desc="Train_Examples")):
            tensorized_example = model.tensorize_example2(example, is_training=True)

            input_ids = torch.from_numpy(tensorized_example[0]).long().to(device)
            input_mask = torch.from_numpy(tensorized_example[1]).long().to(device)
            text_len = torch.from_numpy(tensorized_example[2]).long().to(device)
            # speaker_ids = torch.from_numpy(tensorized_example[3]).long().to(device)
            # genre = torch.tensor(tensorized_example[4]).long().to(device)
            speaker_ids = torch.Tensor(0).long().to(device)
            genre = torch.Tensor(0).long().to(device)
            is_training = tensorized_example[3]
            gold_starts = torch.from_numpy(tensorized_example[4]).long().to(device)
            gold_ends = torch.from_numpy(tensorized_example[5]).long().to(device)
            cluster_ids = torch.from_numpy(tensorized_example[6]).long().to(device)
            sentence_map = torch.Tensor(tensorized_example[7]).long().to(device)
            # sentences = [word for sen in tensorized_example[10] for word in sen]   # 截断后的sentences
            sentences = tensorized_example[8]

            # gold_mention_types = torch.from_numpy(tensorized_example[10]).long().to(device)
            gold_mention_types = example["spans_type"]
            gold_clusters = example["clusters"]
            write_epoch = epoch_i

            predictions, loss = model(sentences, input_ids, input_mask, text_len, speaker_ids, genre,
                                                 is_training,
                                                 gold_starts, gold_ends, cluster_ids, sentence_map, 
                                                 gold_mention_types, gold_clusters)

            accumulated_loss += loss.item()
            if global_step % report_frequency == 0:
                total_time = time.time() - start_time
                steps_per_second = global_step / total_time
                average_loss = accumulated_loss / report_frequency
                print("\n")
                logger.info("step:{} | loss: {:.4f} | step/s: {:.4f}".format(global_step, average_loss, steps_per_second))
                accumulated_loss = 0.0

            # 验证集验证
            if global_step % eval_frequency == 0 and global_step != 0:
                if global_step >= 1 * config["eval_frequency"]:
                    model_save_path = os.path.join(config["model_save_path"],
                                                   "{}".format(num_eva + 1))
                else:
                    model_save_path = config["model_save_path"]
                utils.save_model(model, model_save_path)
                num_eva += 1
                torch.cuda.empty_cache()
                eval_model = CorefModel.from_pretrained(model_save_path, coref_task_config=config)
                eval_model.to(device)
                eval_model.eval()
                try:
                    epoch_f = eval_model.evaluate_metaphor(eval_model, device, eval_mode=True)
                    epoch_f1_eval[num_eva] = epoch_f  # 记录每个epoch的f值
                    print("epoch ", num_eva, " f1:", epoch_f)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception
                except AttributeError as exception:
                    print("Found too many repeated mentions (> 10) in the response, so refusing to score")

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            global_step += 1


        scheduler.step()
        epoch_i += 1

    utils.save_model(model, config["model_save_path"])
    print("*****************************训练完成，已保存模型****************************************")
    torch.cuda.empty_cache()

    print("epoch_f1_eval :")
    for key, value in epoch_f1_eval.items():
        print("epoch ", key, " f1: ", epoch_f1_eval[key])


def eval_coref(config):
    """
    metaphor evaluation : independent span type evaluation
    """
    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    examples = model.get_eval_example()

    logger.info("********** Running Eval ****************")
    logger.info("  Num dev examples = %d", len(examples))

    model.eval()
    meta_predictions = {}
    doc_gold_clusters, doc_mention_to_gold = {}, {}
    meta_top_spans, meta_span_types, doc_gold_spans_type = {}, {}, {}
    meta_gold, doc_gold_clusters_type, doc_sentences = {}, {}, {}
    label_predicts, label_true = [], []
    meta_num, all_pre_clusters_num = 0, 0
    doc_meta_v_span, doc_unmeta_v_span = {}, {}
    with torch.no_grad():
        random.shuffle(examples)
        for example_num, example in enumerate(tqdm(examples, desc="Eval_Examples")):

            tensorized_example = model.tensorize_example2(example, is_training=False)
            input_ids = torch.from_numpy(tensorized_example[0]).long().to(device)
            input_mask = torch.from_numpy(tensorized_example[1]).long().to(device)
            text_len = torch.from_numpy(tensorized_example[2]).long().to(device)
            # speaker_ids = torch.from_numpy(tensorized_example[3]).long().to(device)
            # genre = torch.tensor(tensorized_example[4]).long().to(device)
            speaker_ids = torch.Tensor(0).long().to(device)
            genre = torch.Tensor(0).long().to(device)
            is_training = tensorized_example[3]
            gold_starts = torch.from_numpy(tensorized_example[4]).long().to(device)
            gold_ends = torch.from_numpy(tensorized_example[5]).long().to(device)
            cluster_ids = torch.from_numpy(tensorized_example[6]).long().to(device)
            sentence_map = torch.Tensor(tensorized_example[7]).long().to(device)
            sentences = tensorized_example[8]

            gold_mention_types = example["spans_type"]
            gold_clusters = example["clusters"]
            
            # doc_meta_v_span[example_num] = example["meta_v_span"]
            # doc_unmeta_v_span[example_num] = example["unmeta_v_span"]

            # gold_clusters_type = example["clusters_type"]    # 对不同的隐喻链做评估
            # doc_gold_clusters_type[example_num] = gold_clusters_type
            
            doc_sentences[example_num] = sentences[0]

            label_true.append(int(example["label"]))

            (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
             top_antecedents, top_antecedent_scores, top_span_mention_scores, top_span_types), loss = model(sentences,
                                                                                                            input_ids,
                                                                                                            input_mask,
                                                                                                            text_len,
                                                                                                            speaker_ids,
                                                                                                            genre,
                                                                                                            is_training,
                                                                                                            gold_starts,
                                                                                                            gold_ends,
                                                                                                            cluster_ids,
                                                                                                            sentence_map,
                                                                                                          
                                                                                                            gold_mention_types,
                                                                                                            gold_clusters)
            # predicted_antecedents = model.get_predicted_antecedents(top_antecedents.cpu(), top_antecedent_scores.cpu())
            predicted_antecedents = model.get_metaphor_ants_2(top_antecedents.cpu(), top_span_starts.cpu(), top_span_ends.cpu(), 
                                                            top_antecedent_scores.cpu())
            
            predicted_clusters, doc_gold_clusters[example_num], doc_mention_to_gold[example_num] \
                = model.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"])
            metaphor_link_score = model.get_metaphor_link_score(top_span_starts, top_span_ends, top_antecedents.cpu(),
                                                                top_antecedent_scores.cpu())
            del_overlap_clusters = model.delete_overlapped_clusters(predicted_clusters, metaphor_link_score)
            
            meta_predictions[example_num] = del_overlap_clusters
            
            
            all_pre_clusters_num += len(meta_predictions[example_num])

            if len(gold_starts) > 0:
                meta_num += 1

            meta_top_spans[example_num] = []
            for k, start in enumerate(top_span_starts):
                end = top_span_ends[k]
                meta_top_spans[example_num].append((int(start), int(end)))

            # meta_span_types[example_num] = top_span_types

            gold_span_type = []
            for span_and_type in example["spans_type"]:
                one_span_type = [span_and_type[0]]
                type = model.get_span_type_value(span_and_type[1])
                one_span_type.append(type)
                gold_span_type.append(one_span_type)
            doc_gold_spans_type[example_num] = gold_span_type

            meta_gold[example_num] = example["clusters"]  # 答案

            if len(meta_predictions[example_num]) > 0:  # 预测结果
                # print("the predicted clusters:", meta_predictions[example_num])
                label_predicts.append(1)  # 2
            else:
                label_predicts.append(0)
    print("--------------------------the metaphor mention result:")
    fuzzy_ratio = 0.5
    p_men, r_men, f_men = get_metaphor_mention_prf(meta_predictions, meta_gold)
    print("precision:", round(p_men * 100, 5), " recall:", round(r_men * 100, 5), " f1:", round(f_men * 100, 5))

    fuzzy_p_men, fuzzy_r_men, fuzzy_f_men = get_meta_fuzzy_prf(meta_predictions, meta_gold, fuzzy_ratio)
    print("\nthe  fuzzy metaphor mention result:")
    print("precision:", round(fuzzy_p_men * 100, 5), " recall:", round(fuzzy_r_men * 100, 5), " f1:", round(fuzzy_f_men * 100, 5))
    
    # p_men_type, r_men_type, f_men_type = get_metaphor_mention_prf_one_type(meta_predictions, meta_gold, doc_gold_spans_type, meta_span_types, meta_top_spans)
    # print("see one type classify mention result:")
    # print("precision:", round(p_men_type * 100, 5), " recall:", round(r_men_type * 100, 5), " f1:", round(f_men_type * 100, 5))

    # p_men_type_fuzzy, r_men_type_fuzzy, f_men_type_fuzzy = get_meta_fuzzy_prf_type(meta_predictions, meta_gold, fuzzy_ratio, doc_gold_spans_type, meta_span_types, meta_top_spans)
    # print("see one type classify mention fuzzy result:")
    # print("precision:", round(p_men_type_fuzzy * 100, 5), " recall:", round(r_men_type_fuzzy * 100, 5), " f1:", round(f_men_type_fuzzy * 100, 5))

    print("\n------------------------the metaphor links results:")

    # print("len(meta_predictions):", len(meta_predictions))
    print("all predict clusters num:", all_pre_clusters_num)
    p_1, r_1, f_1 = Meta_evaluete_all_doc(meta_predictions, doc_gold_clusters, doc_mention_to_gold)
    print("all doc to compute meta links result:", ' p:', round(p_1, 5), " r:", round(r_1, 5), " f:", round(f_1, 5))

    p_2, r_2, f_2 = Meta_evaluete_all_doc_fuzzy(meta_predictions, doc_gold_clusters, doc_mention_to_gold)
    print("\nall doc to compute meta links result fuzzy:", ' p:', round(p_2, 5), " r:", round(r_2, 5), " f:", round(f_2, 5))
    
    # add one span type classify
    # p_3, r_3, f_3 = Meta_evaluete_all_doc_one_type(meta_predictions, doc_gold_clusters, doc_mention_to_gold, doc_gold_spans_type, meta_span_types, meta_top_spans)
    # print("see one type meta links result:", ' p:', round(p_3, 5), " r:", round(r_3, 5), " f:", round(f_3, 5))
    
    # add one span type fuzzy result link
    # p_4, r_4, f_4 = Meta_evaluete_all_doc_fuzzy_one_type(meta_predictions, doc_gold_clusters, doc_mention_to_gold, doc_gold_spans_type, meta_span_types, meta_top_spans)
    # print("see one type meta links fuzzy result:", ' p:', round(p_4, 5), " r:", round(r_4, 5), " f:", round(f_4, 5))

    # print("\n-------------------------the span type prediction result:")
    # p_type, r_type, f_type = Meta_span_type_result(meta_top_spans, meta_span_types, meta_predictions,
    #                                                doc_gold_spans_type, meta_gold, doc_meta_v_span, doc_unmeta_v_span)
    # print("span type p, r, f:", p_type, "  ", r_type, "  ", f_type)

    print("\n句子的隐喻性识别结果为：\n", classification_report(label_true, label_predicts, digits=4))
    
    # print("\n-------------different metaphor result----------------")
    # different_metaphor_result(meta_predictions, doc_gold_clusters_type, meta_top_spans, meta_span_types, doc_sentences, doc_gold_spans_type)


def eval_coref_with_type(config):
    """
    隐喻识别模型验证
    :param config: 配置参数
    :return: None
    """
    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    examples = model.get_eval_example()

    logger.info("********** Running Eval ****************")
    logger.info("  Num dev examples = %d", len(examples))

    model.eval()
    meta_predictions = {}
    doc_gold_clusters, doc_mention_to_gold = {}, {}
    meta_top_spans, meta_span_types, doc_gold_spans_type = {}, {}, {}
    meta_gold, doc_gold_clusters_type, doc_sentences = {}, {}, {}
    doc_predict_gold_span_2_type = {}
    label_predicts, label_true = [], []
    meta_num, all_pre_clusters_num = 0, 0
    doc_meta_v_span, doc_unmeta_v_span = {}, {}
    with torch.no_grad():
        # random.shuffle(examples)
        for example_num, example in enumerate(tqdm(examples, desc="Eval_Examples")):
            # print("example_num:", example_num)
            tensorized_example = model.tensorize_example2(example, is_training=False)

            input_ids = torch.from_numpy(tensorized_example[0]).long().to(device)
            input_mask = torch.from_numpy(tensorized_example[1]).long().to(device)
            text_len = torch.from_numpy(tensorized_example[2]).long().to(device)
            # speaker_ids = torch.from_numpy(tensorized_example[3]).long().to(device)
            # genre = torch.tensor(tensorized_example[4]).long().to(device)
            speaker_ids = torch.Tensor(0).long().to(device)
            genre = torch.Tensor(0).long().to(device)
            is_training = tensorized_example[3]
            gold_starts = torch.from_numpy(tensorized_example[4]).long().to(device)
            gold_ends = torch.from_numpy(tensorized_example[5]).long().to(device)
            cluster_ids = torch.from_numpy(tensorized_example[6]).long().to(device)
            sentence_map = torch.Tensor(tensorized_example[7]).long().to(device)
            sentences = tensorized_example[8]

            gold_mention_types = example["spans_type"]
            gold_clusters = example["clusters"]

#            doc_meta_v_span[example_num] = example["meta_v_span"]
#            doc_unmeta_v_span[example_num] = example["unmeta_v_span"]

            gold_clusters_type = example["clusters_type"]    # 对不同的隐喻链做评估
            doc_gold_clusters_type[example_num] = gold_clusters_type
            doc_sentences[example_num] = sentences[0]

            label_true.append(int(example["label"]))
            
            (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
             top_antecedents, top_antecedent_scores, top_span_mention_scores, top_span_pair_types), loss = model(sentences,
                                                                                                            input_ids,
                                                                                                            input_mask,
                                                                                                            text_len,
                                                                                                            speaker_ids,
                                                                                                            genre,
                                                                                                            is_training,
                                                                                                            gold_starts,
                                                                                                            gold_ends,
                                                                                                            cluster_ids,
                                                                                                            sentence_map,
                                                                                                            gold_mention_types,
                                                                                                            gold_clusters)

            predicted_antecedents = model.get_metaphor_ants_2(top_antecedents.cpu(), top_span_starts.cpu(), top_span_ends.cpu(), 
                                                            top_antecedent_scores.cpu())

            predicted_clusters, doc_gold_clusters[example_num], doc_mention_to_gold[example_num] \
                = model.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"])

            metaphor_link_score = model.get_metaphor_link_score(top_span_starts, top_span_ends, top_antecedents.cpu(),
                                                                top_antecedent_scores.cpu())
            del_overlap_clusters = model.delete_overlapped_clusters(predicted_clusters, metaphor_link_score)
            meta_predictions[example_num] = del_overlap_clusters
            
            all_pre_clusters_num += len(meta_predictions[example_num])

            if len(gold_starts) > 0:
                meta_num += 1

            meta_top_spans[example_num] = []
            for k, start in enumerate(top_span_starts):
                end = top_span_ends[k]
                meta_top_spans[example_num].append((int(start), int(end)))

            predict_span_2_type, gold_span_2_type = span_pair_metaphor_link(meta_predictions[example_num],
                                                                            doc_gold_clusters[example_num],
                                                                            top_span_starts, top_span_ends,
                                                                            top_span_pair_types, top_antecedents)

            meta_span_types[example_num] = predict_span_2_type  # predict span to type top_span_types
            doc_predict_gold_span_2_type[example_num] = gold_span_2_type    # gold span to predict type

            gold_span_type = []
            for span_and_type in example["spans_type"]:
                one_span_type = [span_and_type[0]]
                type = model.get_span_type_value(span_and_type[1])   # type id
                one_span_type.append(type)
                gold_span_type.append(one_span_type)

            doc_gold_spans_type[example_num] = gold_span_type

            meta_gold[example_num] = example["clusters"]

            if len(meta_predictions[example_num]) > 0:  # 预测结果
                # print("the predicted clusters:", meta_predictions[example_num])
                label_predicts.append(1)  # 2
            else:
                label_predicts.append(0)
    print("--------------------------the metaphor mention result:")
    fuzzy_ratio = 0.5
    p_men, r_men, f_men = get_metaphor_mention_prf(meta_predictions, meta_gold)
    print("precision:", round(p_men * 100, 5), " recall:", round(r_men * 100, 5), " f1:", round(f_men * 100, 5))
    
    fuzzy_p_men, fuzzy_r_men, fuzzy_f_men = get_meta_fuzzy_prf(meta_predictions, meta_gold, fuzzy_ratio)
    print("the  fuzzy metaphor mention result:")
    print("precision:", round(fuzzy_p_men * 100, 5), " recall:", round(fuzzy_r_men * 100, 5), " f1:", round(fuzzy_f_men * 100, 5))
    
    # p_men_type, r_men_type, f_men_type = get_metaphor_mention_prf_type(meta_predictions, meta_gold, doc_gold_spans_type, meta_span_types)
    # print("add type angle of mention result:")
    # print("precision:", round(p_men_type * 100, 5), " recall:", round(r_men_type * 100, 5), " f1:", round(f_men_type * 100, 5))
    #
    # p_men_type_fuzzy, r_men_type_fuzzy, f_men_type_fuzzy = get_metaphor_mention_prf_type_fuzzy(meta_predictions, meta_gold, fuzzy_ratio, doc_gold_spans_type, meta_span_types)
    # print("add type angle of mention fuzzy result:")
    # print("precision:", round(p_men_type_fuzzy * 100, 5), " recall:", round(r_men_type_fuzzy * 100, 5), " f1:", round(f_men_type_fuzzy * 100, 5))

    print("\n------------------------the metaphor links results:")
    # print("len(meta_predictions):", len(meta_predictions))
    print("all predict clusters num:", all_pre_clusters_num)
    p_1, r_1, f_1 = Meta_evaluete_all_doc(meta_predictions, doc_gold_clusters, doc_mention_to_gold)
    print("all doc to compute meta links result:", ' p:', round(p_1, 5), " r:", round(r_1, 5), " f:", round(f_1, 5))
    # print("\n")
    p_2, r_2, f_2 = Meta_evaluete_all_doc_fuzzy(meta_predictions, doc_gold_clusters, doc_mention_to_gold)
    print("all doc to compute meta links result fuzzy:", ' p:', round(p_2, 5), " r:", round(r_2, 5),
          " f:", round(f_2, 5))
    
    # add type angle
    # p_3, r_3, f_3 = Meta_evaluete_all_doc_type(meta_predictions, doc_gold_clusters, doc_mention_to_gold, doc_gold_spans_type, meta_span_types, doc_sentences)
    # print("add type angle of the metaphor link result:", " p:", round(p_3, 5), " r:", round(r_3, 5), " f:", round(f_3, 5))
    #
    # p_4, r_4, f_4 = Meta_evaluete_all_doc_type_fuzzy(meta_predictions, doc_gold_clusters, doc_mention_to_gold, doc_gold_spans_type, meta_span_types)
    # print("add type angle of the metaphor link  fuzzy result:", " p:", round(p_4, 5), " r:", round(r_4, 5), " f:", round(f_4, 5))

#    print("\n-------------------------the span type prediction result:")
#    p_type, r_type, f_type = Meta_span_type_result(meta_top_spans, meta_span_types, meta_predictions,
#                                                   doc_gold_spans_type, meta_gold)
#    print("\n-------------------------the span type prediction result:")
#    p_type, r_type, f_type = Meta_span_type_result(meta_top_spans, meta_span_types, meta_predictions,
#                                                   doc_gold_spans_type, meta_gold, doc_meta_v_span, doc_unmeta_v_span)                                               

#    p_type, r_type, f_type = Meta_span_pair_type_result(meta_span_types, meta_predictions,
#                                                        doc_gold_spans_type, doc_gold_clusters, doc_sentences,
#                                                        doc_meta_v_span, doc_unmeta_v_span)
#    print("span type p, r, f:", round(p_type, 5), "  ", round(r_type, 5), "  ", round(f_type, 5))

#    print("句子的隐喻性识别结果为：\n", classification_report(label_true, label_predicts, digits=4))

    print("-------------different metaphor result----------------")
    # different_metaphor_result(meta_predictions, doc_gold_clusters_type, meta_top_spans, meta_span_types, doc_sentences)
    different_span_pair_metaphor_result(meta_predictions, doc_gold_clusters_type, doc_sentences, meta_span_types,
                                        label_true, doc_gold_spans_type)

         
def test_metaphor(config):
    """
    只训练了隐喻关系，为pipeline方法保存隐喻链
    :param config: 配置参数
    :return: None
    """
    model = CorefModel.from_pretrained(config["model_save_path"], coref_task_config=config)
    model.to(device)

    output_filename = config["test_output_path"]
    examples = model.get_test_example()

    logger.info("********** Running Test ****************")
    logger.info("  Num test examples = %d", len(examples))

    model.eval()
    with open(output_filename, 'w', encoding="utf-8") as output_file:
        with torch.no_grad():
            for example_num, example in enumerate(tqdm(examples, desc="Test_Examples")):
                tensorized_example = model.tensorize_example2(example, is_training=False)

                all_span_salience = None
                # sentences = [word for sen in example["sentences"] for word in sen]
                input_ids = torch.from_numpy(tensorized_example[0]).long().to(device)
                input_mask = torch.from_numpy(tensorized_example[1]).long().to(device)
                text_len = torch.from_numpy(tensorized_example[2]).long().to(device)
                # speaker_ids = torch.from_numpy(tensorized_example[3]).long().to(device)
                # genre = torch.tensor(tensorized_example[4]).long().to(device)
                speaker_ids = torch.Tensor(0).long().to(device)
                genre = torch.Tensor(0).long().to(device)
                is_training = tensorized_example[3]
                gold_starts = torch.from_numpy(tensorized_example[4]).long().to(device)
                gold_ends = torch.from_numpy(tensorized_example[5]).long().to(device)
                cluster_ids = torch.from_numpy(tensorized_example[6]).long().to(device)
                sentence_map = torch.Tensor(tensorized_example[7]).long().to(device)

                sentences = tensorized_example[8]

                gold_mention_types = example["spans_type"]
                
                gold_clusters = example["clusters"]
                gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]

                # top_span_mention_scores
                (_, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores,
                 top_span_mention_scores, top_span_pair_types), _ = \
                    model(sentences, input_ids, input_mask, text_len, speaker_ids,
                          genre, is_training, gold_starts, gold_ends, cluster_ids,
                          sentence_map, gold_mention_types, gold_clusters)

                predicted_antecedents = model.get_metaphor_ants_2(top_antecedents.cpu(), top_span_starts.cpu(),
                                                                  top_span_ends.cpu(),
                                                                  top_antecedent_scores.cpu())

                predicted_clusters, _ = model.get_metaphor_clusters_2(top_span_starts, top_span_ends,
                                                                                      predicted_antecedents)
                
                metaphor_link_score = model.get_metaphor_link_score(top_span_starts, top_span_ends, top_antecedents.cpu(),
                                                                top_antecedent_scores.cpu())
                
                del_overlap_clusters = model.delete_overlapped_clusters(predicted_clusters, metaphor_link_score)

                example["predicted_clusters"] = del_overlap_clusters

                # 将句中索引——>文字
                # example_sentence = utils.flatten(example["sentences"])
                # predicted_list = []
                # for same_entity in example["predicted_clusters"]:
                #     same_entity_list = []
                #     num_same_entity = len(same_entity)
                #     for index in range(num_same_entity):
                #         entity_name = ''.join(example_sentence[same_entity[index][0]: same_entity[index][1] + 1])
                #         same_entity_list.append(entity_name)
                #     predicted_list.append(same_entity_list)
                #     same_entity_list = []  # 清空list
                # 
                # example["predicted_idx2entity"] = predicted_list
                # example["top_spans"] = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
                # example['head_scores'] = []
                # 
                # # print(example["sentences"])  把正确的类写出来
                # gold_list = []
                # for same_entity in example["clusters"]:
                #     same_entity_list = []
                #     num_same_entity = len(same_entity)
                #     for index in range(num_same_entity):
                #         entity_name = ''.join(example_sentence[same_entity[index][0]: same_entity[index][1] + 1])
                #         same_entity_list.append(entity_name)
                #     gold_list.append(same_entity_list)
                #     same_entity_list = []

                # print("top span types:", top_span_types)
                # top_spans_type_dict = {}  # top span 对应其预测的span type str
                # for i, span in enumerate(example["top_spans"]):
                #     span_type_str = model.get_type_str(int(top_span_types[i]))
                #     top_spans_type_dict[span] = span_type_str

                # 根据span pair type来整理span type结果
                # predict_span_2_type, gold_span_2_type = span_pair_metaphor_link(example["predicted_clusters"],
                #                                                                 gold_clusters,
                #                                                                 top_span_starts, top_span_ends,
                #                                                                 top_span_pair_types,
                #                                                                 top_antecedents)

                result = example
                result["predicted_clusters"] = example["predicted_clusters"]
                # result["doc_key"] = example["doc_key"]
                # result["sentences"] = example["sentences"]  # 原文
                # result["predicted_clusters"] = example["predicted_clusters"]  # 预测luster
                # result["predicted_idx2entity"] = example["predicted_idx2entity"]
                # result["gold_idx2entity"] = gold_list
                # result["clusters"] = example["clusters"]
                # result["top_spans"] = example["top_spans"]
                # result["sentence_map"] = example["sentence_map"]
                # result["gold_spans_type"] = example["spans_type"]
                # result["predict_spans_type"] = []
                # for clu in example["predicted_clusters"]:
                #     for span in clu:
                #         span_type_int = predict_span_2_type[tuple(span)]
                #         span_type_str = model.get_type_str(int(span_type_int))
                #         result["predict_spans_type"].append([span, span_type_str])

                output_file.write(json.dumps(result, ensure_ascii=False))
                output_file.write("\n")
                if example_num % 100 == 0:
                    print('\n')
                    print("写入 {} examples.".format(example_num + 1))



if __name__ == "__main__":

    os.environ["data_dir"] = "./data"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_experiment = "bert_base_chinese"
    config = utils.read_config(run_experiment, "./src/experiments.conf")
    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # -----

    random.seed(41)              # 原始 41
    np.random.seed(41)
    torch.manual_seed(41)
    torch.cuda.manual_seed_all(41)
    torch.backends.cudnn.deterministic = True

    # 训练阶段
    if config["do_train"]:
        train_coref(config)

    # 验证阶段
    if config["do_eval"]:
        try:
            # eval_coref(config)   # type independent eval
            eval_coref_with_type(config)   # span pair type eval
        except AttributeError as exception:
            print("Found too many repeated mentions (> 10) in the response, so refusing to score")

    # 测试阶段
    if config["do_test"]:
        test_metaphor(config)
