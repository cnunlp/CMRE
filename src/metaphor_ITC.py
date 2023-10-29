# coding=utf-8

from __future__ import absolute_import, division, print_function

import json, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import cmp_to_key
from tqdm import tqdm, trange

# 项目模块
import utils
from utils import to_cuda
import conll
import metrics
# from metrics import Meta_evaluate, Meta_evaluate_fuzzy
from metrics import Meta_evaluete_all_doc, Meta_evaluete_all_doc_fuzzy

from bert import tokenization, modeling
from bert.tokenization import BertTokenizer
from bert.modeling import BertPreTrainedModel
from bert.modeling import BertModel

# from roberta import modeling_roberta
# import modeling_roberta
# from roberta.modeling_roberta import RobertaModel

from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig

import xlwt, csv

class Squeezer(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return torch.squeeze(input, dim=self.dim)

class Score(nn.Module):
    """计算得分"""

    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout=0.3):
        super(Score, self).__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input):
        output = self.score(input)
        return output

class Span_Classifier(nn.Module):
    """span分类器"""
    def __init__(self, input_dim, hidden_dim, num_class, dropout=0.3):
        super(Span_Classifier, self).__init__()
        # 线性变换： 输入层->隐含层
        self.span_classify = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_class)
        )

    def forward(self, inputs):
        outputs = self.span_classify(inputs)
        hidden_emb = self.span_classify[0](inputs)
        # probs = F.softmax(outputs, dim=1)
        # return probs
        return outputs, hidden_emb           #

class CorefModel(BertPreTrainedModel):  # -----------
    # class CorefModel(RobertaModel):
    def __init__(self, config, coref_task_config):
        super(CorefModel, self).__init__(config)

        self.config = coref_task_config
        self.max_segment_len = self.config['max_segment_len']
        self.max_span_width = self.config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(self.config["genres"])}
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None
        self.bert_config = modeling.BertConfig.from_json_file(self.config["bert_config_file"])
        self.tokenizer = BertTokenizer.from_pretrained(self.config['vocab_file'], do_lower_case=True)
        self.bert = BertModel(config=self.bert_config)
        # self.bert = BertModel.from_pretrained(self.config["pretrained_model"])     # self.config["pretrained_model"]
        # self.bert_config = RobertaConfig.from_pretrained(self.config["bert_config_file"])
        # self.bert = RobertaModel(config=self.bert_config)   # roberta model
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        self.emb_dim = self.bert_config.hidden_size * 2 + int(self.config["use_features"]) * 20 + int(
            self.config["model_heads"]) * self.bert_config.hidden_size   #
        self.slow_antecedent_dim = self.emb_dim * 3 + int(self.config["use_metadata"]) * 40 + int(
            self.config["use_features"]) * 20 + int(self.config['use_segment_distance']) * 20 + \
                                   + int(self.config["use_span_type"]) * 200    # *3  + 40
        # _ant
        self.static_emb_dim = self.bert_config.hidden_size + int(self.config["use_features"]) * 20 + int(
            self.config["model_heads"]) * self.bert_config.hidden_size  # static scores dim

        # span 长度 Embedding
        if self.config["use_features"]:
            self.span_width_embedding = nn.Embedding(
                num_embeddings=self.config["max_span_width"],
                embedding_dim=self.config["feature_size"])
        # span head Embedding(ok)
        if self.config["model_heads"]:
            # print("------加入span head 信息------")
            self.masked_mention_score = nn.Sequential(
                nn.Linear(self.bert_config.hidden_size, 1),
                Squeezer(dim=1))

        # 计算指代得分,两层前向神经网络(ok)
        self.mention_scores = Score(self.emb_dim, self.config["ffnn_size"])  # emb_dim = 768 * 3 + 20 = 2324

        # 计算静态的指代得分，也是两层前向神经网络
        self.static_mention_scores = Score(self.static_emb_dim, self.config["ffnn_size"])  # 768 + 20

        # prior_width_embedding
        if self.config['use_prior']:
            self.span_width_prior_embeddings = nn.Embedding(
                num_embeddings=self.config["max_span_width"],
                embedding_dim=self.config["feature_size"])

            # 计算长度得分，两层前向神经网络
            self.width_scores = Score(self.config["feature_size"], self.config["ffnn_size"])

        # 加入salience分数 -----------------
        if self.config['use_salience']:
            self.span_salience_embeddings = nn.Embedding(
                num_embeddings=self.config["num_salience"],
                embedding_dim=self.config["feature_size"])
            self.salience_scores = Score(self.config["feature_size"], self.config["ffnn_size"])

        # doc类别Embedding[7,20]
        self.genres_embedding = nn.Embedding(
            num_embeddings=len(self.genres),
            embedding_dim=self.config["feature_size"])

        # 前c个前指的得分  一个分类器 + dropout
        self.fast_antecedent_scores = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Dropout(self.config["dropout_rate"]))     # nn.Linear(self.emb_dim, self.emb_dim),
        # 前指距离embedding
        if self.config['use_prior']:
            self.antecedent_distance_embedding = nn.Embedding(
                num_embeddings=10,
                embedding_dim=self.config["feature_size"])

            self.antecedent_distance_linear = nn.Linear(self.config["feature_size"], 1)

        if self.config["use_metadata"]:
            # [2,20]
            self.same_speaker_embedding = nn.Embedding(
                num_embeddings=2,
                embedding_dim=self.config["feature_size"])

        if self.config["use_features"]:    # -------_ant
            self.antecedent_offset_embedding = nn.Embedding(
                num_embeddings=10,
                embedding_dim=self.config["feature_size"])

        if self.config['use_segment_distance']:
            self.segment_distance_embedding = nn.Embedding(
                num_embeddings=self.config['max_training_sentences'],
                embedding_dim=self.config["feature_size"])

        # 三维的输入 ffnn 两层前向神经网络
        if self.config['fine_grained']:
            self.slow_antecedent_scores = nn.Sequential(
                nn.Linear(self.slow_antecedent_dim, self.config["ffnn_size"]),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config["dropout_rate"]),
                nn.Linear(self.config["ffnn_size"], 1),
                Squeezer(dim=-1)
            )

            # 分类器 + sigmoid
            self.coref_layer_linear = nn.Sequential(
                nn.Linear(self.emb_dim * 2, self.emb_dim),
                nn.Sigmoid()
            )

        # if self.config["use_span_type"]:
        self.span_types_net = Span_Classifier(self.emb_dim, 1000, 5)      # self.config["feature_size"]  1000

        self.span_type_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=self.config["feature_size"])

        self.apply(self.init_bert_weights)

    def forward(self, sentences, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts,
                gold_ends, cluster_ids, sentence_map, gold_mention_types, gold_clusters):
        # bert_encoder最后一层输出
        emb_mention_doc, _ = self.bert(input_ids=input_ids, attention_mask=input_mask,
                                       output_all_encoded_layers=False)  # [batch_size, seg_len, hidden_size]
        # emb_mention_doc = self.bert(input_ids=input_ids, attention_mask=input_mask).last_hidden_state
        # print("emb_mention_doc:", emb_mention_doc.size())

        mention_doc = self.flatten_emb_by_sentence(emb_mention_doc, input_mask)  # [batch_size*seg_len, hidden_size]
        num_words = torch.tensor(mention_doc.shape[0])  # [batch_size*seg_len]  裁剪以后的句子中实际的总词数
        # 根据最大子串长度，获得候选子串
        flattened_sentence_indices = sentence_map  # num_word
        candidate_starts = torch.arange(num_words).view(-1, 1).repeat(1, self.max_span_width)  # [num_words_len, max_span_width]
        candidate_ends = candidate_starts + torch.arange(self.max_span_width).view(1, -1)  # [num_words_len, max_span_width]

        # 句子开始、结束索引
        candidate_start_sentence_indices = flattened_sentence_indices[candidate_starts]  # [num_words_len, max_span_width]
        candidate_end_sentence_indices = flattened_sentence_indices[torch.clamp(candidate_ends, max=num_words - 1)]  # [num_words_len, max_span_width]

        candidate_mask = to_cuda((candidate_ends < num_words)) & to_cuda(torch.eq(candidate_start_sentence_indices,
                                                                                  candidate_end_sentence_indices))  # [num_words_len, max_span_width]
        flattened_candidate_mask = candidate_mask.view(-1)  # [num_words * max_span_width]

        candidate_starts = candidate_starts.view(-1)[flattened_candidate_mask]  # [num_candidates]
        candidate_ends = candidate_ends.view(-1)[flattened_candidate_mask]  # [num_candidates]
        # 候选簇
        if is_training:
            candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids)  # [num_candidates]
        # 输入的Embedding
        candidate_span_emb = self.get_span_emb(mention_doc, candidate_starts, candidate_ends, static=False)

        # 指代得分
        candidate_mention_scores = self.get_mention_scores(sentences, candidate_span_emb, candidate_starts,
                                                           candidate_ends, is_training)
        candidate_mention_scores = candidate_mention_scores.squeeze(1)  # [num_candidates]

        candidate_mention_types, candidate_mention_type_one, candidata_span_type_embs = self.get_mention_types(candidate_span_emb)  # [num_candidates, class_num]
        # print("candidate mention type hidden emb size:", candidate_span_hidden_emb.size())

        # 裁剪得分低的指代
        max_vlaue = torch.floor(num_words.type(torch.float)) * self.config["top_span_ratio"]
        k = torch.clamp(torch.tensor(3900), max=max_vlaue.int())
        c = torch.clamp(torch.tensor(self.config["max_top_antecedents"]), max=k)   # c = k !!
        
        top_span_indices = self.extract_top_spans(candidate_mention_scores, candidate_starts, candidate_ends, k)
        top_span_indices = top_span_indices.type(torch.int64)

        # 裁剪后的短语开始结束索引，embedding，簇，得分
        top_span_starts = candidate_starts[top_span_indices]  # [k]
        top_span_ends = candidate_ends[top_span_indices]  # [k]
        top_span_emb = candidate_span_emb[top_span_indices]  # [k, emb]
        
        if is_training:
            top_span_cluster_ids = candidate_cluster_ids[top_span_indices]  # [k]

        top_span_types = candidate_mention_types[top_span_indices]      # [k, 5]
        top_span_type_ones = candidate_mention_type_one[top_span_indices]   # [k]
        top_span_mention_scores = candidate_mention_scores[top_span_indices]  # [k]
        top_span_type_embs = candidata_span_type_embs[top_span_indices]

        if self.config['use_metadata']:
            genre_emb = self.genres_embedding(genre)  # [20,]
            speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)
            top_span_speaker_ids = speaker_ids[top_span_starts]
        else:
            genre_emb = None
            top_span_speaker_ids = None

        dummy_scores = to_cuda(torch.zeros(k, 1))  # [k,1]
        # 得到top-c个前指
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = \
            self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
        # print("top_fast_antecedent_scores:", top_fast_antecedent_scores)

        num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
        word_segments = torch.arange(num_segs).view(-1, 1).repeat(1, seg_len)
        bool_inputmask = input_mask == 1
        # bool_inputmask = torch.tensor(bool_inputmask)
        flat_word_segments = word_segments.view(-1)[bool_inputmask.view(-1)]

        mention_segments = flat_word_segments[top_span_starts].view(-1, 1)
        antecedent_segments = flat_word_segments[top_span_starts[top_antecedents]]

        if self.config['use_segment_distance']:
            segment_distance = torch.clamp((mention_segments - antecedent_segments),
                                           0, (self.config['max_training_sentences'] - 1))
        else:
            segment_distance = None

        # 计算top_c个前指得分
        # 二阶
        if self.config['fine_grained']:
            for i in range(self.config["coref_depth"]):
                top_antecedent_emb = top_span_emb[top_antecedents]  # [k,c,emb]

                slow_antecedent_scores = self.get_slow_antecedent_scores(top_span_emb, top_antecedents,
                                                                         top_antecedent_emb, top_antecedent_offsets,
                                                                         top_span_speaker_ids, genre_emb,
                                                                         top_span_types,
                                                                         segment_distance)     # add span type
                top_antecedent_scores = top_fast_antecedent_scores + slow_antecedent_scores                #------降低Sa权重
                # print("slow_antecedent_scores:", slow_antecedent_scores)
                # print("top_antecedent_scores:", top_antecedent_scores)
                top_antecedent_weights = F.softmax(torch.cat((dummy_scores, top_antecedent_scores), dim=1), dim=-1)  # [k, c + 1]

                top_antecedent_emb = torch.cat((top_span_emb.unsqueeze(1), top_antecedent_emb), dim=1)  # [k, c + 1, emb]
                attended_span_emb = torch.sum(top_antecedent_weights.unsqueeze(2) * top_antecedent_emb, 1)  # [k, emb]

                cat_span_emb = torch.cat((top_span_emb, attended_span_emb), dim=1)

                f = self.coref_layer_linear(cat_span_emb)
                top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]

        # 一阶
        else:
            top_antecedent_scores = top_fast_antecedent_scores

        top_antecedent_scores = torch.cat((dummy_scores, top_antecedent_scores), dim=1)  # [k, c + 1]
        # print("top_antecedent_scores:", top_antecedent_scores)
        # 非训练阶段省去计算loss
        if not is_training:
            loss = torch.tensor(0)
            return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                    top_antecedents, top_antecedent_scores, top_span_mention_scores, top_span_type_ones], loss

        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents]  # [k, c]
        top_antecedent_cluster_ids += to_cuda(torch.log(top_antecedents_mask.float()))  # [k, c]

        same_cluster_indicator = torch.eq(top_antecedent_cluster_ids, top_span_cluster_ids.view(-1, 1))
        non_dummy_indicator = (top_span_cluster_ids > 0).view(-1, 1)  # [k, 1]
        pairwise_labels = same_cluster_indicator & non_dummy_indicator  # [k, c]

        dummy_labels = ~ (pairwise_labels.any(dim=1, keepdim=True))  # [k, 1]
        top_antecedent_labels = torch.cat((dummy_labels, pairwise_labels), dim=1)  # [k, c+1]
        # loss函数
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)
        loss = torch.sum(loss)

        if len(gold_starts) > 0:
            type_loss = self.gold_span_type_loss(candidate_starts, candidate_ends, candidate_mention_types,
                                                 gold_mention_types)
            if type_loss != None:
                loss += type_loss
        else:
            type_loss = None

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedents, top_antecedent_scores, top_span_mention_scores], loss

    def get_train_example(self):
        with open(self.config["train_path"], encoding="utf-8") as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return examples

    def get_eval_example(self):
        with open(self.config["eval_path"], encoding="utf-8") as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return examples

    def get_test_example(self):
        with open(self.config["test_path"], encoding="utf-8") as f:
            examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return examples

    def tensorize_example(self, example, is_training):
        """样例处理"""
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in utils.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))  # 计算每个簇的次数
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        # print("len of sentences:", len(sentences))
        num_words = sum(len(s) for s in sentences)  # 实际的token数
        speakers = example["speakers"]
        # assert num_words == len(speakers), (num_words, len(speakers))
        speaker_dict = self.get_speaker_dict(utils.flatten(speakers))
        sentence_map = example['sentence_map']

        max_sentence_length = self.config["max_segment_len"]
        text_len = np.array([len(s) for s in sentences])

        input_ids, input_mask, speaker_ids = [], [], []
        static_word_emb = []
        word_emb = self.bert.embeddings.word_embeddings.weight.data
        for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            speaker_ids.append(sent_speaker_ids)
            input_mask.append(sent_input_mask)
            static_word_emb.append(word_emb[sent_input_ids])  # ------

        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        # static_word_emb = np.array(static_word_emb)
        static_word_emb = torch.tensor([item.cpu().detach().numpy() for item in static_word_emb])
        # print("num_words: ", num_words)
        # print("sum(input mask): ", np.sum(input_mask))
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        doc_key = example["doc_key"]
        self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
        self.gold[doc_key] = example["clusters"]
        genre = self.genres.get(doc_key[:2], 0)

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        example_tensors = (input_ids, input_mask, text_len, speaker_ids, genre, is_training,
                           gold_starts, gold_ends, cluster_ids, sentence_map, sentences, static_word_emb)  # ---------

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            if self.config['single_example']:
                return self.truncate_example(*example_tensors)  # 包括截断的sentences
            else:
                offsets = range(self.config['max_training_sentences'], len(sentences),
                                self.config['max_training_sentences'])
                tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
                return tensor_list
        # else:
        # return example_tensors
        # elif len(sentences) > 28:
        #     offsets = range(self.config['max_training_sentences'], len(sentences),
        #                     self.config['max_training_sentences'])
        #     tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
        #     return tensor_list
        else:
            return example_tensors

    def tensorize_example2(self, example, is_training):
        """样例处理"""
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in utils.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))  # 一个mention单独为一类
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1   # 同一个cluster中的mention对应clu id相同

        #
        gold_mention_types = np.zeros(len(gold_mentions))       # 取出每个gold mention所对应的type值
        for span_type in example["spans_type"]:
            type_value = self.get_span_type_value(span_type[1])
            gold_mention_types[gold_mention_map[tuple(span_type[0])]] = type_value

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        # speakers = example["speakers"]
        # assert num_words == len(speakers), (num_words, len(speakers))
        # speaker_dict = self.get_speaker_dict(utils.flatten(speakers))
        sentence_map = example['sentence_map']

        max_sentence_length = self.config["max_segment_len"]   # 句长
        text_len = np.array([len(s) for s in sentences])  # 统计每句话长度

        input_ids, input_mask, speaker_ids = [], [], []
        # static_word_emb = []
        word_emb = self.bert.embeddings.word_embeddings.weight.data
        for i, sentence in enumerate(sentences):
            # print("sentence:", sentence)
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)

            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            # static_word_emb.append(word_emb[sent_input_ids])  # ------
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        # static_word_emb = torch.tensor([item.cpu().detach().numpy() for item in static_word_emb])
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        '''
        doc_key = example["doc_key"]
        self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
        self.gold[doc_key] = example["clusters"]
        genre = self.genres.get(doc_key[:2], 0)
        '''

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        # example_tensors = (input_ids, input_mask, text_len, speaker_ids, genre, is_training,
                           # gold_starts, gold_ends, cluster_ids, sentence_map, sentences, static_word_emb)

        example_tensors = (input_ids, input_mask, text_len, is_training,
                           gold_starts, gold_ends, cluster_ids, sentence_map, sentences, gold_mention_types)


        if is_training and len(sentences) > self.config["max_training_sentences"]:
            if self.config['single_example']:
                return self.truncate_example(*example_tensors)
            else:
                offsets = range(self.config['max_training_sentences'], len(sentences),
                                self.config['max_training_sentences'])
                tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
                return tensor_list
        else:
            return example_tensors

    def get_speaker_dict(self, speakers):
        """返回说话者对应的字典"""
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    def get_span_type_value(self, span_type):
        """返回span类型对应的值"""
        # type_dict = {'其他': 0, '本体': 1, '喻体': 2, '喻体属性': 3, '喻体动作': 4, '喻体部件': 5,
        # '本体属性': 6, '本体部件': 7, '存疑': 0}
        # , "存疑": 5, "其他": 5
        type_dict = {'本体': 0, '喻体': 1, '喻体属性': 2, '喻体动作': 3, '喻体部件': 4}

        return type_dict[span_type]

    def get_type_str(self, span_int):
        """根据span 序号返回对应的str"""
        # type_int_2_str = {0: "其他", 1: "本体", 2: "喻体", 3: "喻体属性", 4: "喻体动作", 5: "本体属性", 6: "本体部件"}
        type_int_2_str = {0: "本体", 1: "喻体", 2: "喻体属性", 3: "喻体动作", 4: "喻体部件"}

        return type_int_2_str[span_int]

    def tensorize_mentions(self, mentions):
        """找到指代的起始"""
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def truncate_example(self, input_ids, input_mask, text_len,  is_training, gold_starts, gold_ends,
                         cluster_ids, sentence_map, sentences, static_word_emb, sentence_offset=None):
        """截断句长"""
        # speaker_ids, genre
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences) if sentence_offset is None \
            else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
        # speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]
        sentences = sentences[sentence_offset:sentence_offset + max_training_sentences]  # 保留下来的sentences内容
        static_word_emb = static_word_emb[sentence_offset:sentence_offset + max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return input_ids, input_mask, text_len, is_training, gold_starts, gold_ends, cluster_ids, sentence_map, sentences, static_word_emb

    def modify_top_span_scores(self, top_span_mention_scores, top_span_starts, top_span_ends, sentences):
        pronouns = ["你", "我", "他", "她", "它", "您", "你们", "我们", "他们",
                    "它们", "这个", "那个", "这些", "那些", "其", "那里", "那儿", "这儿"]
        sentences = [word for sen in sentences for word in sen]
        str_mention2_same_span = {}  # 找出内容一样的span，及其对应分数{ str_span1:[(span1, score), (span2,score)], ...}
        # print("length of top_span_mention_scores: ", top_span_mention_scores, "length of top_span_starts")
        # print("top_span_mention_scores: ", top_span_mention_scores)
        # print("top_span_mention_scores's type: ", type(top_span_mention_scores[0]))
        mention_score_rank = {}  # 记录修改分数以后mention排名
        for i in range(len(top_span_starts)):
            span_str = ''.join(sentences[top_span_starts[i]:top_span_ends[i] + 1])
            # if span_str in pronouns:
            #     continue
            if span_str not in str_mention2_same_span:
                str_mention2_same_span[span_str] = [(i, top_span_mention_scores[i])]
            else:
                str_mention2_same_span[span_str].append((i, top_span_mention_scores[i]))

        for span_str, cont in str_mention2_same_span.items():  # 将内容相同的span对应分数取其最大的
            if len(cont) == 1:  # 只有一个span则不用改top span分数  修改代词分数
                mention_score_rank[span_str] = cont[0][1]
                continue
            # elif cont in pronouns:         # 不修改代词分数
            # continue
            else:
                max_score = cont[0][1]
                for j in range(len(cont)):
                    if max_score < cont[j][1]:
                        max_score = cont[j][1]
                # print('0-span ', span_str, ' cont: ', cont)
                mention_score_rank[span_str] = max_score
                for k in range(len(cont)):
                    b = max_score.clone()
                    top_span_mention_scores[cont[k][0]] = b  # 对应top span scores改为最大的
                    # print('1: ', top_span_mention_scores[cont[k][1]])
        # print("change score result: ", str_mention2_same_span)
        # print("\nrank score result: ", sorted(mention_score_rank.items(), key=lambda kv:(kv[1], kv[0]),reverse=True))
        return top_span_mention_scores

    def cluster_top_span_scores(self, top_span_mention_scores, top_span_starts, top_span_ends, sentences):

        pronouns = ["你", "我", "他", "她", "它", "您", "你们", "我们", "他们",
                    "它们", "这个", "那个", "这些", "那些", "其", "那里", "那儿", "这儿"]
        sentences = [word for sen in sentences for word in sen]
        str_mention2_same_span = {}  # 找出内容一样的span，及其对应分数{ str_span1:[(span1, score), (span2,score)], ...}
        origin_mention = {}
        mention_score_rank = {}  # 记录修改分数以后mention排名
        for i in range(len(top_span_starts)):
            span_str = ''.join(sentences[top_span_starts[i]:top_span_ends[i] + 1])
            if span_str not in str_mention2_same_span:
                str_mention2_same_span[span_str] = [(i, top_span_mention_scores[i])]
            else:
                str_mention2_same_span[span_str].append((i, top_span_mention_scores[i]))
            origin_mention[(span_str, (int(top_span_starts[i]),int(top_span_ends[i])))] = top_span_mention_scores[i]

        top_cluster_num = 0.7 * len(str_mention2_same_span)    # 取一定比例排名靠前的same top spans  0.2 *
        for span_str, cont in str_mention2_same_span.items():  # 将内容相同的span对应分数取其最大的
            if len(cont) == 1:  # 只有一个span则不用改top span分数
                mention_score_rank[span_str] = cont[0][1]
                continue
            # elif cont in pronouns:   # 不修改代词分数
            #     continue
            else:
                max_score = cont[0][1]
                for j in range(len(cont)):
                    if max_score < cont[j][1]:
                        max_score = cont[j][1]
                # print('0-span ', span_str, ' cont: ', cont)
                mention_score_rank[span_str] = max_score

        mention_score_rank_list = sorted(mention_score_rank.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
        origin_mention_rank_list = sorted(origin_mention.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
        for i, cont_tuple in enumerate(mention_score_rank_list):   # 根据排名来修改分数
            if i < top_cluster_num:
                span_str = cont_tuple[0]
                # print("span:", span_str)
                max_score = cont_tuple[1]
                # print("max score:", max_score)
                cont = str_mention2_same_span[span_str]
                if len(cont) == 1:
                    continue
                else:
                    for k in range(len(cont)):
                        b = max_score.clone()
                        # print(cont[k][0])
                        top_span_mention_scores[cont[k][0]] = b  # 对应top span scores改为最大的
        mention_orin = [span[0][0] for span in origin_mention_rank_list]
        mention_rank = [span[0] for span in mention_score_rank_list]
        print("origin mention rank:", mention_orin)
        print("mention rank result:", mention_rank)
        return top_span_mention_scores

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        """根据mask展平embedding"""
        num_sentences = emb.shape[0]
        max_sentence_length = emb.shape[1]

        emb_rank = emb.dim()
        if emb_rank == 2:
            flattened_emb = emb.view(num_sentences * max_sentence_length)
        elif emb_rank == 3:
            flattened_emb = emb.view(num_sentences * max_sentence_length, emb.shape[2])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        mask = text_len_mask.view(num_sentences * max_sentence_length) == 1
        return flattened_emb[mask]

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = torch.eq(to_cuda(labeled_starts.view(-1, 1)),
                              to_cuda(candidate_starts.view(1, -1)))  # [num_labeled, num_candidates]
        same_end = torch.eq(to_cuda(labeled_ends.view(-1, 1)),
                            to_cuda(candidate_ends.view(1, -1)))  # [num_labeled, num_candidates]
        same_span = same_start & same_end  # [num_labeled, num_candidates]
        candidate_labels = torch.matmul(to_cuda(labels.view(1, -1).float()), same_span.float())  # [1, num_candidates]
        candidate_labels = candidate_labels.squeeze(0)  # [num_candidates]
        return candidate_labels

    def get_span_emb(self, context_outputs, span_starts, span_ends, static=False):
        """得到span_embedding([span start, span end, span width embedding, span head embedding])"""
        span_emb_list = []
        if static == False:
            # span start
            span_start_emb = context_outputs[span_starts]  # [num_candidates ,hidden_size]
            span_emb_list.append(span_start_emb)
            # span end
            span_end_emd = context_outputs[span_ends]
            span_emb_list.append(span_end_emd)
        else:  # 计算average span emb
            average_span_emb = []
            for i in range(len(span_starts)):
                start_span = span_starts[i]
                # one_span_emb = context_outputs[start_span]
                end_span = span_ends[i]
                one_span_emb = context_outputs.narrow(0, start_span, end_span - start_span + 1).mean(dim=0,
                                                                                                     keepdim=True)
                average_span_emb.append(one_span_emb)
                # one_span_emb = [[one_dim] for one_dim in one_span_emb]
            average_span_emb = torch.tensor([item.cpu().detach().numpy() for item in average_span_emb]).cuda()
            average_span_emb = torch.squeeze(average_span_emb)
            # print("shape of average span emb: ", len(average_span_emb))
            # average_span_emb = torch.Tensor(average_span_emb)
            span_emb_list.append(average_span_emb)

        span_width = 1 + span_ends - span_starts  # [num_candidates]

        # span width embedding
        if self.config["use_features"]:
            span_width_index = span_width - 1  # [num_candidates]
            # print("span_width_index:", span_width_index)
            span_width_emb = self.span_width_embedding(
                to_cuda(span_width_index))  # [num_candidates, self.config["feature_size"]]
            # span_width_emb = self.span_width_embedding(span_width_index)
            span_width_emb = self.dropout(span_width_emb)

            span_emb_list.append(span_width_emb)  # [num_candidates, 20]

        # span head embedding
        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_emb = torch.matmul(mention_word_scores, context_outputs)  # [K, T]
            span_emb_list.append(head_attn_emb)

        span_emb = torch.cat(span_emb_list, 1)  # 将这几个特征横向连接起来，所以列长度=768+768+ 768 + 20

        return span_emb

    def get_mention_scores(self, sentences, span_emb, span_starts, span_ends, is_training):
        """计算指代得分(span embedding 经过两层前向神经网络)
        """
        span_scores = self.mention_scores(span_emb)

        if self.config['use_prior']:
            span_width_emb = self.span_width_prior_embeddings.weight  # [30,20]
            span_width_index = span_ends - span_starts

            width_scores = self.width_scores(span_width_emb)
            width_scores = width_scores[span_width_index]
            span_scores += width_scores

        if self.config['use_salience']:  # 计算salience分数

            sentences_num = len(sentences)  # 句子数
            sentences = [word for sen in sentences for word in sen]
            # max_num = 0      #
            # max_span = '0'
            span_str_num_id = {}  # 记录每个span在文本中出现的次数, 和对应的mention [num, [(mention1), (mention2), ...]]
            for i, start in enumerate(span_starts):  # 单独写个函数计算这部分
                span_str = ''.join(sentences[start: span_ends[i] + 1])
                if span_str in span_str_num_id:
                    span_str_num_id[span_str][0] += 1  # 次数加一
                    span_str_num_id[span_str][1].append(i)  # 记录位置
                    # if span_str_num_id[span_str][0] > max_num:
                    # max_num = span_str_num_id[span_str][0]   # 记录出现的最大次数
                    # max_span = span_str
                else:
                    span_str_num_id[span_str] = [1, [i]]  # 第一个存span str出现的次数， 第二个存对应的mention起始位置

            # print("max_num is : ", max_num, " span is :", max_span)
            span_to_num = []  # 每个span i对应出现的次数
            for key, value in span_str_num_id.items():
                span_content = span_str_num_id[key]
                spans_id = span_content[1]
                # if is_training and all_span_salience:    # 在训练阶段加上全局的数量
                # if key in all_span_salience:
                # span_content[0] = all_span_salience[key]
                # print("span ", key, "has ", span_content[0])
                for span_id in spans_id:  # 对应位置的mention
                    if span_content[0] >= self.config["num_salience"]:
                        span_content[0] = self.config["num_salience"] - 1  # 出现的次数替换
                    # span_content[0] = span_content[0] * (11/sentences_num)
                    span_to_num.insert(span_id, span_content[0])

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            span_to_num = torch.tensor(span_to_num).to(device)
            span_salience_emb = self.span_salience_embeddings.weight  # [max_num, 20]
            span_salience_scores = self.salience_scores(span_salience_emb)
            span_salience_scores = span_salience_scores[span_to_num]

            span_scores += span_salience_scores

        return span_scores

    def get_mention_types(self, span_emb):
        """计算候选span的类型"""
        span_types, span_type_embs = self.span_types_net(span_emb)  # [batch_size, class_num]
        value, types_indics = torch.max(span_types, 1)

        # return types_indics
        return span_types, types_indics, span_type_embs

    @staticmethod
    def extract_top_spans(span_scores, cand_start_idxes, cand_end_idxes, top_span_num):
        """获得前k个短语"""

        sorted_span_idxes = torch.argsort(span_scores, descending=True).tolist()

        top_span_idxes = []
        end_idx_to_min_start_dix, start_idx_to_max_end_idx = {}, {}
        selected_span_num = 0

        for span_idx in sorted_span_idxes:
            crossed = False
            start_idx = cand_start_idxes[span_idx]
            end_idx = cand_end_idxes[span_idx]

            if end_idx == start_idx_to_max_end_idx.get(start_idx, -1):
                continue

            for j in range(start_idx, end_idx + 1):
                if j in start_idx_to_max_end_idx and j > start_idx and start_idx_to_max_end_idx[j] > end_idx:
                    crossed = True
                    break

                if j in end_idx_to_min_start_dix and j < end_idx and end_idx_to_min_start_dix[j] < start_idx:
                    crossed = True
                    break

            if not crossed:
                top_span_idxes.append(span_idx)
                selected_span_num += 1

                if start_idx not in start_idx_to_max_end_idx or end_idx > start_idx_to_max_end_idx[start_idx]:
                    start_idx_to_max_end_idx[start_idx] = end_idx

                if end_idx not in end_idx_to_min_start_dix or start_idx < end_idx_to_min_start_dix[end_idx]:
                    end_idx_to_min_start_dix[end_idx] = start_idx

            if selected_span_num == top_span_num:
                break

        def compare_span_idxes(i1, i2):
            if cand_start_idxes[i1] < cand_start_idxes[i2]:
                return -1
            elif cand_start_idxes[i1] > cand_start_idxes[i2]:
                return 1
            elif cand_end_idxes[i1] < cand_end_idxes[i2]:
                return -1
            elif cand_end_idxes[i1] > cand_end_idxes[i2]:
                return 1
            else:
                return 0

        top_span_idxes.sort(key=cmp_to_key(compare_span_idxes))

        return (torch.Tensor(top_span_idxes) + torch.tensor(top_span_idxes[0]) * (top_span_num - selected_span_num))

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        """计算前c个得分"""
        k = top_span_emb.shape[0]
        top_span_range = torch.arange(k)  # [k]

        antecedent_offsets = top_span_range.view(-1, 1) - top_span_range.view(1, -1)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]

        fast_antecedent_scores = top_span_mention_scores.view(-1, 1) + top_span_mention_scores.view(1, -1)  # [k, k]
        fast_antecedent_scores += torch.log(to_cuda(antecedents_mask.float()))
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)

        if self.config['use_prior']:
            antecedent_distance_buckets = self.get_offset_bucket_idxes_batch(antecedent_offsets)

            antecedent_distance_emb = self.antecedent_distance_embedding.weight
            antecedent_distance_emb = self.dropout(antecedent_distance_emb)
            distance_scores = self.antecedent_distance_linear(antecedent_distance_emb)

            antecedent_distance_scores = distance_scores.squeeze(1)[antecedent_distance_buckets]

            fast_antecedent_scores += antecedent_distance_scores

        _, top_antecedents = torch.topk(fast_antecedent_scores, c, sorted=False)
        # print("top_antecedents:", top_antecedents)
        # print("fast_antecedent_scores:", fast_antecedent_scores)

        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = self.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = self.batch_gather(antecedent_offsets, top_antecedents)

        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_fast_antecedent_scores(self, top_span_emb):
        """计算前K个前指的得分"""
        source_top_span_emb = self.fast_antecedent_scores(top_span_emb)
        target_top_span_emb = self.dropout(top_span_emb)

        return torch.matmul(source_top_span_emb, target_top_span_emb.t())

    def get_type_weight(self, top_starts, top_ends, top_antecedents, top_span_types, gold_span_types):
        gold_span_2_type = {}
        for span_type in gold_span_types:
            gold_span_2_type[tuple(span_type[0])] = self.get_span_type_value(span_type[1])

        top_ant_types = top_span_types[top_antecedents]
        ants_type_weight = []
        top_span_type_weight = []  # 判断top span的type预测是否正确
        for i, ants in enumerate(top_antecedents):
            top_start = top_starts[i]
            top_end = top_ends[i]
            top_span_type = top_span_types[i]
            if (top_start, top_end) in gold_span_2_type:
                if top_span_type == gold_span_2_type[(top_start, top_end)]:
                    top_span_type_weight.append(1)
                else:
                    top_span_type_weight.append(0)
            else:
                top_span_type_weight.append(0)
            # 求ant types是否预测正确
            ants_types = top_ant_types[i]
            ants_starts = top_starts[ants]
            ants_ends = top_ends[ants]
            one_ant_type_weight = []
            for j, ant_start in enumerate(ants_starts):
                ant_end = ants_ends[j]
                ant_type = ants_types[j]
                if (ant_start, ant_end) in gold_span_2_type:
                    if ant_type == gold_span_2_type[(ant_start, ant_end)]:
                        one_ant_type_weight.append(1)
                    else:
                        one_ant_type_weight.append(0)
                else:
                    one_ant_type_weight.append(0)
            ants_type_weight.append(one_ant_type_weight)

        top_span_type_weight = np.array(top_span_type_weight)
        top_span_type_weight = to_cuda(torch.from_numpy(top_span_type_weight).long())
        top_span_type_weight = top_span_type_weight.unsqueeze(-1)
        ants_type_weight = np.array(ants_type_weight)
        ants_type_weight = to_cuda(torch.from_numpy(ants_type_weight).long())
        ants_type_weight = ants_type_weight.unsqueeze(-1)

        return top_span_type_weight, ants_type_weight

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, genre_emb, top_span_types_all, segment_distance=None):
        k = top_span_emb.shape[0]
        c = top_antecedents.shape[1]

        feature_emb_list = []

        # 说话者、类别信息
        if self.config["use_metadata"]:
            top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedents]
            same_speaker = torch.eq(top_span_speaker_ids.view(-1, 1), top_antecedent_speaker_ids)
            speaker_pair_emb = self.same_speaker_embedding(same_speaker.type(torch.int64))  # [k, c, emb20]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = genre_emb.view(1, -1).view(1, -1).repeat(k, c, 1)  # [k, c, emb20]
            feature_emb_list.append(tiled_genre_emb)
        #
        if self.config["use_features"]:
            antecedent_distance_buckets = to_cuda(self.get_offset_bucket_idxes_batch(top_antecedent_offsets))
            antecedent_distance_emb = self.antecedent_offset_embedding(antecedent_distance_buckets)  # [k, c, emb20]
            feature_emb_list.append(antecedent_distance_emb)  # [k, c, emb20]
        #
        if segment_distance is not None:
            segment_distance_emb = self.segment_distance_embedding(to_cuda(segment_distance))  # [k, emb]
            feature_emb_list.append(segment_distance_emb)

        if self.config["use_span_type"]:
            # top_antecedent_types = top_span_types[top_antecedents]
            # ant_type_emb = self.span_type_embedding(top_antecedent_types)
            # top_span_type_emb = self.span_type_embedding(top_span_types)
            
            
            # ant_type_emb = top_span_type_hidden_emb[top_antecedents]
            # top_span_type_emb = top_span_type_hidden_emb
            
            # print("top_span_type_hidden_emb size:", top_span_type_hidden_emb.size())
            # print("ant span type hidden emb size:", ant_type_emb.size())
            # print("after unsqueeze  top span type emb size:", top_span_type_emb.unsqueeze(1).repeat(1, c, 1).size())
            # feature_emb_list.append(top_span_type_emb.unsqueeze(1).repeat(1, c, 1))
            # feature_emb_list.append(ant_type_emb)
              
            
            type_indexes = to_cuda(torch.tensor([0, 1, 2, 3, 4]))
            top_span_type_indexes = type_indexes.expand(k, 5)   # [k, 5]
            # print("top_span_type_indexes size:", top_span_type_indexes.size())
            top_span_type_emb = self.span_type_embedding(top_span_type_indexes)   # [k, 5, 20]
            # print("top_span_type_emb size 0:", top_span_type_emb.size())
            top_span_types_all = top_span_types_all.unsqueeze(2).repeat(1, 1, 20)   # type weight [k, 5, 20]
            # print("top_span_types_all size:", top_span_types_all.size())
            top_span_type_emb = top_span_types_all * top_span_type_emb        # [k, 5, 20]
            top_span_type_emb = top_span_type_emb.view(k, -1)             # [k, 5 * 20]
            # print("top_span_type_emb size 1:", top_span_type_emb.size())
            
            
            ant_span_type_indexes = type_indexes.expand(k, c, 5)   # [k, c, 5]
            # print("ant_span_type_indexes size:", ant_span_type_indexes.size())
            ant_span_type_emb = self.span_type_embedding(ant_span_type_indexes)   # [k, c, 5, 20]
            # print("ant_span_type_emb 0:", ant_span_type_emb.size())
            
            ant_span_type_all = top_span_types_all[top_antecedents]   # [k, c, 5, 20]
            # print("ant_span_all types size 0 :", ant_span_type_all.size())
            
            ant_span_type_emb = ant_span_type_all * ant_span_type_emb     # [k, c, 5, 20]
            # ant_span_type_emb = torch.cat(ant_span_type_emb, 3)      # [k, c, 5 * 20]
            ant_span_type_emb = ant_span_type_emb.view(k, c, -1)
            # print("ant_span_type_emb 1:", ant_span_type_emb.size())

            feature_emb_list.append(top_span_type_emb.unsqueeze(1).repeat(1, c, 1))
            feature_emb_list.append(ant_span_type_emb)

        target_emb = top_span_emb.unsqueeze(1)        # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb      # [k, c, emb]
        similarity_emb2 = target_emb - top_antecedent_emb
        target_emb = target_emb.repeat(1, c, 1)      # [k, c, emb]

        # --------
        feature_emb = torch.cat(feature_emb_list, 2)  # [k, c, emb80 每个特征20]
        feature_emb = self.dropout(feature_emb)

        # 三维
        pair_emb = torch.cat((target_emb, top_antecedent_emb, similarity_emb, feature_emb), 2)  # [k, c, emb]
        # pair_emb = torch.cat((target_emb, top_antecedent_emb, similarity_emb), 2)
        slow_antecedent_scores = self.slow_antecedent_scores(pair_emb)  # [k, c]

        return slow_antecedent_scores

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = encoded_doc.shape[0]
        num_c = span_starts.shape[0]

        doc_range = torch.arange(num_words).view(1, -1).repeat(num_c, 1)
        mention_mask = (doc_range >= (span_starts.view(-1, 1))) & (doc_range <= span_ends.view(-1, 1))

        word_attn = self.masked_mention_score(encoded_doc)
        mention_word_attn = F.softmax(torch.log(to_cuda(mention_mask.float())) + to_cuda(word_attn.view(1, -1)), dim=-1)

        return mention_word_attn

    def get_offset_bucket_idxes_batch(self, offsets_batch):
        """
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        log_space_idxes_batch = (torch.log(offsets_batch.float()) / math.log(2)).floor().long() + 3

        identity_mask_batch = (offsets_batch <= 4).long()

        return torch.clamp(
            identity_mask_batch * offsets_batch + (1 - identity_mask_batch) * log_space_idxes_batch, min=0, max=9)

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + torch.log(antecedent_labels.float())
        marginalized_gold_scores = torch.logsumexp(gold_scores, dim=1)  # [k]
        log_norm = torch.logsumexp(antecedent_scores, dim=1)  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def batch_gather(self, emb, indices):
        batch_size = emb.shape[0]
        seqlen = emb.shape[1]
        if len(emb.shape) > 2:
            emb_size = emb.shape[2]
        else:
            emb_size = 1
        flattened_emb = emb.view(batch_size * seqlen, emb_size)
        offset = to_cuda((torch.arange(batch_size) * seqlen).view(-1, 1))
        gathered = flattened_emb[indices + offset]

        if len(emb.shape) == 2:
            gathered = torch.squeeze(gathered, 2)
        return gathered

    def gold_span_type_loss(self, candidate_starts, candidate_ends, candidate_span_types, gold_mention_types):
        """
        计算gold span的预测情况
        """
        # candidate_span_types = candidate_span_types.cpu().detach().numpy().tolist()
        span_2_type_dict = {}
        for i, start in enumerate(candidate_starts):
            end = candidate_ends[i]
            # span_2_type_dict[(int(start), int(end))] = candidate_span_types[i]
            span_2_type_dict[(int(start), int(end))] = i

        predict_span_type = []
        gold_mention_type = []
        for span_type in gold_mention_types:
            if tuple(span_type[0]) in span_2_type_dict:
                type_index = span_2_type_dict[tuple(span_type[0])]
                cand_type = torch.unsqueeze(candidate_span_types[type_index], 0)
                predict_span_type.append(cand_type)
                gold_mention_type.append(self.get_span_type_value(span_type[1]))

        predict_span_type = torch.cat(predict_span_type, 0)
        # print("predict_span_type:", predict_span_type)
        # predict_span_type = np.array(predict_span_type)
        # predict_span_type = to_cuda(torch.from_numpy(predict_span_type).float())
        gold_mention_type = np.array(gold_mention_type)
        gold_mention_type = to_cuda(torch.from_numpy(gold_mention_type).long())

        if len(predict_span_type) == 0:
            return None
        # print("predict_span_type:", predict_span_type)
        # print("gold_mention_type:", gold_mention_type)
        crossloss = torch.nn.CrossEntropyLoss(reduction='sum')        # torch.nn.CrossEntropyLoss(reduction='sum')------try
        crossloss = crossloss.cuda()
        loss = crossloss(predict_span_type, gold_mention_type)

        return loss

    def get_span_recall(self, top_span_starts, top_span_ends, gold_starts, gold_ends, sentences):

        top_spans = list(zip((int(i) for i in top_span_starts), (int(i) for i in top_span_ends)))
        gold_mentions = list(zip((int(i) for i in gold_starts), (int(i) for i in gold_ends)))
        right_span_num = 0
        not_find_gold_span, all_gold_span_str = [], []
        for gold_span in gold_mentions:
            span_str = ''.join(sentences[0][gold_span[0]:gold_span[1] + 1])
            if gold_span in top_spans:
                right_span_num += 1
            else:
                not_find_gold_span.append(span_str + str(gold_span))
            all_gold_span_str.append(span_str + str(gold_span))
        if len(gold_mentions) == 0:
            span_recall = 0
        else:
            span_recall = right_span_num / len(gold_mentions)
            span_recall = round(span_recall, 5)
        return span_recall, not_find_gold_span, all_gold_span_str

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        """获得预测的前值簇"""
        predicted_antecedents = []
        # print("ant of shape ", len(antecedents))
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_metaphor_ants(self, antecedents, antecedent_scores):
        """获得预测分数>0的下标"""
        sort_value, sort_index = torch.sort(antecedent_scores, descending=True)
        # print("sort_index:", sort_index)
        sort_index = sort_index - 1
        # print("sort_index - 1:", sort_index)

        predicted_antecedents = []   # 二维的
        for i, index_list in enumerate(sort_index):
            span_ant = []
            if index_list[0] < 0:
                predicted_antecedents.append([-1])
                continue
            for index in index_list:
                if index < 0:
                    break
                span_ant.append(antecedents[i, index])
            if len(span_ant) > 0:
                predicted_antecedents.append(span_ant)
            else:
                predicted_antecedents.append([-1])
        # print("predicted_antecedents:", predicted_antecedents)
        return predicted_antecedents

    def get_metaphor_ants_2(self, antecedents, top_span_starts, top_span_ends, antecedent_scores):
        """
        获得预测分数>0的下标
        删除与最高分的候选有交集的其他候选
        """
        sort_value, sort_index = torch.sort(antecedent_scores, descending=True)
        sort_index = sort_index - 1

        predicted_antecedents = []  # 二维的
        for i, index_list in enumerate(sort_index):
            span_ant = []
            if index_list[0] < 0:
                predicted_antecedents.append([-1])
                continue
            max_ant = antecedents[i, index_list[0]]  # 最高分的候选span
            max_ant_start = int(top_span_starts[max_ant])
            max_ant_end = int(top_span_ends[max_ant])
            max_ant_span = (max_ant_start, max_ant_end)
            for index in index_list:
                if index < 0:  # 之后的候选下标都<0
                    break
                overlap_flag = 0
                ant = antecedents[i, index]
                ant_start = int(top_span_starts[ant])
                ant_end = int(top_span_ends[ant])
                ant_span = (ant_start, ant_end)
                if max_ant_span == ant_span:
                    overlap_flag = 0
                elif max_ant_start <= ant_start <= max_ant_end or ant_start <= max_ant_start <= ant_end:
                    # print("top span:", top_span, " ant span:", ant_span)
                    overlap_flag = 1

                if overlap_flag == 0:
                    span_ant.append(antecedents[i, index])

            if len(span_ant) > 0:
                predicted_antecedents.append(span_ant)
            else:
                predicted_antecedents.append([-1])
        # print("predicted_antecedents:", predicted_antecedents)
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        """获得预测的共指簇"""
        mention_to_predicted = {}  # 存mention所对应的cluster id
        predicted_clusters = []  # 存每个cluster有哪些mention
        for i, predicted_index in enumerate(predicted_antecedents):
            # predicted_antecedents和top_span_starts是一一对应的，pre_ant存的是top—span的下标
            if predicted_index < 0:
                continue
            assert i > predicted_index, (i, predicted_index)    # i > predicted_index  共指的前指必须在当前span的前面
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)  # 所属第i个cluster  单独为一类
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster  # mention对应所属类序号

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def get_metaphor_clusters(self, top_span_starts, top_span_ends, predicted_ants):
        """获取隐喻中的隐喻关系链（可能包含单个span构成的链）"""
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_ants):
            if predicted_index < 0:
                # mention = (int(top_span_starts[i]), int(top_span_ends[i]))
                # predicted_cluster = len(predicted_clusters)
                # # print("i: ", i, " mention:", mention, "predicted_cluster:", predicted_cluster)
                # predicted_clusters.append([mention])
                # mention_to_predicted[mention] = predicted_cluster
                continue
            assert i > predicted_index, (i, predicted_index)
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            # if predicted_antecedent in mention_to_predicted:   # ant 已经在某一个链中
            #     predicted_cluster = mention_to_predicted[predicted_antecedent]
            # else:
            predicted_cluster = len(predicted_clusters)  # 所属第i个cluster  单独为一类
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster  # mention对应所属类序号

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def delete_include_predict_clusters(self, predicted_clusters):
        """
        删除不同预测的隐喻链中有包含关系的预测链
        """
        predicted_clusters = [list(clu) for clu in predicted_clusters]
        new_predict_clusters, used_span = [], []   # 已经删除过链的span
        pre_span_2_clu = {}    # span对应所在的链
        pre_span_2_span = {}   # 与span在同一个隐喻链中的span
        for pre_cluster in predicted_clusters:
            for pre_span in pre_cluster:
                if pre_span in pre_span_2_clu:
                    pre_span_2_clu[pre_span].append(pre_cluster)
                else:
                    pre_span_2_clu[pre_span] = [pre_cluster]
            span_a = pre_cluster[0]
            span_b = pre_cluster[1]
            if span_a not in pre_span_2_span:
                pre_span_2_span[span_a] = [span_b]
            else:
                pre_span_2_span[span_a].append(span_b)

            if span_b not in pre_span_2_span:
                pre_span_2_span[span_b] = [span_a]
            else:
                pre_span_2_span[span_b].append(span_a)

        # 删除有包含关系的隐喻span
        for pre_cluster in predicted_clusters:

            for p_span in pre_cluster:
                if p_span not in used_span:
                    used_span.append(p_span)

                    meta_spans = pre_span_2_span[p_span]    # 构成隐喻的span的集合list
                    if len(meta_spans) == 1:   # 只是单链,也就是这个链
                        # if pre_cluster not in new_predict_clusters:
                        #     new_predict_clusters.append(pre_cluster)
                        #     break
                        continue
                    # print("p_span:", p_span)
                    m_span_flag = 0
                    for j, m_span in enumerate(meta_spans):
                        if m_span_flag == 1:
                            m_span = meta_spans[j-1]
                            j -= 1
                        start = m_span[0]    # 当前的meta span 与其他 meta span做比较
                        end = m_span[1]
                        # print("m_span:", m_span)
                        if j <= len(meta_spans) - 2:
                            for k, compare_span in enumerate(meta_spans[j+1:]):
                                # print("compare_span:", compare_span)
                                c_start = compare_span[0]
                                c_end = compare_span[1]

                                if start <= c_start and end >= c_end:   # m span 包含了这个compare span
                                    # print("type 1")
                                    # del pre_span_2_span[p_span][j+k+1]     # 删除这个c span
                                    for l, l_span in enumerate(pre_span_2_span[p_span]):
                                        if l_span == compare_span:
                                            del pre_span_2_span[p_span][l]
                                            break

                                    for c, c_span in enumerate(pre_span_2_span[compare_span]):
                                        if c_span == p_span:
                                            del pre_span_2_span[compare_span][c]
                                            break

                                    continue

                                elif c_start <= start and end <= c_end:  # c span 包含了这个m span
                                    # print("type 2")
                                    # del pre_span_2_span[p_span][j]   # 删除m span
                                    for l, l_span in enumerate(pre_span_2_span[p_span]):
                                        if l_span == m_span:
                                            del pre_span_2_span[p_span][l]
                                            # break

                                    for c, c_span in enumerate(pre_span_2_span[m_span]):
                                        if c_span == p_span:
                                            del pre_span_2_span[m_span][c]
                                            # break
                                    m_span_flag = 1
                                    # break   # m span已经被删除，无法与其他span比较

        for span, meta_list in pre_span_2_span.items():
            if len(meta_list) == 0:
                continue

            for m_span in meta_list:
                new_clu = [span, m_span]
                new_clu_2 = [m_span, span]
                if new_clu not in new_predict_clusters and new_clu_2 not in new_predict_clusters:
                    new_predict_clusters.append(new_clu)

        new_predict_clusters = [tuple(clu) for clu in new_predict_clusters]
        # print("\npredicted_clusters:", predicted_clusters)
        # print("new_predict_clusters:", new_predict_clusters)

        return new_predict_clusters

    def delete_overlapped_clusters(self, predicted_clusters, metaphor_link_scores):
        """
        删除有交叠情况的隐喻链，保留分数更高的隐喻链
        :param predicted_clusters:  clu格式是list还是tuple
        :param metaphor_link_scores: 看里面的格式 span的格式是list还是tuple！！
        :return:
        """

        pre_span_2_span = {}  # 与span在同一个隐喻链中的span
        for pre_cluster in predicted_clusters:
            span_a = pre_cluster[0]
            span_b = pre_cluster[1]
            if span_a not in pre_span_2_span:
                pre_span_2_span[span_a] = [span_b]
            else:
                pre_span_2_span[span_a].append(span_b)

            if span_b not in pre_span_2_span:
                pre_span_2_span[span_b] = [span_a]
            else:
                pre_span_2_span[span_b].append(span_a)

        new_predict_clusters, in_new_predict_cluster = [], []
        for span, ant_list in pre_span_2_span.items():
            if len(ant_list) == 1:
                continue
            ant_score_list = metaphor_link_scores[tuple(span)]
            # print("span:", span, "ant_list:", ant_list)
            delete_ant_list = []
            for j, ant_span in enumerate(ant_list):
                start = ant_span[0]
                end = ant_span[1]

                if j <= len(ant_list) - 2:
                    # other_ant_span_list = ant_list[j+1:]
                    other_ant_span_list = pre_span_2_span[span]
                    # print("j:", j, " other_ant_span_list:", other_ant_span_list)
                    for k, other_ant_span in enumerate(other_ant_span_list):
                        if other_ant_span == ant_span:
                            continue
                        overlap_flag = 0
                        o_start = other_ant_span[0]
                        o_end = other_ant_span[1]
                        if start <= o_start <= end or o_start <= start <= o_end:
                            overlap_flag = 1
                        elif start <= o_start and o_end <= end:
                            overlap_flag = 1
                        elif o_start <= start and end <= o_end:
                            overlap_flag = 1
                        else:
                            overlap_flag = 0
                        # print("ant_span:", ant_span, " other_ant_span:", other_ant_span, " overlap_flag:", overlap_flag)
                        if overlap_flag == 1:  # 如果有交集才比较这两个隐喻链的分数
                            # print("overleapped ant span:", ant_span, " other ant span:", other_ant_span)  
                            ant_span_score, o_span_score = 0, 0
                            for m, s_ant_span in enumerate(ant_score_list[0]):
                                if tuple(ant_span) == s_ant_span:
                                    ant_span_score = ant_score_list[1][m]
                                elif tuple(other_ant_span) == s_ant_span:
                                    o_span_score = ant_score_list[1][m]

                            if ant_span_score >= o_span_score:  # 删除 other span
                                if k not in delete_ant_list:
                                    delete_ant_list.append(k)
                                # del pre_span_2_span[span][j+k+1]
                                # del pre_span_2_span[span][k]

                                other_span_ant_list = pre_span_2_span[other_ant_span]
                                for l, o_ant_span in enumerate(other_span_ant_list):
                                    if o_ant_span == span:
                                        del pre_span_2_span[other_ant_span][l]
                                        break
                                # break


                            elif ant_span_score < o_span_score:  # 删除 ant span
                                if j not in delete_ant_list:
                                    delete_ant_list.append(j)
                                # del pre_span_2_span[span][j]
                                # print("del ant span:", ant_span)
                                ant_span_ant_list = pre_span_2_span[ant_span]
                                for l, o_ant_span in enumerate(ant_span_ant_list):
                                    if o_ant_span == span:
                                        del pre_span_2_span[ant_span][l]
                                        break
                                # break
            delete_ant_list.sort(reverse=True)
            # # print("delete_ant_list:", delete_ant_list)
            for del_k in delete_ant_list:
                # print("del_k:", del_k)
                del pre_span_2_span[span][del_k]

        for span, meta_list in pre_span_2_span.items():
            if len(meta_list) == 0:
                continue

            for m_span in meta_list:
                new_clu = [span, m_span]
                new_clu_2 = [m_span, span]
                if new_clu not in new_predict_clusters and new_clu_2 not in new_predict_clusters:
                    new_predict_clusters.append(new_clu)

        new_predict_clusters = [tuple(clu) for clu in new_predict_clusters]
        # print("\npredicted_clusters:", predicted_clusters)
        # print("new_predict_clusters:", new_predict_clusters)

        return new_predict_clusters

    def get_metaphor_clusters_2(self, top_span_starts, top_span_ends, predicted_ants):
        """获取隐喻中的隐喻关系链（可能包含单个span构成的链）"""
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index_list in enumerate(predicted_ants):

            for predicted_index in predicted_index_list:
                if predicted_index == -1:
                    continue
                predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
                predicted_cluster = len(predicted_clusters)  # 所属第i个cluster  单独为一类
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster  # mention对应所属类序号

                mention = (int(top_span_starts[i]), int(top_span_ends[i]))
                predicted_clusters[predicted_cluster].append(mention)
                mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}   # 有问题
        # print("predicted_clusters:", predicted_clusters)
        # predicted_clusters = self.delete_include_predict_clusters(predicted_clusters)

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                if mention in mention_to_gold:
                    mention_to_gold[mention].append(gc)
                else:
                    mention_to_gold[mention] = [gc]
        # predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
        #                                                                        predicted_antecedents)
        meta_pre_clusters, meta_span_to_preclu = self.get_metaphor_clusters_2(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        # evaluator.update(meta_pre_clusters, gold_clusters, meta_span_to_preclu, mention_to_gold)
        # print("coref way get predict clusters:", predicted_clusters)
        # print("metaphor get predict clusters:", meta_pre_clusters)
        # evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        # p, r, f = Meta_evaluate(predicted_clusters, gold_clusters, mention_to_gold)   # _fuzzy
        # return predicted_clusters, gold_clusters, mention_to_gold
        return meta_pre_clusters, gold_clusters, mention_to_gold

    def load_eval_data(self):
        if self.eval_data is None:
            with open(self.config["eval_path"]) as f:
                self.eval_data = [json.loads(jsonline) for jsonline in f.readlines()]
                # self.eval_data = random.shuffle(examples)

            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, model, device, official_stdout=False, keys=None, eval_mode=False):  # train阶段验证
        self.load_eval_data()

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()
        doc_keys = []
        with torch.no_grad():
            for example_num, example in enumerate(tqdm(self.eval_data, desc="Eval_Examples")):
                tensorized_example = model.tensorize_example(example, is_training=False)

                # sentences = [word for sen in example["sentences"] for word in sen]    # --------------
                input_ids = torch.from_numpy(tensorized_example[0]).long().to(device)
                input_mask = torch.from_numpy(tensorized_example[1]).long().to(device)
                text_len = torch.from_numpy(tensorized_example[2]).long().to(device)
                speaker_ids = torch.from_numpy(tensorized_example[3]).long().to(device)
                genre = torch.tensor(tensorized_example[4]).long().to(device)
                is_training = tensorized_example[5]
                gold_starts = torch.from_numpy(tensorized_example[6]).long().to(device)
                gold_ends = torch.from_numpy(tensorized_example[7]).long().to(device)
                cluster_ids = torch.from_numpy(tensorized_example[8]).long().to(device)
                sentence_map = torch.Tensor(tensorized_example[9]).long().to(device)
                all_span_salience = None
                sentences = tensorized_example[10]
                # static_word_emb = tensorized_example[11].to(device)
                write_epoch = -1
                exam_i = -1
                if keys is not None and example['doc_key'] not in keys:
                    continue
                doc_keys.append(example['doc_key'])
                # print("doc key:", example["doc_key"])

                (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                 top_antecedents, top_antecedent_scores, top_span_mention_scores), loss = model(sentences, input_ids,
                                                                                                input_mask, text_len,
                                                                                                speaker_ids, genre,
                                                                                                is_training,
                                                                                                gold_starts, gold_ends,
                                                                                                cluster_ids,
                                                                                                sentence_map,
                                                                                                 write_epoch, exam_i)

                # predicted_antecedents = self.get_predicted_antecedents(top_antecedents.cpu(),
                #                                                        top_antecedent_scores.cpu())
                predicted_antecedents = self.get_metaphor_ants(top_antecedents.cpu(),
                                                                       top_antecedent_scores.cpu())
                coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends,
                                                                            predicted_antecedents, example["clusters"],
                                                                            coref_evaluator)

        summary_dict = {}
        if eval_mode:
            conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, self.subtoken_maps,
                                                 official_stdout)
            average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            summary_dict["Average F1 (conll)"] = average_f1
            print("Average F1 (conll): {:.2f}%".format(average_f1))

        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return summary_dict, f

    def evaluate_metaphor(self, model, device, eval_mode=False):    # 隐喻识别 train阶段验证
        self.load_eval_data()

        meta_predictions = {}
        doc_gold_clusters = {}
        doc_mention_to_gold = {}
        coref_evaluator = metrics.CorefEvaluator()
        p_all, r_all, f_all = 0, 0, 0
        meta_num = 0
        with torch.no_grad():
            for example_num, example in enumerate(tqdm(self.eval_data, desc="Eval_Examples")):
                tensorized_example = model.tensorize_example2(example, is_training=False)

                # sentences = [word for sen in example["sentences"] for word in sen]    # --------------
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
                # static_word_emb = tensorized_example[9].to(device)
                # gold_mention_types = torch.from_numpy(tensorized_example[10]).long().to(device)
                gold_mention_types = example["spans_type"]
                gold_clusters = example["clusters"]

                (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                 top_antecedents, top_antecedent_scores, top_span_mention_scores, top_span_types), loss = model(sentences, input_ids,
                                                                                                input_mask, text_len,
                                                                                                speaker_ids, genre,
                                                                                                is_training,
                                                                                                gold_starts, gold_ends,
                                                                                                cluster_ids,
                                                                                                sentence_map,
                                                                                             gold_mention_types, gold_clusters)
                # predicted_antecedents = self.get_predicted_antecedents(top_antecedents.cpu(),
                #                                                        top_antecedent_scores.cpu())
                predicted_antecedents = self.get_metaphor_ants(top_antecedents.cpu(),
                                                                       top_antecedent_scores.cpu())
                # print("doc:", example_num, " predicted_antecedents:", predicted_antecedents)
                meta_predictions[example_num], doc_gold_clusters[example_num], doc_mention_to_gold[example_num] \
                    = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"],
                                          coref_evaluator)
                if len(gold_starts) > 0:
                    meta_num += 1
                
        summary_dict = {}
        # print("meta_predictions:", meta_predictions)
        print("----------------metaphor link result-------------------")
        print("length of meta_predictions:", example_num+1)
        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, example_num+1))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        print("meta num:", meta_num)
        p_1, r_1, f_1 = Meta_evaluete_all_doc(meta_predictions, doc_gold_clusters, doc_mention_to_gold)
        print("all doc to compute meta links result:", ' p:', p_1, " r:", r_1, " f:", f_1)

        p_2, r_2, f_2 = Meta_evaluete_all_doc_fuzzy(meta_predictions, doc_gold_clusters, doc_mention_to_gold)
        print("all doc to compute meta links result fuzzy:", ' p:', p_2, " r:", r_2, " f:", f_2)

        return summary_dict, f_1  # f

    def get_metaphor_link_score(self, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores):
        """
        保存隐喻链的分数， span_a:[[与其隐喻的span(tuple)], [隐喻的分数]]
        :return:
        """
        sort_value, sort_index = torch.sort(top_antecedent_scores, descending=True)
        sort_index = sort_index - 1
        # print("sort_index:", sort_index)
        # print("sort_value:", sort_value)
        metaphor_link_score = {}
        for i, start in enumerate(top_span_starts):
            end = int(top_span_ends[i])
            span_a = (int(start), end)
            sort_ants_index = sort_index[i]
            if sort_ants_index[0] < 0:
                continue
            for index in sort_ants_index:
                if index < 0:
                    break
                ant = top_antecedents[i, index]
                ant_span = (int(top_span_starts[ant]), int(top_span_ends[ant]))
                ant_score = top_antecedent_scores[i][index + 1]  # ---------
                ant_score = float(ant_score)
                if span_a not in metaphor_link_score:
                    metaphor_link_score[span_a] = [[ant_span], [ant_score]]
                else:
                    metaphor_link_score[span_a][0].append(ant_span)
                    metaphor_link_score[span_a][1].append(ant_score)

                if ant_span not in metaphor_link_score:
                    metaphor_link_score[ant_span] = [[span_a], [ant_score]]
                else:
                    metaphor_link_score[ant_span][0].append(span_a)
                    metaphor_link_score[ant_span][1].append(ant_score)

        return metaphor_link_score
    
