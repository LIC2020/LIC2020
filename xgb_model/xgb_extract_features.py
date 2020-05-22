# -*- coding: utf-8 -*-
import os
import codecs
import functools
import logging
import json
import csv
import sys

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import jieba.posseg as psg

from transformers.tokenization_bert import BasicTokenizer

__author__ = "liuaiting@bupt.edu.cn"

tokenizer = BasicTokenizer(do_lower_case=True)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

eps = 10e-8

# ensemble_list = ["14", "17", "21", "22", "23", "25", "26", "27", "28", "29", "33", "34", "35", "36", "37"]
ensemble_list = ["14", "17", "21", "22", "23", "25", "26", "27", "28", "29", "33", "34", "35", "36", "37", "38", "39"]

punct = set(["。", ".", ",", "，", ";", "；", "!", "！", "?", "？"])

pos_list = ["Ag", "a", "ad", "an", "b", "c", "dg", "d", "e", "f", "g", "h", "j", "k", "l", "m", "Ng", "n", "nr", "ns",
            "nt", "nz", "o", "p", "q", "r", "s", "tg", "t", "u", "vg", "v", "vd", "vn", "w", "x", "y", "z", "un", "eng"]
pos_dict = dict(zip(pos_list, range(len(pos_list))))


def get_all_cands():
    logger.info("############   get_all_cands   ############")
    files = os.listdir("../output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_dev_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    f = json.load(open("../dureader_robust-data/dev.json"))
    cands = {}
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_data_join_utf8/" + pred_file))
        for data in f["data"]:
            for para in data["paragraphs"]:
                for qas in para["qas"]:
                    qid = qas["id"]
                    question = qas["question"]
                    answers = [ans["text"] for ans in qas["answers"]]
                    if not cands.get(qid):
                        cands[qid] = dict()
                    for pred_ans in pred.get(qid):
                        label = 1 if pred_ans["text"] in answers else 0
                        cands[qid][pred_ans["text"]] = label
    # TODO: dev切分成两份，用来做xgb的训练和验证
    with open("xgb_data2/train_cands.csv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["qid", "pred_answer", "label"])
        for qid in list(cands.keys())[:1000]:
            for k, v in cands[qid].items():
                writer.writerow([qid, k, v])
    with open("xgb_data2/valid_cands.csv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["qid", "pred_answer", "label"])
        for qid in list(cands.keys())[1000:]:
            for k, v in cands[qid].items():
                writer.writerow([qid, k, v])

    json.dump(cands, open("xgb_data2/dev_cands.json", "w"), ensure_ascii=False, indent=2)


def get_all_cands_data3():
    logger.info("############   get_all_cands   ############")
    files = os.listdir("../output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_dev_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    f = json.load(open("../dureader_robust-data/dev.json"))
    cands = {}
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_data_join_utf8/" + pred_file))
        for data in f["data"]:
            for para in data["paragraphs"]:
                for qas in para["qas"]:
                    qid = qas["id"]
                    question = qas["question"]
                    answers = [ans["text"] for ans in qas["answers"]]
                    if not cands.get(qid):
                        cands[qid] = dict()
                    for pred_ans in pred.get(qid):
                        label = 1 if pred_ans["text"] in answers else 0
                        cands[qid][pred_ans["text"]] = label
    # TODO: dev全部用来做xgb训练
    with open("xgb_data3/train_cands.csv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["qid", "pred_answer", "label"])
        for qid in list(cands.keys()):
            for k, v in cands[qid].items():
                writer.writerow([qid, k, v])
    json.dump(cands, open("xgb_data3/train_cands.json", "w"), ensure_ascii=False, indent=2)


def get_all_cands_test1():
    logger.info("############   get_all_cands_test1   ############")
    files = os.listdir("../output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_test1_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    f = json.load(open("../dureader_robust-test1/test1_dealed.json"))
    cands = {}
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_data_join_utf8/" + pred_file))
        for data in f["data"]:
            for para in data["paragraphs"]:
                for qas in para["qas"]:
                    qid = qas["id"]
                    question = qas["question"]
                    answers = [ans["text"] for ans in qas["answers"]]
                    if not cands.get(qid):
                        cands[qid] = dict()
                    for pred_ans in pred.get(qid):
                        label = 1
                        cands[qid][pred_ans["text"]] = label
    with open("xgb_data2/test1_cands.csv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["qid", "pred_answer", "label"])
        for qid in cands.keys():
            for k, v in cands[qid].items():
                writer.writerow([qid, k, v])

    json.dump(cands, open("xgb_data2/test1_cands.json", "w"), ensure_ascii=False, indent=2)


def get_all_cands_test2():
    logger.info("############   get_all_cands_test2   ############")
    files = os.listdir("../output_test2")
    pred_files = []
    for file in files:
        if "test2_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    f = json.load(open("../dureader_robust-test2/test2_dealed.json"))
    cands = {}
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_test2/" + pred_file))
        for data in f["data"]:
            for para in data["paragraphs"]:
                for qas in para["qas"]:
                    qid = qas["id"]
                    question = qas["question"]
                    answers = [ans["text"] for ans in qas["answers"]]
                    if not cands.get(qid):
                        cands[qid] = dict()
                    for pred_ans in pred.get(qid):
                        label = 1
                        cands[qid][pred_ans["text"]] = label
    with open("xgb_data2/test2_cands.csv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["qid", "pred_answer", "label"])
        for qid in cands.keys():
            for k, v in cands[qid].items():
                writer.writerow([qid, k, v])

    json.dump(cands, open("xgb_data2/test2_cands.json", "w"), ensure_ascii=False, indent=2)


def get_model_prob_train():
    logger.info("############   get_model_prob_train   ############")
    files = os.listdir("../output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_dev_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    def get_one_model_prob(pred_json, qid, pred_answer):
        prob = 0
        for pred_ans in pred_json[qid]:
            if pred_ans["text"] == pred_answer:
                prob = pred_ans["probability"]
        return prob

    df = pd.read_csv("xgb_data3/train_cands.csv", sep="\t", header=0)
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_data_join_utf8/" + pred_file))
        pred_file_num = "_".join(pred_file.split("_")[:2])
        df[pred_file_num] = df.apply(lambda row: get_one_model_prob(pred, row["qid"], row["pred_answer"]), axis=1)

    df.to_csv("xgb_data3/train.csv", header=True, index=False, sep="\t", encoding="utf-8")
    del df["qid"]
    del df["pred_answer"]
    df.to_csv("xgb_data3/y_train.csv", header=False, index=False, sep="\t", encoding="utf-8", columns=["label"])
    del df["label"]
    df.to_csv("xgb_data3/x_train.csv", header=False, index=False, sep="\t", encoding="utf-8")


def get_model_prob_valid():
    logger.info("############   get_model_prob_valid   ############")
    files = os.listdir("../output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_dev_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    def get_one_model_prob(pred_json, qid, pred_answer):
        prob = 0
        for pred_ans in pred_json[qid]:
            if pred_ans["text"] == pred_answer:
                prob = pred_ans["probability"]
        return prob

    df = pd.read_csv("xgb_data2/valid_cands.csv", sep="\t", header=0)
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_data_join_utf8/" + pred_file))
        pred_file_num = "_".join(pred_file.split("_")[:2])
        df[pred_file_num] = df.apply(lambda row: get_one_model_prob(pred, row["qid"], row["pred_answer"]), axis=1)

    df.to_csv("xgb_data2/valid.csv", header=True, index=False, sep="\t", encoding="utf-8")
    del df["qid"]
    del df["pred_answer"]
    df.to_csv("xgb_data2/y_valid.csv", header=False, index=False, sep="\t", encoding="utf-8", columns=["label"])
    del df["label"]
    df.to_csv("xgb_data2/x_valid.csv", header=False, index=False, sep="\t", encoding="utf-8")


def get_model_prob_test1():
    logger.info("############   get_model_prob_test1   ############")
    files = os.listdir("../output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_test1_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    def get_one_model_prob(pred_json, qid, pred_answer):
        prob = 0
        for pred_ans in pred_json[qid]:
            if pred_ans["text"] == pred_answer:
                prob = pred_ans["probability"]
        return prob

    df = pd.read_csv("xgb_data2/test1_cands.csv", sep="\t", header=0)
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_data_join_utf8/" + pred_file))
        pred_file_num = "_".join(pred_file.split("_")[:2])
        df[pred_file_num] = df.apply(lambda row: get_one_model_prob(pred, row["qid"], row["pred_answer"]), axis=1)

    df.to_csv("xgb_data2/test1_for_train.csv", header=True, index=False, sep="\t", encoding="utf-8")
    del df["qid"]
    del df["pred_answer"]
    df.to_csv("xgb_data2/y_test.csv", header=False, index=False, sep="\t", encoding="utf-8", columns=["label"])
    del df["label"]
    df.to_csv("xgb_data2/x_test.csv", header=False, index=False, sep="\t", encoding="utf-8")


def get_model_prob_test2():
    logger.info("############   get_model_prob_test2   ############")
    files = os.listdir("../output_test2")
    pred_files = []
    for file in files:
        if "test2_nbest" in file and file.split("_")[0] in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    logger.info("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    logger.info("pred_files: [{}]".format(len(pred_files)))

    def get_one_model_prob(pred_json, qid, pred_answer):
        prob = 0
        for pred_ans in pred_json[qid]:
            if pred_ans["text"] == pred_answer:
                prob = pred_ans["probability"]
        return prob

    df = pd.read_csv("xgb_data2/test2_cands.csv", sep="\t", header=0)
    for pred_file in pred_files:
        logger.info(pred_file)
        pred = json.load(open("../output_test2/" + pred_file))
        pred_file_num = "_".join(pred_file.split("_")[:2])
        df[pred_file_num] = df.apply(lambda row: get_one_model_prob(pred, row["qid"], row["pred_answer"]), axis=1)

    df.to_csv("xgb_data2/test2.csv", header=True, index=False, sep="\t", encoding="utf-8")
    del df["qid"]
    del df["pred_answer"]
    df.to_csv("xgb_data2/y_test2.csv", header=False, index=False, sep="\t", encoding="utf-8", columns=["label"])
    del df["label"]
    df.to_csv("xgb_data2/x_test2.csv", header=False, index=False, sep="\t", encoding="utf-8")


def gen_qa_data_from_nbest_file(input_file, nbest_file, output_file):
    f = json.load(open(input_file))
    pred = json.load(open(nbest_file))
    with open(output_file, "w") as fw:
        writer = csv.writer(fw, delimiter="\t")
        writer.writerow(["label", "query", "answer", "qid", "probability", "start_logit", "end_logit"])
        for data in f["data"]:
            for para in data["paragraphs"]:
                for qas in para["qas"]:
                    qid = qas["id"]
                    question = qas["question"]
                    answer = qas["answers"][0]["text"] if qas.get("answers") else ""
                    pred_ans_top1 = pred[qid][0]["text"]
                    pred_ans_top1_prob = pred[qid][0]["probability"]
                    pred_ans_top1_start_logit = pred[qid][0]["start_logit"]
                    pred_ans_top1_end_logit = pred[qid][0]["end_logit"]
                    label_top1 = 1 if pred_ans_top1 == answer else 0
                    res1 = [label_top1, question, pred_ans_top1, qid + "_1",
                            pred_ans_top1_prob, pred_ans_top1_start_logit, pred_ans_top1_end_logit]
                    writer.writerow(res1)
                    try:
                        pred_ans_top2 = pred[qid][1]["text"]
                        pred_ans_top2_prob = pred[qid][1]["probability"]
                        pred_ans_top2_start_logit = pred[qid][1]["start_logit"]
                        pred_ans_top2_end_logit = pred[qid][1]["end_logit"]
                        label_top2 = 1 if pred_ans_top2 == answer else 0
                        res2 = [label_top2, question, pred_ans_top2, qid + "_2",
                                pred_ans_top2_prob, pred_ans_top2_start_logit, pred_ans_top2_end_logit]
                        writer.writerow(res2)

                        pred_ans_top3 = pred[qid][2]["text"]
                        pred_ans_top3_prob = pred[qid][2]["probability"]
                        pred_ans_top3_start_logit = pred[qid][2]["start_logit"]
                        pred_ans_top3_end_logit = pred[qid][2]["end_logit"]
                        label_top3 = 1 if pred_ans_top3 == answer else 0
                        res3 = [label_top3, question, pred_ans_top3, qid + "_3",
                                pred_ans_top3_prob, pred_ans_top3_start_logit, pred_ans_top3_end_logit]
                        writer.writerow(res3)

                    except Exception as e:
                        logger.info(qid, e)


def create_feature_map(file_name):
    # df = pd.read_csv("xgb_data2/valid.csv", header=0, sep="\t")
    # features = df.columns.to_list()[3:]
    # logger.info(features)
    # with open(file_name, 'w', encoding="utf-8") as outfile:
    #     for i, feat in enumerate(features):
    #         outfile.write('f{0}\t{1}\n'.format(i, feat))
    df = pd.read_csv("xgb_data2/valid_add_leijia_prob_ziwei.csv", header=0, sep="\t")
    del df["query"]
    del df["Check_pos"]
    features = df.columns.to_list()[3:]
    logger.info(features)
    with open(file_name, 'w', encoding="utf-8") as outfile:
        for i, feat in enumerate(features):
            outfile.write('f{0}\t{1}\n'.format(i, feat))


def add_leijia_prob(input_file, output_file):
    df = pd.read_csv(input_file, sep="\t", encoding="utf-8", header=0)
    count = 0

    def get_leijia_prob(row):
        nonlocal count
        count += 1
        prob = 0
        for model_id in ensemble_list:
            for epoch in range(1, 6):
                column_name = "{}_{}".format(model_id, epoch)
                prob += row[column_name]
        print(count)
        return prob

    df["leijia_prob"] = df.apply(lambda row: get_leijia_prob(row), axis=1)
    df.to_csv(output_file, sep="\t", encoding="utf-8", header=True, index=False)


def add_ziwei_feature(input_file, output_file, qid_query_file):
    dev_qid_query_json = json.load(open(qid_query_file))

    def get_query(qid):
        return dev_qid_query_json[qid]

    def get_length(answer):
        """
            这个用来检测answer的长度
            Inputs: 答案文本
            Outputs: 一个标量
        """
        return len(str(answer))

    def check_punct(answer):
        '''
            这个用来检测答案中是否包含标点符号。
            因为有的答案是两个半个的句子拼在了一起，不通顺
            比如：
                “双水解,而是沉淀”
            像 "《"，"（"，"）"，这种不予考虑
            我们目前只考虑：中英文句号，逗号，分号， 惊叹号， 问号
            Inputs: 答案文本
            Outputs: 一个标量 （0：不包含标点；1：句中包括标点；2：句子首尾包含标点）
        '''
        answer = str(answer).strip()

        if answer[0] in punct or answer[-1] in punct:
            return 2
        for c in answer[1:-1]:
            if c in punct:
                return 1

        return 0

    def QAOverlap1(question, answer):
        '''
            这个用来检测 Question 和 Answer的相似度
            有的case中有这种情况存在：
                Q:动感单车哪个牌子好
                A:动感单车哪个牌子好，动感单车品牌有蓝堡
            在计算重叠度前需要去掉所有空格并且做小写操作（暂时不去停用词，因为没有一个特别好的停用词表）
            Inputs: 问题文本，答案文本
            Outputs: 重叠长度/min(答案长度,问题长度) (重复单词只算一次（set）)
        '''
        answer = str(answer)
        question = "".join(question.strip().split())  # 去空格
        question = set(tokenizer.tokenize(question))

        answer = "".join(answer.strip().split())  # 去空格
        answer = set(tokenizer.tokenize(answer))

        return len(question & answer) / min(len(answer), len(question))

    def QAOverlap2(question, answer):
        """
            和上面的区别是不删除空格
        """
        answer = str(answer)
        question = set(tokenizer.tokenize(question))
        answer = set(tokenizer.tokenize(answer))

        return len(question & answer) / min(len(answer), len(question))

    def Check_pos(answer):
        """
            检查answer中有哪些词（动词/名词/介词等等）
            符号和词性对应关系可见 https://blog.csdn.net/enter89/article/details/80619805
            Inputs: 答案文本
            Output: List，每个元素代表输入文本中是否有pos_list对应位置的词性，有为1，没有为0
        """
        answer = str(answer)
        words = psg.cut(answer)
        Has_pos = [0] * (len(pos_list) + 1)

        for word, pos in words:
            Has_pos[pos_dict.get(pos, len(pos_list))] = 1

        # return Has_pos
        return "-".join(list(map(str, Has_pos)))

    df = pd.read_csv(input_file, sep="\t", encoding="utf-8", header=0)
    df["query"] = df.apply(lambda row: get_query(row["qid"]), axis=1)
    df["get_length"] = df.apply(lambda row: get_length(row["pred_answer"]), axis=1)
    df["check_punct"] = df.apply(lambda row: check_punct(row["pred_answer"]), axis=1)
    df["QAOverlap1"] = df.apply(lambda row: QAOverlap1(row["query"], row["pred_answer"]), axis=1)
    df["QAOverlap2"] = df.apply(lambda row: QAOverlap2(row["query"], row["pred_answer"]), axis=1)
    df["Check_pos"] = df.apply(lambda row: Check_pos(row["pred_answer"]), axis=1)
    for i, pos in enumerate(pos_list):
        print(i, pos)
        df[pos] = df["Check_pos"].apply(lambda x: x.split('-')[i])

    df.to_csv(output_file, sep="\t", encoding="utf-8", header=True, index=False)


def gen_x_y_csv(input_file, x_output_file, y_output_file):
    df = pd.read_csv(input_file, sep="\t", encoding="utf-8", header=0)
    df.to_csv(y_output_file, sep="\t", encoding="utf-8", header=False, index=False, columns=["label"])
    del df["qid"]
    del df["query"]
    del df["pred_answer"]
    del df["label"]
    del df["Check_pos"]
    df.to_csv(x_output_file, sep="\t", encoding="utf-8", header=False, index=False)


if __name__ == "__main__":
    # gen_qa_data_from_nbest_file(
    #     "../dureader_robust-data/train.json",
    #     "../output_roberta_utf8/2_5_train_nbest_predictions_utf8.json",
    #     "qa_data/roberta_2_5_qa_top3_for_xgb/qa_top3_for_xgb_train.tsv")
    # gen_qa_data_from_nbest_file(
    #     "../dureader_robust-data/dev.json",
    #     "../output_roberta_utf8/2_5_dev_nbest_predictions_utf8.json",
    #     "qa_data/roberta_2_5_qa_top3_for_xgb/qa_top3_for_xgb_dev.tsv")
    # gen_qa_data_from_nbest_file(
    #     "../dureader_robust-test1/test1.json",
    #     "../output_roberta_utf8/2_5_test1_nbest_predictions_utf8.json",
    #     "qa_data/roberta_2_5_qa_top3_for_xgb/qa_top3_for_xgb_test1.tsv")
    # get_all_cands()
    # get_model_prob_train()
    # get_model_prob_valid()
    # create_feature_map("xgb_data2/feature_map")
    # get_all_cands_test1()
    # get_model_prob_test1()
    # add_leijia_prob("xgb_data2/train.csv", "xgb_data2/train_add_leijia_prob.csv")
    # add_ziwei_feature("xgb_data2/train_add_leijia_prob.csv", "xgb_data2/train_add_leijia_prob_ziwei.csv", "../dureader_robust-data/dev_qid_query.json")
    # gen_x_y_csv("xgb_data2/train_add_leijia_prob_ziwei.csv", "xgb_data2/x_train_add_leijia_prob_ziwei.csv", "xgb_data2/y_train_add_leijia_prob_ziwei.csv")
    # add_leijia_prob("xgb_data2/valid.csv", "xgb_data2/valid_add_leijia_prob.csv")
    # add_ziwei_feature("xgb_data2/valid_add_leijia_prob.csv", "xgb_data2/valid_add_leijia_prob_ziwei.csv", "../dureader_robust-data/dev_qid_query.json")
    # gen_x_y_csv("xgb_data2/valid_add_leijia_prob_ziwei.csv", "xgb_data2/x_valid_add_leijia_prob_ziwei.csv", "xgb_data2/y_valid_add_leijia_prob_ziwei.csv")
    # add_leijia_prob("xgb_data2/test1_for_train.csv", "xgb_data2/test1_add_leijia_prob.csv")
    # add_ziwei_feature("xgb_data2/test1_add_leijia_prob.csv", "xgb_data2/test1_add_leijia_prob_ziwei.csv", "../dureader_robust-test1/test1_qid_query.json")
    # gen_x_y_csv("xgb_data2/test1_add_leijia_prob_ziwei.csv", "xgb_data2/x_test_add_leijia_prob_ziwei.csv", "xgb_data2/y_test_add_leijia_prob_ziwei.csv")
    # create_feature_map("xgb_data2/add_leijia_prob_ziwei_feature_map")

    # get_all_cands_data3()
    # get_model_prob_train()
    # add_leijia_prob("xgb_data3/train.csv", "xgb_data3/train_add_leijia_prob.csv")
    # add_ziwei_feature("xgb_data3/train_add_leijia_prob.csv", "xgb_data3/train_add_leijia_prob_ziwei.csv", "../dureader_robust-data/dev_qid_query.json")
    # gen_x_y_csv("xgb_data3/train_add_leijia_prob_ziwei.csv", "xgb_data3/x_train_add_leijia_prob_ziwei.csv", "xgb_data3/y_train_add_leijia_prob_ziwei.csv")

    get_all_cands_test2()
    get_model_prob_test2()
