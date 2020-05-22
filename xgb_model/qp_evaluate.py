#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import csv
import json

__author__ = "liuaiting@bupt.edu.cn"


def metric(input_path, predict_path, threshold, nbest_file):
    """
    pair级别指标.
    [test data format]: label\tquery\tpara\tqid
    [prediction file format]: 0_prob\t1_prob
    :param input_path:
    :param threshold:
    :return:
    """
    input_df = pd.read_csv(input_path, sep='\t', encoding="utf-8", header=0)
    pred_df = pd.read_csv(predict_path, sep='\t', encoding="utf-8", header=None, names=["p0", "p1"])
    df = pd.concat([input_df, pred_df], axis=1)
    # print(input_df.tail())
    # print(pred_df.tail())
    # print(df.tail())

    df.loc[df.p1 > threshold, 'pred'] = 1
    df.loc[df.p1 <= threshold, 'pred'] = 0
    df["pred"] = df["pred"].astype("int")

    def get_para_id(qid):
        para_id = qid.split("_")[0]
        return para_id

    df["para_id"] = df.qid.map(lambda x: get_para_id(x))

    pred = json.load(open(nbest_file))

    def get_probability(qid, answer):
        para_id, i = qid.split("_")[0], int(qid.split("_")[1])
        if pred[para_id][i - 1]["text"] == answer:
            return pred[para_id][i]["probability"]
        else:
            print(qid)
            return 1.0

    def get_p1_probability(p1, probability):
        return p1 * probability

    df["probability"] = df.apply(lambda row: get_probability(row["qid"], row["answer"]), axis=1)
    df["p1*probability"] = df.apply(lambda row: get_p1_probability(row["p1"], row["probability"]), axis=1)

    print(input_path, predict_path, threshold)
    print("Pair级别结果指标")
    print("THRESHOLD={} TOTAL={}".format(threshold, len(df) - 1))
    print("ACC : %.4f" % metrics.accuracy_score(df.label, df.pred))
    print("PRE : %.4f" % metrics.precision_score(df.label, df.pred))
    print("REC : %.4f" % metrics.recall_score(df.label, df.pred))
    print("F1  : %.4f" % metrics.f1_score(df.label, df.pred))
    # print("AUC : %.4f" % metrics.roc_auc_score(df.label, df.p1))
    print("\n")
    df.to_csv("{}_pred".format(predict_path), sep="\t", index=False, encoding="utf-8",
              columns=["para_id", "qid", "label", "p1", "probability", "p1*probability", "query", "answer"])

    # plt.figure()
    # plt.title('Precision/Recall Curve')  # give plot a title
    # plt.xlabel('Recall')  # make axis labels
    # plt.ylabel('Precision')
    # # y_true和y_scores分别是gt label和predict score
    # y_true = df.label
    # y_scores = df.p1
    # precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    # plt.plot(precision, recall)
    # # plt.show()
    # plt.savefig("{}_pr.png".format(input_path))
    # plt.close()


def total_metric(input_path, output_path, threshold):
    """
    [deprecated]
    Query级别指标.
    [input_format]: para_id\tqid\tlabel\tp1\tquery\tpara
    :param input_path:
    :param output_path:
    :param threshold:
    :return:
    """
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    d = {}
    for line in open(input_path, "r", encoding="utf-8"):
        para_id, qid, label, predict, query, para = line.strip().split("\t")
        if para_id == "para_id":
            continue
        d.setdefault(para_id, [])
        d[para_id].append((para, label, predict))

    labels = []
    scores = []
    pred_labels = []

    f = open(output_path, "w", encoding="utf-8")
    P, N, TP, FN, FP, TN = 0, 0, 0, 0, 0, 0
    for para_id in d:
        terms = d[para_id]

        sort_l = [(float(predict), (para, label, predict)) for para, label, predict in
                  terms]  # 针对每一个query的所有question预测结果排序

        sort_l.sort(reverse=True)
        terms_str = "\t".join(["-".join(list(term[1])) for term in sort_l])
        # terms_str = str(sort_l)

        best_term = sort_l[0][1]
        q2, label, predict = best_term
        # print(best_term)

        predict_label = 1 if float(predict) > threshold else 0  # best answer的结果 是否过阈值
        # predict_answer = predict_label * int(label) # 判定best answer预测结果是否正确
        has_answer = int(any([int(label) for para, label, predict in terms]))  # 判定候选question是否有正确答案
        # print(has_answer)
        # predict_answer has_answer  q1 q2_list 作为最终的判定序列
        f.write("\t".join([str(has_answer), label, str(predict_label), predict, para_id, terms_str]) + "\n")

        label = int(label)
        if has_answer == 1: P += 1  # 正类 候选包含正确答案
        if has_answer == 0: N += 1  # 负类 候选不包含正确答案
        if label == 1 and predict_label == 1: TP += 1  # 将正类预测为正类数
        if label == 1 and predict_label == 0: FN += 1  # 将正类预测为负类数
        if label == 0 and predict_label == 1: FP += 1  # 将负类预测为正类数
        if label == 0 and predict_label == 0: TN += 1  # 将负类预测为负类数

        labels.append(label)
        scores.append(predict)
        pred_labels.append(predict_label)

    PRE = TP / (TP + FP + 0.00000001)
    REC = TP / (P + 0.00000001)
    ACC = (TP + TN) / (P + N + 0.00000001)
    F1 = 2 * PRE * REC / (PRE + REC + 0.00000001)

    print(input_path, output_path, threshold)
    print("Query级别结果指标")
    print("THRESHOLD={} P={} N={} TP={} FN={} FP={} TN={}".format(threshold, P, N, TP, FN, FP, TN))
    print("ACC : %.4f" % ACC)
    print("PRE : %.4f" % PRE)
    print("REC : %.4f" % REC)
    print("F1  : %.4f" % F1)
    print("\n")


def doc_metric(input_path, threshold):
    """
    Query-DOC级别指标.
    [input_format]: para_id\tqid\tlabel\tp1\tquery\tpara
    [output_format]: para_id
    :param input_path:
    :param threshold:
    :return:
    """
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8", header=0)
    df.loc[df.p1 > threshold, 'pred'] = 1
    df.loc[df.p1 <= threshold, 'pred'] = 0
    df["pred"] = df["pred"].astype("int")

    g_df_label = df.groupby("para_id")["label"].max()
    g_df_pred = df.groupby("para_id")["pred"].max()
    g_df_p1 = df.groupby("para_id")["p1"].max()
    g_df = pd.concat([g_df_label, g_df_pred, g_df_p1], axis=1)
    # print(g_df.head())

    print(input_path, threshold)
    print("Query-DOC级别结果指标")
    print("THRESHOLD={} TOTAL={}".format(threshold, len(g_df_label) - 1))
    print("ACC : %.4f" % metrics.accuracy_score(g_df_label, g_df_pred))
    print("PRE : %.4f" % metrics.precision_score(g_df_label, g_df_pred))
    print("REC : %.4f" % metrics.recall_score(g_df_label, g_df_pred))
    print("F1  : %.4f" % metrics.f1_score(g_df_label, g_df_pred))
    print("AUC : %.4f" % metrics.roc_auc_score(g_df_label, g_df_p1))
    print("\n")
    g_df.to_csv("{}_doc".format(input_path), sep="\t", header=True, index=True, encoding="utf-8")


def get_top1(pred_file, pred_file_top1):
    df = pd.read_csv(pred_file, sep="\t", encoding="utf-8")
    a = df.groupby("para_id")["p1*probability"].max()
    res_df = pd.merge(
        df,
        a,
        on=["para_id", "p1*probability"])
    res_df.to_csv(pred_file_top1, sep="\t", index=False)


if __name__ == "__main__":
    # metric("qa_data/roberta_2_5_qa_top3/test.tsv", "qa_data/roberta_2_5_qa_top3/test1_test_results.tsv", 0.5)
    # get_top1("qa_data/roberta_2_5_qa_top3/test1_test_results.tsv_pred", "qa_data/roberta_2_5_qa_top3/test1_test_results.tsv_pred_top1")
    # metric("qa_data/roberta_2_5_qa_top3/dev.tsv", "qa_data/roberta_2_5_qa_top3/dev_test_results.tsv", 0.5, "../output_roberta_utf8/2_5_dev_nbest_predictions_utf8.json")
    get_top1("qa_data/roberta_2_5_qa_top3/dev_test_results.tsv_pred",
             "qa_data/roberta_2_5_qa_top3/dev_test_results.tsv_pred_top1")
