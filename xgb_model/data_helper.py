# -*- coding: utf-8 -*-
import json
import csv
import sys
import pandas as pd
import math

__author__ = "liuaiting@bupt.edu.cn"

csv.field_size_limit(sys.maxsize)


def gen_pair_data_for_bert(input_file, pred_file, output_file):
    f = json.load(open(input_file))
    pred = json.load(open(pred_file))
    with open(output_file, "w") as fw:
        writer = csv.writer(fw, delimiter="\t")
        writer.writerow(["label", "query", "answer", "qid"])
        for data in f["data"]:
            for para in data["paragraphs"]:
                for qas in para["qas"]:
                    qid = qas["id"]
                    question = qas["question"]
                    answer = qas["answers"][0]["text"] if qas.get("answers") else ""
                    try:
                        pred_ans_top1 = pred[qid][0]["text"]
                        label_top1 = 1 if pred_ans_top1 == answer else 0
                        res1 = [label_top1, question, pred_ans_top1, qid+"_1"]
                        writer.writerow(res1)

                        pred_ans_top2 = pred[qid][1]["text"]
                        label_top2 = 1 if pred_ans_top2 == answer else 0
                        res2 = [label_top2, question, pred_ans_top2, qid+"_2"]
                        writer.writerow(res2)

                        pred_ans_top3 = pred[qid][2]["text"]
                        label_top3 = 1 if pred_ans_top3 == answer else 0
                        res3 = [label_top3, question, pred_ans_top3, qid+"_3"]
                        writer.writerow(res3)

                    except Exception as e:
                        print(qid, e)


def check_csv():
    with open("../xgb_model/qa_data/roberta_2_5_qa_top3/test.tsv") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 4:
                print(row)


def get_top3_rerank_res(pred_file, pred_file_top1, output_file):
    pred = json.load(open(pred_file))
    df = pd.read_csv(pred_file_top1, sep="\t")
    res_json = {}
    i, j = 0, 0
    for qid in pred.keys():
        try:
            res_json[qid] = df.loc[df["para_id"] == qid]["answer"].to_list()[0]
            i += 1
        except Exception as e:
            # print(qid, e)
            j += 1
            res_json[qid] = pred[qid][0]["text"]
    json.dump(res_json, open(output_file, "w"),
              ensure_ascii=False, indent=4)
    print(i, j)


def get_qid_query_json():
    f = json.load(open("../dureader_robust-test1/test1.json"))
    dev_qid_query_json = {}
    for data in f["data"]:
        for para in data["paragraphs"]:
            for qas in para["qas"]:
                qid = qas["id"]
                question = qas["question"]
                dev_qid_query_json[qid] = question
    json.dump(dev_qid_query_json, open("../dureader_robust-test1/test1_qid_query.json", "w"),
              ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # gen_pair_data_for_bert("../dureader_robust-data/train.json",
    #                        "../output_roberta_utf8/2_5_train_nbest_predictions_utf8.json",
    #                        "qa_data/roberta_2_5_qa_top3/train.tsv")
    # gen_pair_data_for_bert("../dureader_robust-data/dev.json",
    #                        "../output_roberta_utf8/2_5_dev_nbest_predictions_utf8.json",
    #                        "qa_data/roberta_2_5_qa_top3/dev.tsv")
    # gen_pair_data_for_bert("../dureader_robust-test1/test1.json",
    #                        "../output_roberta_utf8/2_5_test1_nbest_predictions_utf8.json",
    #                        "qa_data/roberta_2_5_qa_top3/test.tsv")

    # get_top3_rerank_res("../output_roberta_utf8/2_5_test1_nbest_predictions_utf8.json",
    #                     "qa_data/roberta_2_5_qa_top3/test1_test_results.tsv_pred_top1",
    #                     "qa_data/roberta_2_5_qa_top3/2_5_test1_predictions_utf8_rerank.json")

    # get_top3_rerank_res("../output_roberta_utf8/2_5_dev_nbest_predictions_utf8.json",
    #                     "qa_data/roberta_2_5_qa_top3/dev_test_results.tsv_pred_top1",
    #                     "qa_data/roberta_2_5_qa_top3/2_5_dev_predictions_utf8_rerank.json")
    get_qid_query_json()
