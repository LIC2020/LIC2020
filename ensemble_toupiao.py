# -*- coding: utf-8 -*-
import json
import os
import sys
from itertools import combinations
from collections import OrderedDict

__author__ = "liuaiting@bupt.edu.cn"

# dev_0 = json.load(open("output_utf8/0_dev_nbest_predictions_utf8.json"))
# dev_1 = json.load(open("output_utf8/1_dev_nbest_predictions_utf8.json"))
# dev_2 = json.load(open("output_utf8/2_dev_nbest_predictions_utf8.json"))
# dev_3 = json.load(open("output_utf8/3_dev_nbest_predictions_utf8.json"))
# dev_4 = json.load(open("output_utf8/4_dev_nbest_predictions_utf8.json"))
# dev_5 = json.load(open("output_utf8/5_dev_nbest_predictions_utf8.json"))
# dev_6 = json.load(open("output_utf8/6_dev_nbest_predictions_utf8.json"))
# dev_7 = json.load(open("output_utf8/7_dev_nbest_predictions_utf8.json"))
# dev_8 = json.load(open("output_utf8/8_dev_nbest_predictions_utf8.json"))
# dev_9 = json.load(open("output_utf8/9_dev_nbest_predictions_utf8.json"))
# dev_10 = json.load(open("output_utf8/10_dev_nbest_predictions_utf8.json"))
# dev_11 = json.load(open("output_utf8/11_dev_baidushort_cmrc_lic_nbest_predictions_utf8.json"))
# dev_12 = json.load(open("output_utf8/12_dev_cmrc_baidushort_lic_nbest_predictions_utf8.json"))

# dev_res = [dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_6, dev_7, dev_8, dev_9, dev_10, dev_11, dev_12]

# test1_0 = json.load(open("output_utf8/0_test1_nbest_predictions_utf8.json"))
# test1_1 = json.load(open("output_utf8/1_test1_nbest_predictions_utf8.json"))
# test1_2 = json.load(open("output_utf8/2_test1_nbest_predictions_utf8.json"))
# test1_3 = json.load(open("output_utf8/3_test1_nbest_predictions_utf8.json"))
# test1_4 = json.load(open("output_utf8/4_test1_nbest_predictions_utf8.json"))
# test1_5 = json.load(open("output_utf8/5_test1_nbest_predictions_utf8.json"))
# test1_6 = json.load(open("output_utf8/6_test1_nbest_predictions_utf8.json"))
# test1_7 = json.load(open("output_utf8/7_test1_nbest_predictions_utf8.json"))
# test1_8 = json.load(open("output_utf8/8_test1_nbest_predictions_utf8.json"))
# test1_9 = json.load(open("output_utf8/9_test1_split_nbest_predictions_utf8.json"))
# test1_10 = json.load(open("output_utf8/10_test1_nbest_predictions_utf8.json"))
# test1_11 = json.load(open("output_utf8/11_test1_baidushort_cmrc_lic_nbest_predictions_utf8.json"))
# test1_12 = json.load(open("output_utf8/12_test1_cmrc_baidushort_lic_nbest_predictions_utf8.json"))

# test1_res = [test1_0, test1_1, test1_2, test1_3, test1_4, test1_5, test1_6, test1_7, test1_8, test1_9, test1_10,
#              test1_11, test1_12]


def toupiao(output_file, name="dev"):
    """
    投票，排名作为得分进行加权
    name: dev 或 test1，用于读取相应结果文件，也可以用别的规则读文件
    nbest=10的话，就是从10分-1分
    """
    ensemble_list = ["14", "17", "21", "22", "23", "25", "26", "27", "28", "29", "33", "34", "35", "36", "37", "38", "39"]
    # ensemble_list = ["14_3", "17_2", "17_3", "21_5", "22_5", "23_3", "23_4", "25_4", "26_5", "27_5", "28_5", "29_4", "33_5", "34_5", "35_5", "36_5", "37_5"]

    files = os.listdir("output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_{}_nbest".format(name) in file and file.split("_")[0] in ensemble_list:
        # if "lic_dev_nbest" in file and "_".join(file.split("_")[0:2]) in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    print("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    print("pred_files: 共{}个nbest文件.".format(len(pred_files)))
    res_list = [json.load(open("output_data_join_utf8/" + pred_file)) for pred_file in pred_files]

    try:
        res_json = {}
        for k in list(res_list[0].keys()):
            text_list = {}
            for i in range(len(res_list)):
                for j in range(len(res_list[i][k])):
                    text = res_list[i][k][j]["text"]
                    prob = res_list[i][k][j]["probability"]
                    score = 10 - j    # 得分从10分-1分
                    if not text_list.get(text):
                        # TODO(aitingliu): start_logit 和 end_logit也可以加进来，看看有没有效果增强
                        text_list[text] = 1 * score
                    else:
                        text_list[text] += 1 * score
            # print(text_list)
            # print(sorted(text_list.items(), key=lambda d: d[1], reverse=True))
            res_json[k] = sorted(text_list.items(), key=lambda d: d[1], reverse=True)[0][0]
        json.dump(res_json, open(output_file, "w"), ensure_ascii=False, indent=4)
    except Exception as e1:
        print("err1: ", output_file, e1)


def toupiao2(output_file, name="test1", n=2):
    """
    投票，排名作为得分进行加权，得分截断式衰减
    name: dev 或 test1，用于读取相应结果文件，也可以用别的规则读文件
    n: n=2，表示从第2名（不包括第2名）开始之后得分都是0分
    """
    ensemble_list = ["14", "17", "21", "22", "23", "25", "26", "27", "28", "29", "33", "34", "35", "36", "37", "38", "39"]
    # ensemble_list = ["14_3", "17_2", "17_3", "21_5", "22_5", "23_3", "23_4", "25_4", "26_5", "27_5", "28_5", "29_4", "33_5", "34_5", "35_5", "36_5", "37_5"]

    files = os.listdir("output_data_join_utf8")
    pred_files = []
    for file in files:
        if "lic_{}_nbest".format(name) in file and file.split("_")[0] in ensemble_list:
        # if "lic_dev_nbest" in file and "_".join(file.split("_")[0:2]) in ensemble_list:
            pred_files.append(file)
    pred_files.sort()
    print("ensemble_list: [{}], [{}]".format(len(ensemble_list), ",".join(ensemble_list)))
    print("pred_files: 共{}个nbest文件.".format(len(pred_files)))
    res_list = [json.load(open("output_data_join_utf8/" + pred_file)) for pred_file in pred_files]

    try:  # 有的不够十个候选
        res_json = {}
        for k in list(res_list[0].keys()):
            text_list = {}
            for i in range(len(res_list)):
                for j in range(len(res_list[i][k])):
                    text = res_list[i][k][j]["text"]
                    prob = res_list[i][k][j]["probability"]
                    # TODO(aitingliu): 投票，排名作为得分进行加权
                    # score = 10 - j
                    if j < n:
                        score = 10 - j
                    else:
                        score = 0
                    if not text_list.get(text):
                        text_list[text] = 1 * score
                    else:
                        text_list[text] += 1 * score
            # print(text_list)
            # print(sorted(text_list.items(), key=lambda d: d[1], reverse=True))
            res_json[k] = sorted(text_list.items(), key=lambda d: d[1], reverse=True)[0][0]
        json.dump(res_json, open(output_file, "w"), ensure_ascii=False, indent=4)
    except Exception as e1:
        print("err1: ", output_file, e1)


# def ensemble_v1(res_list, output_file):
#     """
#     集成多个在dev集上的nbest结果。
#     text出现频次 和 probability 加权排序
#     """

#     try:
#         res_json = {}
#         for k in list(res_list[0].keys()):
#             text_list = {}
#             for i in range(len(res_list)):
#                 for j in range(len(res_list[i][k])):
#                     text = res_list[i][k][j]["text"]
#                     prob = res_list[i][k][j]["probability"]
#                     if not text_list.get(text):
#                         # TODO(aitingliu): start_logit 和 end_logit也可以加进来，看看有没有效果增强
#                         text_list[text] = 1 * prob
#                     else:
#                         text_list[text] += 1 * prob
#             # print(text_list)
#             # print(sorted(text_list.items(), key=lambda d: d[1], reverse=True))
#             res_json[k] = sorted(text_list.items(), key=lambda d: d[1], reverse=True)[0][0]
#         json.dump(res_json, open(output_file, "w"), ensure_ascii=False, indent=4)
#     except Exception as e1:
#         print("err1: ", output_file, e1)


# def ensemble_v1_test1(id_list):
#     res_list = [test1_res[i] for i in id_list]
#     version_str = "_".join(list(map(str, id_list)))
#     output_file_path = "ensem/test1_ensemble_v1_{}.json".format(version_str)
#     ensemble_v1(res_list, output_file_path)


# def ensemble_try():
#     f1_dict = {}
#     em_dict = {}

#     # TODO: 删掉8910后 l = [0,1,2,3,4,5,6,7,11,12]
#     l = range(len(dev_res))
#     combs = []
#     for i in range(2, len(l) + 1):
#         combs.extend(list(combinations(l, i)))
#     for id_list in combs:
#         res_list = [dev_res[i] for i in id_list]
#         version_str = "_".join(list(map(str, id_list)))
#         output_file_path = "ensem/dev_ensemble_v1_{}.json".format(version_str)
#         try:
#             ensemble_v1(res_list, output_file_path)
#             cmd = "python evaluate.py dureader_robust-data/dev.json {}".format(output_file_path)
#             tmp = os.popen(cmd).read()
#             print("{}\t{}".format(output_file_path, tmp))
#             f1 = float(json.loads(tmp)["F1"])
#             em = float(json.loads(tmp)["EM"])
#             f1_dict[output_file_path] = f1
#             em_dict[output_file_path] = em
#         except Exception as e2:
#             print("err1: ", output_file_path, e2)

#     max_f1, max_f1_file = sorted(f1_dict.items(), key=lambda d: d[1], reverse=True)[0]
#     max_em, max_em_file = sorted(em_dict.items(), key=lambda d: d[1], reverse=True)[0]
#     print("###################")
#     print("max_f1: {}\t{}".format(max_f1, max_f1_file))
#     print("max_em: {}\t{}".format(max_em, max_em_file))

#     f1_dict = OrderedDict(sorted(f1_dict.items(), key=lambda d: d[1], reverse=True))
#     em_dict = OrderedDict(sorted(em_dict.items(), key=lambda d: d[1], reverse=True))
#     json.dump(f1_dict, open("ensemv1/f1_ensemble_v1.json", "w"), ensure_ascii=False, indent=4)
#     json.dump(em_dict, open("ensemv1/em_ensemble_v1.json", "w"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # ensemble_try()
    # ensemble_v1_test1([1, 2, 3, 11, 12])
    # ensemble_v1_test1([2, 3, 6, 7, 11, 12])
    # ensemble_v1_test1([3, 7, 11, 12])
    # ensemble_v1_test1([3, 6, 11, 12])

    toupiao2("dev_toupiao2.json", "dev")
    toupiao2("test1_toupiao2.json", "test1")




