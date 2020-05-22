# -*- coding: utf-8 -*-
import json
from matplotlib import pyplot as plt
import os

__author__ = "liuaiting@bupt.edu.cn"


def main1(input_file, name):
    print(input_file)
    input_path, input_file_name = os.path.split(input_file)
    fig_output_path = os.path.join(input_path, "fig")
    if not os.path.exists(fig_output_path):
        os.mkdir(fig_output_path)
    q_len_count = dict()
    a_len_count = dict()
    d_len_count = dict()
    f_json = json.load(open(input_file))
    for data in f_json['data']:
        for para in data['paragraphs']:
            doc = para["context"]
            d_l = len(doc)
            for qa in para['qas']:
                query = qa["question"]
                q_l = len(query)
                for ans in qa["answers"]:
                    answer = ans["text"]
                    a_l = len(answer)

                    if not q_len_count.get(q_l):
                        q_len_count[q_l] = 1
                    else:
                        q_len_count[q_l] += 1
                    if not a_len_count.get(a_l):
                        a_len_count[a_l] = 1
                    else:
                        a_len_count[a_l] += 1
                    if not d_len_count.get(d_l):
                        d_len_count[d_l] = 1
                    else:
                        d_len_count[d_l] += 1

    # 统计query长度平均值
    s = 0
    for k, v in q_len_count.items():
        s += k * v
    print("统计query长度平均值：{}, 最大值：{}".format(s / sum(q_len_count.values()), max(list(q_len_count.keys()))))

    # 统计answer长度平均值
    s = 0
    for k, v in a_len_count.items():
        s += k * v
    print("统计answer长度平均值：{}, 最大值：{}".format(s / sum(a_len_count.values()), max(list(a_len_count.keys()))))

    # 统计doc长度平均值
    s = 0
    for k, v in d_len_count.items():
        s += k * v
    print("统计doc长度平均值：{}, 最大值：{}".format(s / sum(d_len_count.values()), max(list(d_len_count.keys()))))

    # 统计query长度分布
    x = sorted(list(q_len_count.keys()))
    y = [q_len_count[k] for k in x]
    leiji = [sum(y[:i + 1]) for i in range(len(x))]
    plt.title("{}_query".format(os.path.split(input_file)[1]), fontsize=24)
    plt.xlabel("#length", fontsize=14)
    plt.ylabel("#count", fontsize=14)
    plt.plot(x, y)
    plt.savefig("{}/{}_{}_query.png".format(fig_output_path, input_file_name, name))
    plt.plot(x, leiji)
    plt.savefig("{}/{}_{}_query_leiji.png".format(fig_output_path, input_file_name, name))
    plt.close()

    # 统计answer长度分布
    x = sorted(list(a_len_count.keys()))
    y = [a_len_count[k] for k in x]
    leiji = [sum(y[:i + 1]) for i in range(len(x))]
    plt.title("{}_answer".format(os.path.split(input_file)[1]), fontsize=24)
    plt.xlabel("#length", fontsize=14)
    plt.ylabel("#count", fontsize=14)
    plt.plot(x, y)
    plt.savefig("{}/{}_{}_answer.png".format(fig_output_path, input_file_name, name))
    plt.plot(x, leiji)
    plt.savefig("{}/{}_{}_answer_leiji.png".format(fig_output_path, input_file_name, name))
    plt.close()

    # 统计doc长度分布
    x = sorted(list(d_len_count.keys()))
    y = [d_len_count[k] for k in x]
    leiji = [sum(y[:i + 1]) for i in range(len(x))]
    plt.title("{}_doc".format(os.path.split(input_file)[1]), fontsize=24)
    plt.xlabel("#length", fontsize=14)
    plt.ylabel("#count", fontsize=14)
    plt.plot(x, y)
    plt.savefig("{}/{}_{}_doc.png".format(fig_output_path, input_file_name, name))
    plt.plot(x, leiji)
    plt.savefig("{}/{}_{}_doc_leiji.png".format(fig_output_path, input_file_name, name))
    plt.close()


def main2(input_file):
    """统计squad-style-data格式的数据集规模，即qa对数量"""
    print(input_file)
    total_count = 0
    neg_count = 0
    f_json = json.load(open(input_file))
    for instance in f_json["data"]:
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                # if qas["answers"] == []:
                #     neg_count += 1
    print("total: {}".format(total_count))


if __name__ == "__main__":
    # main1("dureader_robust-data/train.json", "train")
    # main2("dureader_robust-data/train.json")  # 14520
    # main2("dureader_robust-data/dev.json")  # 1417
    # main2("dureader_robust-test1/test1.json")  # 50000
    # main2("data_join/train_baidushort_cmrc.json")  # 87884
    # main2("cmrc2018/cmrc2018_train_keep_1ans.json")  # 10142
    # main2("cmrc2018/cmrc2018_dev_keep_1ans.json")  # 3219
    # main2("cmrc2018/cmrc2018_trial_keep_1ans.json")  # 1002
    # main2("baidu_short/train_for_lic.json")  # 48526
    # main2("cetc/train_lic_format.json")  # 77873
    # main2("cetc/train_lic_format_500_30.json")  # 24054
    # main2("cetc/train_lic_format_1000_30.json")  # 46502
    # main2("cetc/train_lic_format_1500_30.json")  # 56376
    # main2("data_join/train_cetc500_baidushort_cmrc_lic.json")  # 101463
    # main2("data_join/train_cetc500_baidushort_cmrc.json")  # 86943
    # main2("data_join/train_baidushort_cmrc_lic.json")  # 77409
    # main2("data_join/train_baidushort_cmrc.json")  # 62889
    # main2("data_join/train_baidushort.json")  # 48526
    # main2("data_join/train_cmrc.json")  # 14363
    # main2("data_join/train_cetc1500.json")  # 56376
    # main2("data_join/train_lic.json")  # 14520
    # main2("DRCD_new/DRCD_new.json")  # 26936
    # main2("CAIL_new/CAIL_new.json")  # 32991
    # main2("dureader_new/dureader_zhidao_new.train.json")  # 30263
    # main2("dureader_new/dureader_search_new.train.json")  # 17795
    # main2("dureader_robust-data/train_extended_v1.json")  # 31745
    # main2("dureader_robust-data/train_extended_v5.json")  # 31745
    # main2("dureader_robust-test2/test2.json")
    main2("data_join/train_webqa.json")  # 146939

    # main1("data_join/train_drcd.json", "TRAIN")
    # main1("data_join/train_cmrc.json", "TRAIN")
    # main1("data_join/train_lic.json", "TRAIN")
    # main1("data_join/train_baidushort.json", "TRAIN")
    # main1("data_join/train_dureader.json", "TRAIN")
    # main1("data_join/train_cetc1500.json", "TRAIN")
    # main1("dureader_robust-data/dev.json", "DEV")
    # main1("dureader_robust-data/train.json", "TRAIN")

