# -*- coding: utf-8 -*-
import json
import os

__author__ = "liuaiting@bupt.edu.cn"


def pred_to_utf8(input_file):
    print("input_file :\t", input_file)
    a = json.load(open(input_file))
    dir_name, file_name = os.path.split(input_file)
    output_dir_name = dir_name + "_utf8"
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    file_name_list = file_name.split(".")
    output_file_name = file_name_list[0] + "_utf8." + file_name_list[1]
    output_file = os.path.join(output_dir_name, output_file_name)
    json.dump(a, open(output_file, "w"), ensure_ascii=False, indent=4)
    print("output_file:\t", output_file)


def json_to_utf8(input_file):
    a = json.load(open(input_file))
    dir_name, file_name = os.path.split(input_file)
    output_file = os.path.join(dir_name, "utf8_" + file_name)
    json.dump(a, open(output_file, "w"), ensure_ascii=False, indent=2)


def keep_top10(input_file, output_file):
    a = json.load(open(input_file))
    for k in a.keys():
        a[k] = a[k][:10]
    json.dump(a, open(output_file, "w"), ensure_ascii=False, indent=4)


def diff(file1, file2, diff_file):
    f1 = json.load(open(file1))
    f2 = json.load(open(file2))
    with open(diff_file, "w") as f:
        for k in f1.keys():
            if f1[k] != f2[k]:
                res = "{}\t{}\t{}\n".format(k, f1[k], f2[k])
                f.write(res)


def merge_predict():
    # part1 = json.load(open("output_data_join_utf8/5_1_baidushort+cmrc_lic_train_baidushort_part1_predictions_utf8.json"))
    # part2 = json.load(open("output_data_join_utf8/5_1_baidushort+cmrc_lic_train_baidushort_part2_predictions_utf8.json"))
    # part1.update(part2)
    # json.dump(part1, open("output_data_join_utf8/5_1_baidushort+cmrc_lic_train_baidushort_predictions_utf8.json", "w"),
    #           ensure_ascii=False, indent=2)

    part1 = json.load(open("output_data_join_utf8/5_1_baidushort+cmrc_lic_train_baidushort_part1_nbest_predictions_utf8.json"))
    part2 = json.load(open("output_data_join_utf8/5_1_baidushort+cmrc_lic_train_baidushort_part2_nbest_predictions_utf8.json"))
    part1.update(part2)
    json.dump(part1, open("output_data_join_utf8/5_1_baidushort+cmrc_lic_train_baidushort_nbest_predictions_utf8.json", "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # pred_to_utf8("output/11_dev_predictions.json")
    # pred_to_utf8("output/11_dev_nbest_predictions.json")
    # pred_to_utf8("output/11_test1_predictions.json")
    # pred_to_utf8("output/11_test1_nbest_predictions.json")

    # pred_to_utf8("output_albert_xlarge/4_1_dev_predictions.json")
    # pred_to_utf8("output_albert_xlarge/4_1_dev_nbest_predictions.json")
    # pred_to_utf8("output_albert_xlarge/4_1_test1_predictions.json")
    # pred_to_utf8("output_albert_xlarge/4_1_test1_nbest_predictions.json")

    # pred_to_utf8("output_albert_xxlarge/2_1_dev_predictions.json")
    # pred_to_utf8("output_albert_xxlarge/2_1_dev_nbest_predictions.json")
    # pred_to_utf8("output_albert_xxlarge/2_1_test1_predictions.json")
    # pred_to_utf8("output_albert_xxlarge/2_1_test1_nbest_predictions.json")

    # pred_to_utf8("output/11_dev_baidushort_cmrc_lic_predictions.json")
    # pred_to_utf8("output/11_dev_baidushort_cmrc_lic_nbest_predictions.json")
    # pred_to_utf8("output/11_test1_baidushort_cmrc_lic_predictions.json")
    # pred_to_utf8("output/11_test1_baidushort_cmrc_lic_nbest_predictions.json")

    # pred_to_utf8("output_bert/3_1_dev_predictions.json")
    # pred_to_utf8("output_bert/3_1_dev_nbest_predictions.json")
    # pred_to_utf8("output_bert/3_1_test1_predictions.json")
    # pred_to_utf8("output_bert/3_1_test1_nbest_predictions.json")

    # pred_to_utf8("output_roberta/12_5_dev_predictions.json")
    # pred_to_utf8("output_roberta/12_5_dev_nbest_predictions.json")
    # pred_to_utf8("output_roberta/12_5_test1_predictions.json")
    # pred_to_utf8("output_roberta/12_5_test1_nbest_predictions.json")
    # pred_to_utf8("output_roberta/12_5_train_predictions.json")
    # pred_to_utf8("output_roberta/12_5_train_nbest_predictions.json")
    # pred_to_utf8("output_roberta/12_5_train_lic_part2_predictions.json")
    # pred_to_utf8("output_roberta/12_5_train_lic_part2_nbest_predictions.json")

    # pred_to_utf8("output_data_join/40_1_dev_predictions.json")
    # pred_to_utf8("output_data_join/40_1_dev_nbest_predictions.json")
    # pred_to_utf8("output_data_join/40_1_test1_predictions.json")
    # pred_to_utf8("output_data_join/40_1_test1_nbest_predictions.json")
    # pred_to_utf8("output_data_join/40_1_train_lic_predictions.json")
    # pred_to_utf8("output_data_join/40_1_train_lic_nbest_predictions.json")

    # pred_to_utf8("output_data_join/27_5_dev_predictions_40.json")
    # pred_to_utf8("output_data_join/27_5_dev_nbest_predictions_40.json")
    pred_to_utf8("output_data_join/27_5_test1_predictions_40.json")
    pred_to_utf8("output_data_join/27_5_test1_nbest_predictions_40.json")
    
    # merge_predict()

    # json_to_utf8("dureader_new/dureader_search_new.train.json")
    # json_to_utf8("dureader_new/dureader_zhidao_new.train.json")

    # pred_to_utf8("output_roberta/4_4_dev_top3_predictions.json")
    # pred_to_utf8("output_roberta/4_4_dev_top3_nbest_predictions.json")

    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_dev_logit_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cail1_lic_dev_logit_predictions.json")
    # pred_to_utf8("output_albert_xxlarge/2_1_dev_logit_predictions.json")
    # pred_to_utf8("output_data_join/27_5_drcd1_lic_dev_logit_predictions.json")
    # pred_to_utf8("output_data_join/18_3_cmrc1_lic_dev_logit_predictions.json")
    # pred_to_utf8("output_data_join/17_2_baidushort2_cmrc2_drcd2_cail1_lic_dev_logit_predictions.json")
    # pred_to_utf8("output_data_join/8_3_baidushort_cmrc_lic_dev_logit_predictions.json")
    # pred_to_utf8("output_roberta/2_5_dev_logit_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_dev_logit_predictions.json")
    # pred_to_utf8("output_data_join/14_3_baidushort_cmrc_drcd_cail_cetc1500_lic_dev_logit_predictions.json")

    # pred_to_utf8("output_roberta/2_5_lic_extended_v4_train_nbest_predictions.json")
    # pred_to_utf8("output_roberta/2_5_lic_extended_v4_train_predictions.json")

    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_cail_nbest_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_cail_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_drcd_nbest_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_drcd_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_cmrc_nbest_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_cmrc_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_lic_nbest_predictions.json")
    # pred_to_utf8("output_data_join/27_5_cmrc1_drcd1_cail1_lic_train_lic_predictions.json")
