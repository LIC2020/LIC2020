#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import logging
import json

import pandas as pd
import xgboost as xgb

import common_function

__author__ = "liuaiting@bupt.edu.cn"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--test_data_file", type=str, default='xgb_data2/test1_cands.csv')
parser.add_argument("--x_test_file", type=str, default='xgb_data2/x_test.csv')
parser.add_argument("--y_test_file", type=str, default='xgb_data2/y_test.csv')
# parser.add_argument("--test_data_file", type=str, default='xgb_data/valid_cands.csv')
# parser.add_argument("--x_test_file", type=str, default='xgb_data/x_valid.csv')
# parser.add_argument("--y_test_file", type=str, default='xgb_data/y_valid.csv')
# parser.add_argument("--dtest_file", type=str, default='xgb_data/dtest.buffer')
parser.add_argument("--pred_data_file", type=str, default='model14/test1_results_xgb.csv_tmp_')
parser.add_argument("--model_path", type=str, default="model14/model")
parser.add_argument("--ntree_limit", type=int, default=107, help="ntree_limit")
args = parser.parse_args()

common_function.print_args(args)

common_function.makedir(args.pred_data_file)


def predict():
    ############# reading data  #################################################
    logger.info("Starting to read testing samples...")
    test_df = pd.read_csv(args.test_data_file, sep="\t", header=0, encoding="utf-8")
    logger.info("Finish reading testing samples !")

    ###################### load a CSV file into DMatrix ######################
    x_test = pd.read_csv(args.x_test_file, header=None, encoding="utf-8", sep="\t")
    y_test = pd.read_csv(args.y_test_file, header=None, encoding="utf-8", sep="\t")
    d_test = xgb.DMatrix(x_test, y_test)

    ###################### load a XGBoost binary file into DMatrix ######################

    # d_test = xgb.DMatrix(args.dtest_file)

    ###################### do predict ###########################
    bst = xgb.Booster()  # init model
    bst.load_model(args.model_path)  # load model

    p_test = bst.predict(d_test, ntree_limit=args.ntree_limit)

    test_df["score"] = p_test.ravel()
    logger.info("write pred_result")
    test_df.to_csv(args.pred_data_file + str(args.ntree_limit),
                   header=True, index=False, encoding='utf-8', sep="\t")
    logger.info("write pred_result done")
    logger.info("write final_pred_result")
    test_df = pd.read_csv(args.pred_data_file + str(args.ntree_limit), sep="\t", header=0)
    test_df_pred = test_df.groupby("qid")["score"].max()
    res_df = pd.merge(
        test_df,
        test_df_pred,
        on=["qid", "score"], how="inner", copy=False)
    res_df.to_csv(args.pred_data_file + str(args.ntree_limit) + "_final",
                  header=True, index=False, encoding='utf-8', sep="\t")
    logger.info("write final_pred_result done")
    res_json = {}
    for index, row in res_df.iterrows():
        logger.info(index)
        qid = row["qid"]
        answer = row["pred_answer"]
        res_json[qid] = answer
    logger.info("write final_pred_result_json")
    json.dump(res_json, open(args.pred_data_file + str(args.ntree_limit) + "_final.json", "w"),
              ensure_ascii=False, indent=4)
    logger.info("write final_pred_result_json done")


if __name__ == "__main__":
    predict()
