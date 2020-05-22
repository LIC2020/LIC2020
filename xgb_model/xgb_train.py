#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
import logging
import argparse
import os
import json

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import common_function

__author__ = "liuaiting@bupt.edu.cn"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# TODO: set %env JOBLIB_TEMP_FOLDER=/tmp, otherwise will raise "OSError: [Errno 28] No space left on device"
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--train_data_file", type=str, default='xgb_data3/train_cands.csv')
parser.add_argument("--valid_data_file", type=str, default='xgb_data/valid_cands.csv')
parser.add_argument("--test_data_file", type=str, default='xgb_data/test1_cands.csv')
parser.add_argument("--x_train_file", type=str, default='xgb_data3/x_train.csv')
parser.add_argument("--x_valid_file", type=str, default='xgb_data/x_valid.csv')
parser.add_argument("--x_test_file", type=str, default='xgb_data/x_test.csv')
parser.add_argument("--y_train_file", type=str, default='xgb_data3/y_train.csv')
parser.add_argument("--y_valid_file", type=str, default='xgb_data/y_valid.csv')
parser.add_argument("--y_test_file", type=str, default='xgb_data/y_test.csv')
# parser.add_argument("--dtrain_file", type=str, default='xgb_data/dtrain.buffer')
# parser.add_argument("--dvalid_file", type=str, default='xgb_data/dvalid.buffer')
# parser.add_argument("--dtest_file", type=str, default='xgb_data/dtest.buffer')
parser.add_argument("--pred_data_file", type=str, default='model31/results_xgb.csv')
parser.add_argument("--model_path", type=str, default="./model31")
parser.add_argument("--early_stopping_rounds", type=int, default=10, help="[default=50]")
parser.add_argument("--num_boost_round", type=int, default=200, help="[default=1000]")
parser.add_argument("--booster", type=str, default="gbtree", help="[default= gbtree ]")
parser.add_argument("--eta", type=float, default=0.2, help="[default=0.3]")
parser.add_argument("--gamma", type=float, default=0, help="[default=0]")
parser.add_argument("--max_depth", type=int, default=3, help="[default=6]")
parser.add_argument("--min_child_weight", type=float, default=5, help="[default=1]")
parser.add_argument("--max_delta_step", type=float, default=0, help="[default=0]")
parser.add_argument("--subsample", type=float, default=0.9, help="[default=1]")
parser.add_argument("--colsample_bytree", type=float, default=0.9, help="[default=1]")
parser.add_argument("--colsample_bylevel", type=float, default=1, help="[default=1]")
parser.add_argument("--lamda", type=float, default=0.6, help="[default=1]")
parser.add_argument("--alpha", type=float, default=0.0001, help="[default=0]")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--scale_pos_weight", type=float, default=0.0005, help="[default=1]")
parser.add_argument("--objective", type=str, default="binary:logistic")
parser.add_argument("--eval_metric", type=str, default="error,logloss")
args = parser.parse_args()
"""
https://xgboost.readthedocs.io/en/latest/parameter.html
"""
common_function.print_args(args)
common_function.makedir(args.pred_data_file)
common_function.makedir(args.x_train_file)
common_function.makedir(args.y_train_file)
common_function.makedir(args.model_path)


def train():
    # TODO: Saving DMatrix into a XGBoost binary file will make loading faster
    ######################### load csv data ########################
    # # TODO: load a CSV file into DMatrix
    logger.info("Start loading csv.")
    x_train = pd.read_csv(args.x_train_file, header=None, encoding="utf-8", sep="\t")
    x_valid = pd.read_csv(args.x_valid_file, header=None, encoding="utf-8", sep="\t")
    # x_test = pd.read_csv(args.x_test_file, header=None, encoding="utf-8", sep="\t")

    y_train = pd.read_csv(args.y_train_file, header=None, encoding="utf-8", sep="\t")
    y_valid = pd.read_csv(args.y_valid_file, header=None, encoding="utf-8", sep="\t")
    # y_test = pd.read_csv(args.y_test_file, header=None, encoding="utf-8", sep="\t")

    d_train = xgb.DMatrix(x_train, y_train)
    d_valid = xgb.DMatrix(x_valid, y_valid)
    logger.info("Done loading csv.")

    ########################## load DMatrix #########################
    # # TODO: load a XGBoost binary file into DMatrix
    # d_train = xgb.DMatrix(args.dtrain_file)
    # d_valid = xgb.DMatrix(args.dvalid_file)
    # d_test = xgb.DMatrix(args.dtest_file)

    ########################### train models ########################
    params = {
        "booster": args.booster,
        "eta": args.eta,
        "gamma": args.gamma,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "max_delta_step": args.max_delta_step,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "lambda": args.lamda,
        "alpha": args.alpha,
        "scale_pos_weight": args.scale_pos_weight,
        "objective": args.objective,
        "eval_metric": list(args.eval_metric.split(","))
    }

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    # watchlist = [(d_train, 'train')]

    bst = xgb.train(params, d_train, args.num_boost_round, watchlist, early_stopping_rounds=args.early_stopping_rounds)
    bst.save_model(args.model_path + '/model')
    bst.dump_model(args.model_path + '/model.dump')

    ## make the submission
    # p_test = bst.predict(xgb.DMatrix(x_test))
    # test_df = pd.read_csv(args.test_data_file, sep="\t", header=0, encoding="utf-8")
    # test_df["score"] = p_test.ravel()
    # test_df.to_csv(args.pred_data_file, header=True, index=False, encoding='utf-8', sep="\t")
    # test_df_pred = test_df.groupby("qid")["score"].max()
    # res_df = pd.merge(
    #     test_df,
    #     test_df_pred,
    #     on=["qid", "score"])
    # res_df.to_csv(args.pred_data_file + str(args.ntree_limit) + "_final",
    #               header=True, index=False, encoding='utf-8', sep="\t")
    # res_json = {}
    # for index, row in res_df.iterrows():
    #     qid = row["qid"]
    #     answer = row["pred_answer"]
    #     res_json[qid] = answer
    # json.dump(res_json, open(args.pred_data_file + str(args.ntree_limit) + "_final.json", "w"),
    #           ensure_ascii=False, indent=4)

    ## make the submission for best
    # p_test = bst.predict(xgb.DMatrix(x_test), ntree_limit=bst.best_ntree_limit)
    # test_df = pd.read_csv(args.test_data_file, sep="\t", header=0, encoding="utf-8")
    # test_df["score"] = p_test.ravel()
    # test_df.to_csv(args.pred_data_file + "_best", header=True, index=False, encoding='utf-8', sep="\t")
    # test_df_pred = test_df.groupby("qid")["score"].max()
    # res_df = pd.merge(
    #     test_df,
    #     test_df_pred,
    #     on=["qid", "score"])
    # res_df.to_csv(args.pred_data_file + str(args.ntree_limit) + "_final",
    #               header=True, index=False, encoding='utf-8', sep="\t")
    # res_json = {}
    # for index, row in res_df.iterrows():
    #     qid = row["qid"]
    #     answer = row["pred_answer"]
    #     res_json[qid] = answer
    # json.dump(res_json, open(args.pred_data_file + str(args.ntree_limit) + "_final.json", "w"),
    #           ensure_ascii=False, indent=4)

    logger.info("best_iteration: {}".format(bst.best_iteration))
    logger.info("ntree_limit=bst.best_ntree_limit: {}".format(bst.best_ntree_limit))
    logger.info("best_score: {}".format(bst.best_score))


def grid_search_params():
    # TODO: load a CSV file into DMatrix
    logger.info("Start loading csv.")
    x_train = pd.read_csv(args.x_train_file, header=None, encoding="utf-8", sep="\t")
    y_train = pd.read_csv(args.y_train_file, header=None, encoding="utf-8", sep="\t")
    logger.info("Done loading csv.")

    ######################## grid seach params ############################
    other_params = {
        "booster": args.booster,
        "learning_rate": args.eta,  # [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
        "n_estimators": args.num_boost_round,
        "gamma": args.gamma,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "max_delta_step": args.max_delta_step,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "colsample_bylevel": args.colsample_bylevel,
        "reg_lambda": args.lamda,
        # "reg_alpha": args.alpha,
        "scale_pos_weight": args.scale_pos_weight,
        "objective": args.objective,
        "eval_metric": list(args.eval_metric.split(","))
    }
    cv_params = {
        # "n_estimators": range(10, 50, 10),
        # "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3]
        # 'max_depth': range(3, 10, 1),
        # 'min_child_weight': range(1, 6, 1)
        # 'gamma': [i/10.0 for i in range(0, 5)]
        # 'subsample': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
        # 'colsample_bylevel': [i / 10.0 for i in range(6, 10)]
        'reg_alpha': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
        # 'reg_lambda': [i / 10.0 for i in range(6, 10)]
        # 'scale_pos_weight': [1, 0.1, 0.15, 10, 15]

    }
    model = xgb.XGBClassifier(**other_params)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    grid_search = GridSearchCV(model, param_grid=cv_params, scoring="neg_log_loss", n_jobs=-1, cv=kfold, verbose=2)
    grid_result = grid_search.fit(x_train, y_train[0])
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":
    train()
    # grid_search_params()
