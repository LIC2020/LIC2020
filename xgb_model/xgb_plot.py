#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import xgboost as xgb
import matplotlib.pyplot as plt

__author__ = "liuaiting@bupt.edu.cn"

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")
parser.add_argument("--model_path", type=str, default="model14/model")
parser.add_argument("--max_num_features", type=int, default=30)
parser.add_argument("--fig_path", type=str, default="fig/")
args = parser.parse_args()


def main():
    bst = xgb.Booster(model_file=args.model_path)  # init model
    fig, ax = plt.subplots()
    xgb.plot_importance(bst, ax=ax, max_num_features=args.max_num_features)
    fig_name = os.path.join(args.fig_path, "model14_feature_{}.png".format(args.max_num_features))
    plt.savefig(fig_name)


if __name__ == "__main__":
    main()
