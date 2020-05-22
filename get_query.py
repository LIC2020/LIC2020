# -*- coding: utf-8 -*-
import json
import os

__author__ = "liuaiting@bupt.edu.cn"


def get_query():
    a = json.load(open("dureader_robust-data/train.json"))

    with open("lic_query.txt", "w") as f:
        for data in a["data"]:
            for para in data["paragraphs"]:
                for qas in para["qas"]:
                    question = qas["question"]
                    f.write(question + "\n")


if __name__ == "__main__":
    get_query()
