# -*- coding: utf-8 -*-
import json
import os
import sys
from itertools import combinations
from collections import OrderedDict

__author__ = "liuaiting@bupt.edu.cn"


def data_split():
    f = json.load(open("data_join/train_baidushort.json"))
    part1 = {
        "version": "2.0",
        "data": [
            {
                "paragraphs": [],
                "title": ""
            }
        ]
    }
    part2 = {
        "version": "2.0",
        "data": [
            {
                "paragraphs": [],
                "title": ""
            }
        ]
    }
    part1["data"][0]["paragraphs"].extend(f["data"][0]["paragraphs"][:len(f["data"][0]["paragraphs"]) // 2])
    part2["data"][0]["paragraphs"].extend(f["data"][0]["paragraphs"][len(f["data"][0]["paragraphs"]) // 2:])
    print(len(f["data"][0]["paragraphs"]))
    print(len(part1["data"][0]["paragraphs"]))
    print(len(part2["data"][0]["paragraphs"]))

    json.dump(part1, open("data_join/train_baidushort_part1.json", "w"), ensure_ascii=False, indent=2)
    json.dump(part2, open("data_join/train_baidushort_part2.json", "w"), ensure_ascii=False, indent=2)


def data_split2():
    f = json.load(open("data_join/train_lic.json"))
    part1 = {
        "version": "2.0",
        "data": [
            {
                "paragraphs": [],
                "title": ""
            }
        ]
    }
    part2 = {
        "version": "2.0",
        "data": [
            {
                "paragraphs": [],
                "title": ""
            }
        ]
    }
    part1["data"][0]["paragraphs"].extend(f["data"][0]["paragraphs"][:(len(f["data"][0]["paragraphs"]) // 10) * 7])
    part2["data"][0]["paragraphs"].extend(f["data"][0]["paragraphs"][(len(f["data"][0]["paragraphs"]) // 10) * 7:])
    print(len(f["data"][0]["paragraphs"]))
    print(len(part1["data"][0]["paragraphs"]))
    print(len(part2["data"][0]["paragraphs"]))

    json.dump(part1, open("data_join/train_lic_part1.json", "w"), ensure_ascii=False, indent=2)
    json.dump(part2, open("data_join/train_lic_part2.json", "w"), ensure_ascii=False, indent=2)


def data_split_dev():
    f = json.load(open("dureader_robust-data/dev.json"))
    part1 = {
        "version": "2.0",
        "data": [
            {
                "paragraphs": [],
                "title": ""
            }
        ]
    }
    part2 = {
        "version": "2.0",
        "data": [
            {
                "paragraphs": [],
                "title": ""
            }
        ]
    }
    part1["data"][0]["paragraphs"].extend(f["data"][0]["paragraphs"][:1000])
    part2["data"][0]["paragraphs"].extend(f["data"][0]["paragraphs"][1000:])
    print(len(f["data"][0]["paragraphs"]))
    print(len(part1["data"][0]["paragraphs"]))
    print(len(part2["data"][0]["paragraphs"]))

    json.dump(part1, open("dureader_robust-data/dev_part1.json", "w"), ensure_ascii=False, indent=2)
    json.dump(part2, open("dureader_robust-data/dev_part2.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # data_split()
    # data_split2()
    data_split_dev()
