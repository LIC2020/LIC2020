# -*- coding:utf-8 -*-
import sys
import json
import jieba.posseg as psg
from transformers.tokenization_bert import BasicTokenizer

__author__ = 'Ziwei Bai'

tokenizer = BasicTokenizer(do_lower_case=True)

punct = set(["。", ".", ",", "，", ";", "；", "!", "！", "?", "？"])

pos_list = ["Ag", "a", "ad", "an", "b", "c", "dg", "d", "e", "f", "g", "h", "j", "k", "l", "m", "Ng", "n", "nr", "ns",
            "nt", "nz", "o", "p", "q", "r", "s", "tg", "t", "u", "vg", "v", "vd", "vn", "w", "x", "y", "z", "un", "eng"]
pos_dict = dict(zip(pos_list, range(len(pos_list))))


def get_length(answer):
    """
        这个用来检测answer的长度
        Inputs: 答案文本
        Outputs: 一个标量
    """
    return len(answer)


def check_punct(answer):
    """
        这个用来检测答案中是否包含标点符号。
        因为有的答案是两个半个的句子拼在了一起，不通顺
        比如：
            “双水解,而是沉淀”
        像 "《"，"（"，"）"，这种不予考虑
        我们目前只考虑：中英文句号，逗号，分号， 惊叹号， 问号
        Inputs: 答案文本
        Outputs: 一个标量 （0：不包含标点；1：句中包括标点；2：句子首尾包含标点）
    """
    answer = answer.strip()

    if answer[0] in punct or answer[-1] in punct:
        return 2
    for c in answer[1:-1]:
        if c in punct:
            return 1

    return 0


def QAOverlap1(question, answer):
    """
        这个用来检测 Question 和 Answer的相似度
        有的case中有这种情况存在：
            Q:动感单车哪个牌子好
            A:动感单车哪个牌子好，动感单车品牌有蓝堡
        在计算重叠度前需要去掉所有空格并且做小写操作（暂时不去停用词，因为没有一个特别好的停用词表）
        Inputs: 问题文本，答案文本
        Outputs: 重叠长度/min(答案长度,问题长度) (重复单词只算一次（set）)
    """

    question = "".join(question.strip().split())  # 去空格
    question = set(tokenizer.tokenize(question))

    answer = "".join(answer.strip().split())  # 去空格
    answer = set(tokenizer.tokenize(answer))

    return len(question & answer) / min(len(answer), len(question))


def QAOverlap2(question, answer):
    """
        和上面的区别是不删除空格
    """

    question = set(tokenizer.tokenize(question))
    answer = set(tokenizer.tokenize(answer))

    return len(question & answer) / min(len(answer), len(question))


def Check_pos(answer):
    """
        检查answer中有哪些词（动词/名词/介词等等）
        符号和词性对应关系可见 https://blog.csdn.net/enter89/article/details/80619805
        Inputs: 答案文本
        Output: List，每个元素代表输入文本中是否有pos_list对应位置的词性，有为1，没有为0
    """

    words = psg.cut(answer)
    Has_pos = [0] * (len(pos_list) + 1)

    for word, pos in words:
        Has_pos[pos_dict.get(pos, len(pos_list))] = 1

    # return Has_pos
    return "-".join(list(map(str, Has_pos)))


# if __name__ == "__main__":
#     source_data = json.load(open(sys.argv[1]))["data"]
#     nbest_data = json.load(open(sys.argv[2]))
#
#     for entry in source_data:
#         for paragraph in entry["paragraphs"]:
#             for qa in paragraph["qas"]:
#                 _id = qa["id"]
#                 question = qa["question"]
#                 nbest = nbest_data[_id]
#                 for result in nbest:
#                     answer = result["text"]
#                     length = get_length(answer)
#                     has_punct = check_punct(answer)
#                     overlap1 = QAOverlap1(question, answer)
#                     overlap2 = QAOverlap2(question, answer)
#                     pos = Check_pos(answer)
#                     pos = list(map(str, pos))
#
#                     print("{} ||| {}".format(question, answer))
#                     print("{}\t{}\t{}\t{}\t{}".format(length, has_punct, overlap1, overlap2, " ".join(pos)))
