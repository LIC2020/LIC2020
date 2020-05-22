# -*- coding: utf-8 -*-
import json

__author__ = "liuaiting@bupt.edu.cn"


def join1(output_file):
    lic2020_json = json.load(open("dureader_robust-data/train.json"))  # 14520
    lic2020_dev_json = json.load(open("dureader_robust-data/dev.json"))  # 1417
    # baidu_short_json = json.load(open("baidu_short/train_for_lic.json"))  # 48526
    # cmrc2018_train_json = json.load(open("cmrc2018/cmrc2018_train_keep_1ans.json"))  # 10142
    # cmrc2018_dev_json = json.load(open("cmrc2018/cmrc2018_dev_keep_1ans.json"))  # 3219
    # cmrc2018_trial_json = json.load(open("cmrc2018/cmrc2018_trial_keep_1ans.json"))  # 1002
    # cetc_json = json.load(open("cetc/train_lic_format.json"))  # 77873
    # cetc_500_json = json.load(open("cetc/train_lic_format_500_30.json"))  # 24054
    # cetc_1000_json = json.load(open("cetc/train_lic_format_1000_30.json"))  # 46502
    # cetc_1500_json = json.load(open("cetc/train_lic_format_1500_30.json"))  # 56376
    # drcd_json = json.load(open("DRCD_new/DRCD_new.json"))  # 26936
    # cail_json = json.load(open("CAIL_new/CAIL_new.json"))  # 38099
    # dureader_zhidao_json = json.load(open("dureader_new/dureader_zhidao_new.train.json"))
    # dureader_search_json = json.load(open("dureader_new/dureader_search_new.train.json"))

    squad_json = {
        "version": "1.1",
        "data": []
    }
    # squad_json["data"].extend(cetc_json["data"])
    # squad_json["data"].extend(cetc_500_json["data"])
    # squad_json["data"].extend(cetc_1000_json["data"])
    # squad_json["data"].extend(cetc_1500_json["data"])
    # squad_json["data"].extend(baidu_short_json["data"])
    # squad_json["data"].extend(cmrc2018_train_json["data"])
    # squad_json["data"].extend(cmrc2018_dev_json["data"])
    # squad_json["data"].extend(cmrc2018_trial_json["data"])
    squad_json["data"].extend(lic2020_json["data"])
    squad_json["data"].extend(lic2020_dev_json["data"])
    # squad_json["data"].extend(drcd_json["data"])
    # squad_json["data"].extend(cail_json["data"])
    # squad_json["data"].extend(dureader_search_json["data"])
    # squad_json["data"].extend(dureader_zhidao_json["data"])

    json.dump(squad_json, open(output_file, "w"), ensure_ascii=False, indent=2)


def as_squad_v1_format(input_file, output_file, name):
    a = json.load(open(input_file))
    b = {
        "version": "1.1",
        "data": []
    }
    reindex = 0
    query_filter_num = 0
    for data in a['data']:
        data_json = {
            "paragraphs": [],
            "title": data.get("title", "")
        }
        for para in data['paragraphs']:
            context = para["context"]
            if context:
                para_json = {
                    "context": context,
                    "qas": []
                }
                for qas in para['qas']:
                    question = qas["question"]
                    if question:
                        query_json = {
                            "question": question,
                            "id": name + "_" + str(reindex),
                            "answers": [],
                        }
                        if name == "TRAIN" and len(qas["answer"]) > 1:
                            print("TRAIN error: more than one answer")

                        for ans in qas["answer"]:  # TODO: error
                            answer = ans["text"]
                            answer_start = ans["answer_start"]
                            if answer not in context:
                                print("error3: answer not in context.")
                                print(context)
                                print(answer)
                            ans_json = {
                                "text": answer,
                                "answer_start": answer_start
                            }
                            query_json["answers"].append(ans_json)
                        para_json["qas"].append(query_json)
                        reindex += 1
                    else:
                        query_filter_num += 1
                        print("error2: Question empty!", qas["id"])
                if para_json["qas"]:
                    data_json["paragraphs"].append(para_json)
            else:
                print("error1: Context empty!")
                print(para)
        if data_json["paragraphs"]:
            b["data"].append(data_json)

    json.dump(b, open(output_file, "w"), ensure_ascii=False, indent=2)


def check_squad_v1(input_file, name):
    """
    {
    "data": [
        {
            "paragraphs": [
                {
                    "context": "",
                    "id": "",  ## optional
                    "qas": [
                        {
                            "question": "",
                            "id": "",
                            "answers": [
                                {
                                    "text": "",
                                    "answer_start": -1,
                                }
                            ]
                        }
                    ]
                }
            ],
            "title": "",
            "id": ""  ## optional
        }
    ]
    }
    """
    f_key = {'data', 'version'}
    f_key_optional = {'data'}
    data_key = {'title', 'paragraphs'}
    data_key_optional = {'title', 'paragraphs'}
    para_key = {'context', 'qas'}
    para_key_optional = {'context', 'qas', 'id'}
    qas_key = {'question', 'id', 'answers'}
    ans_key = {'answer_start', 'text'}
    ans_key_optional = {'answer_start', 'text', "id"}
    a = json.load(open(input_file))
    assert set(a.keys()) == f_key or set(a.keys()) == f_key_optional
    for data in a["data"]:
        assert set(data.keys()) == data_key or set(data.keys()) == data_key_optional
        for para in data["paragraphs"]:
            assert set(para.keys()) == para_key or set(para.keys()) == para_key_optional
            for qas in para["qas"]:
                assert set(qas.keys()) == qas_key
                if name == "TRAIN":
                    if len(qas["answers"]) > 1:
                        print("TRAIN error: more than one answer.")
                for ans in qas["answers"]:
                    assert set(ans.keys()) == ans_key or set(ans.keys()) == ans_key_optional
                    if para["context"].find(ans["text"]) == -1:
                        print("qid     :", qas["id"])
                        print("context :", para["context"])
                        print("answer  :", ans["text"])


if __name__ == "__main__":
    # join1("data_join/train_baidushort.json")
    # join1("data_join/train_cmrc.json")
    # join1("data_join/train_cetc.json")
    # join1("data_join/train_baidushort_cmrc.json")
    # join1("data_join/train_baidushort_cmrc_lic.json")
    # join1("data_join/train_cetc500_baidushort_cmrc_lic.json")
    # join1("data_join/train_cetc500_baidushort_cmrc.json")
    # join1("data_join/train_cetc1500.json")
    # join1("data_join/train_lic.json")
    # join1("data_join/train_drcd.json")
    # join1("data_join/train_dureader.json")
    join1("data_join/train_lic_all.json")

    # check_squad_v1("DRCD_new/DRCD_new.json", "TRAIN")
    # check_squad_v1("CAIL_new/CAIL_new.json", "TRAIN")
    # check_squad_v1("dureader_new/dureader_search_new.train.json", "TRAIN")
    # check_squad_v1("dureader_new/dureader_zhidao_new.train.json", "TRAIN")




