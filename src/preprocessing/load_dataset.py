import os
import json
import argparse

from config import cfg

def load_webqsp():
    load_data_path = cfg.preprocessing["step0"]["load_data_path"]
    dump_data_path = cfg.preprocessing["step0"]["dump_data_path"]
    folder_path = cfg.preprocessing["step0"]["dump_data_folder"]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(load_data_path, "r") as f:
        train_dataset = json.loads(f.read())
        data_list = []
        for json_obj in train_dataset["Questions"]:
            question = json_obj["ProcessedQuestion"]
            for parse in json_obj["Parses"]:
                topic_entities = [parse["TopicEntityMid"]]
                answers = []
                for answer_json_obj in parse["Answers"]:
                    if answer_json_obj["AnswerType"] == "Entity":
                        answers.append(answer_json_obj["AnswerArgument"])
                if len(answers) == 0:
                    continue
                data_list.append({
                    "question": question,
                    "topic_entities": topic_entities,
                    "answers": answers,
                })
        with open(dump_data_path, "w") as f:
            for json_obj in data_list:
                f.write(json.dumps(json_obj) + "\n")
