import os
import json
import argparse
from collections import namedtuple

DatasetPath = namedtuple('DataPath', ['train_dataset_path', 'test_dataset_path'])

def load_webqsp(path: DatasetPath):
    tmpDatasetPath = DatasetPath(
        train_dataset_path="../tmp/retriever/train.jsonl",
        test_dataset_path="../tmp/retriever/test.jsonl"
    )
    with open(path.train_dataset_path, "r") as f:
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
        with open(tmpDatasetPath.train_dataset_path, "w") as f:
            for json_obj in data_list:
                f.write(json.dumps(json_obj) + "\n")
    
    with open(path.test_dataset_path, "r") as f:
        test_dataset = json.loads(f.read())
        data_list = []
        for json_obj in test_dataset["Questions"]:
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
        with open(tmpDatasetPath.test_dataset_path, "w") as f:
            for json_obj in data_list:
                f.write(json.dumps(json_obj) + "\n")
