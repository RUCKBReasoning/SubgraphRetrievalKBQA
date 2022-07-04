from utils import load_dict
import json
from tqdm import tqdm
relation_list, relation2id = load_dict('./tmp/relations.txt')
relation_list.append('END OF HOP')
print(len(relation_list))

question2sparql = json.load(open("./tmp/CWQ/ComplexWebQuestions_train.json"))
question2sparql += json.load(open("./tmp/CWQ/ComplexWebQuestions_dev.json"))
question2sparql += json.load(open("./tmp/CWQ/ComplexWebQuestions_test.json"))
question2sparql = {i["ID"]:i["sparql"] for i in question2sparql}
unfound_question = 0
not_in_list = 0
def slic_is_valid(slic):
    global unfound_question,not_in_list
    check_list = []
    if slic["positive_rel"]!="END OF HOP":
        check_list.append(relation_list[slic["positive_rel"]])
    check_list.extend([relation_list[j] for j in slic["history"]])
    for j in check_list:
        if slic["id"] not in question2sparql:
            #print("question not found: ",slic["question"])
            unfound_question+=1
            return False
        if j not in question2sparql[slic["id"]]:
            not_in_list+=1
            return False
    return True
fin = open("./tmp/cwq_full_with_search_state.jsonl",'r')
slic_all = [(json.loads(i),i) for i in tqdm(fin.readlines()) if len(i.strip())>0]
fin.close()
slic_all = [i[1].strip() for i in tqdm(slic_all) if slic_is_valid(i[0])]
print("question unfound",unfound_question)
fout = open("./tmp/cwq_full_with_search_state.jsonl",'w')
for i in tqdm(slic_all):
    fout.write(i+"\n")
fout.close()
