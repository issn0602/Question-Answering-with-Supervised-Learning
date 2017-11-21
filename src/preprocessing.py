import json

#import matplotlib
#import random
from nltk.tokenize import RegexpTokenizer

### Convert from JSON to glove vector

### Convert from JSON into TXT
#random.seed(123)


def read_json_data(json_data,data):
    tokenizer = RegexpTokenizer(r'\w+')
    paragraphs = []
    para_count = []  # no of questions for each para
    questions = []
    answers = []
    qas_count = []  # para index for each question

    for i in range(len(json_data)):
        # print(json_data[i]["title"])
        for j in range(len(json_data[i]["paragraphs"])):
            # print(json_data[i]["paragraphs"][j]["context"])
            # print(tokenizer.tokenize(json_data[i]["paragraphs"][j]["context"]))
            paragraphs = (tokenizer.tokenize(json_data[i]["paragraphs"][j]["context"]))
            #para_count.append(len(json_data[i]["paragraphs"][j]["qas"]))

            for k in range(len(json_data[i]["paragraphs"][j]["qas"])):
                questions = (tokenizer.tokenize(str(json_data[i]["paragraphs"][j]["qas"][k]["question"])))
                answers = (tokenizer.tokenize(str(json_data[i]["paragraphs"][j]["qas"][k]["answers"][0]["text"])))
                #qas_count.append(j)
                #	print("\t" + str(json_data[i]["paragraphs"][j]["qas"][k]["question"]))
                #	print("\t" + str(json_data[i]["paragraphs"][j]["qas"][k]["answers"][0]["text"]))
                #	print()
                # for data in json_data:
                data.append([paragraphs,questions,answers])
                #	print(data)
    #print(qas_count)
    #for i in range(len(questions)):
     #   data.append([paragraphs[qas_count[i]], questions[i], answers[i]])
    return data

def open_json_file(json_file):

    json_data_raw = open(json_file, 'r')
    json_data = json.load(json_data_raw)
    return (json_data)
# print(json_data["data"][0]["title"])
# print(json_data["data"][0]["paragraphs"][0]["context"])

def parse_data(json_data):
    data = []
    data = read_json_data(json_data["data"],data)
    return (data)

#json_file = "../train-v1.1.json"
#json_data = open_json_file(json_file)
#data = parse_data(json_data)
#for i in range(len(data)):
 #   print(str(data[i])+"\n")
#print(data)
