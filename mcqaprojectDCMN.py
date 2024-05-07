# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:58:53 2021

@author: GS63
"""

import os
import csv
import json
import glob
import pandas as pd
from json.decoder import JSONDecoder
from pathlib import Path
from tqdm import tqdm, trange
import torch
import random
import numpy as np
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW
from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast
from transformers import BertTokenizer
from params import  race_middle_train_file_path, race_middle_dev_file_path , race_middle_test_file_path
from params import  race_high_train_file_path, race_high_dev_file_path , race_high_test_file_path
from params import  dream_train_file_path, dream_dev_file_path, dream_test_file_path
from params import MCTest_test_file_path, MCTest_testAns_file_path
from params import datasetsDREAM, datasetsRACE, datasetsMCTEST
from params import race_raw_dev_path, race_raw_test_path, race_raw_train_path
from pytorch_pretrained_bert.modeling import BertConfig, BertForMultipleChoice
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam # WarmupLinearSchedule


class Race1(object):
    """We are going to train race dataset with bert."""
    def __init__(self,
                 race_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.race_id = race_id
        self.context_sentence = context_sentence # sentance
        self.start_ending = start_ending  # !uestion 
        # self.ending_0 = ending_0  #Option 1
        # self.ending_1 = ending_1  #Option 2
        # self.ending_2 = ending_2
        # self.ending_3 = ending_3
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "race_id: {}".format(self.race_id),
            "context_sentence: {}".format(self.context_sentence),
            "start_ending: {}".format(self.start_ending),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)

class Race2(object):
    """We are going to train race dataset with bert."""
    def __init__(self,
                 race_id,
                 article,
                 question,
                 option_0,
                 option_1,
                 option_2,
                 option_3,
                 label = None):
        self.race_id = race_id
        self.article = article # sentance
        self.question = question  # question 
        self.option_0 = option_0  #Option 1
        self.option_1 = option_1  #Option 2
        self.option_2 = option_2
        self.option_3 = option_3
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "race_id: {}".format(self.race_id),
            "article: {}".format(self.article),
            "question: {}".format(self.question),
            "option_0: {}".format(self.option_0),
            "option_1: {}".format(self.option_1),
            "option_2: {}".format(self.option_2),
            "option_3: {}".format(self.option_3),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return ", ".join(l)


class InputFeatures1(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class InputFeatures2(object):
    """ RACE dataset input feature 
    Args:
        example_id: document id
        choice_features: fecture ['input_ids', 'input_mask', 'segment_ids']
        label: label for answer
    """

    def __init__(self,
                 example_id,
                 article_features,
                 question_features,
                 option0_features,
                 option1_features,
                 option2_features,
                 option3_features,
                 label
                 ):
        self.example_id = example_id
        self.article_features = {
            'input_ids': article_features[1],
            'input_mask': article_features[2],
            'segment_ids': article_features[3]
        }

        self.question_features = {
            'input_ids': question_features[1],
            'input_mask': question_features[2],
            'segment_ids': question_features[3]
        }

        self.option0_features = {
            'input_ids': option0_features[1],
            'input_mask': option0_features[2],
            'segment_ids': option0_features[3]
        }

        self.option1_features = {
            'input_ids': option1_features[1],
            'input_mask': option1_features[2],
            'segment_ids': option1_features[3]
        }

        self.option2_features = {
                'input_ids': option2_features[1],
                'input_mask': option2_features[2],
                'segment_ids': option2_features[3]
        }

        self.option3_features = {
                'input_ids': option3_features[1],
                'input_mask': option3_features[2],
                'segment_ids': option3_features[3]
        }

        self.label = label



class MCQATrainModel:
    #set Trains
    ansLabel_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    #define difficulty
    difficulty_set = ["middle", "high"]
    #define dataset type
    dataset_type = ["train", "dev", "test"]
    
    
    def __init__(self, dataset):
        self = self
        self.dataset = dataset
        
        
    def loadDataFile(self, fileName):
        with open(fileName) as f:
            Sentence = []
            OutLabel = []
            Docs =f.readlines()
            for text in Docs: # look into each line
                    tempSen = text.replace("\n", "") # remove \n string
                    Sentence.append(tempSen)
            return (Sentence, OutLabel)
        
    def loadDreamJsonDataFile(self, fileName):
        conventionSent = []
        questionAns =[]
        index = []
        with open(fileName, 'r', encoding="utf-8") as f:
            samples = json.loads(f.read())
            #decode each array
            for sample in samples:
                self.sample = sample
                #print(sample[1][0]["answer"])
                conventionSent.append(sample[0])  # Conventional sentence
                questionAns.append(sample[1])  # question, choice, answer 
                index.append(sample[2])  #index
        return conventionSent, questionAns, index      
    
    def loadRaceJsonDataFile(self, fileName):
        with open(fileName, 'r', encoding="utf-8") as f:
            data =  json.loads(f.read())            
            return data["id"], data["article"], data["questions"], data["options"], data["answers"]
    
    def loadMCTestTsvDataFile(self, fileName):
        # print("*"*50)
        # print("file name: ", fileName)
        questID= []
        questArt = []
        questAsk = []
        questOpt = []
        with open(fileName) as f:
            data =  csv.reader(f, delimiter="\t")
            for row in data:
                tempAsk = []
                tempOpt = []
                # print("*"*50)
                # print(row[0]) # id
                # print(row[2]) # Article
                # print("-"*10)
                # print(row[3]) # Question 1
                # print(row[4]) # OptA -- Question1
                # print(row[5]) # OptB -- Question1
                # print(row[6]) # OptC -- Question1
                # print(row[7]) # OptD -- Question1
                # print("-"*10)
                # print(row[8]) # Question 2
                # print(row[9]) # OptA -- Question2
                # print(row[10]) # OptB -- Question2
                # print(row[11]) # OptC -- Question2
                # print(row[12]) # OptD -- Question2
                # print("-"*10)
                # print(row[13]) # Question 3
                # print(row[14]) # OptA -- Question3
                # print(row[15]) # OptB -- Question3
                # print(row[16]) # OptC -- Question3
                # print(row[17]) # OptD -- Question3
                # print("-"*10)
                # print(row[18]) # Question 4
                # print(row[19]) # OptA -- Question4
                # print(row[20]) # OptB -- Question4
                # print(row[21]) # OptC -- Question4
                # print(row[22]) # OptD -- Question4
                
                questID.append(row[0])
                questArt.append(row[2])
                tempAsk.append(row[3])
                tempAsk.append(row[8])
                tempAsk.append(row[13])
                tempAsk.append(row[18])
                questAsk.append(tempAsk)
                tempOpt.append(row[4:8])
                tempOpt.append(row[9:13])
                tempOpt.append(row[14:18])
                tempOpt.append(row[19:23])
                questOpt.append(tempOpt)
        

        return questID, questArt, questAsk, questOpt
    
    
    def loadMCTestTsvAnsFile(self, fileName):
        #print("file name: ", fileName)
        questAns = []
        with open(fileName) as f:
            data =  csv.reader(f, delimiter="\t")
            for row in data:
                tempAns = []
                # print(row[0]) # Answer Quest1
                # print(row[1]) # Answer Quest2
                # print(row[2]) # Answer Quest3
                # print(row[3]) # Answer Quest4
                questAns.append(row)
            
            return questAns
                
            
        
   
                
    def scanFileList(self, path):
        fileList = []
        for file in os.listdir(path):
            # print(file)
            fileList.append(os.path.join(path, file))
        return fileList
 
    def read_race(input_dir, data_grade = ["high","middle"]):
        samples = []    
        for grade in data_grade:
            dir_name = input_dir + '/' + grade + '/'
            
            files_list = glob.glob(dir_name + "*.txt")
            files_list = sorted(files_list, key=lambda x: int((x.split('/')[-1]).split('.')[0]))
            print("After sorted:",files_list[0])
            
            for file_name in files_list:
                f = open(file_name,'r',encoding='utf-8')
            
                sample = json.load(f)
                answers = sample['answers']
                text = sample["article"]
                questions = sample['questions']
                options = sample['options']
                rid = file_name[:-4] 
                #print(file_name)
                for i in range(len(answers)):
                    samples.append(Race1(
                        race_id = rid+":"+str(i),
                        context_sentence = text, 
                        start_ending = questions[i], 
                        ending_0 = options[i][0],
                        ending_1 = options[i][1],
                        ending_2 = options[i][2], 
                        ending_3 = options[i][3],
                        label = ord(answers[i])-65
                        ))
            return samples 
        
    def read_raceModify(self, input_dir, maxSentence):
        samples = []
        data_grade = ["middle","high"]
        for grade in data_grade:
            print("level", grade)
            dir_name = input_dir + grade + '/'
            #
            fileList = []
            for file in os.listdir(dir_name):
                # print(file)
                fileList.append(os.path.join(dir_name, file))
            fileList = sorted(fileList, key=lambda x: int((x.split('/')[-1]).split('.')[0]))
            print("After sorted:",fileList[0])
            sentenceCnt = 0
            for file_name in fileList:
                f = open(file_name,'r',encoding='utf-8')
                sentenceCnt += 1
                if(sentenceCnt >= maxSentence):
                    break
                sample = json.load(f)
                answers = sample['answers']
                text = sample["article"]
                questions = sample['questions']
                options = sample['options']
                #rid = file_name[:-4] 
                rid = sample['id']
                #print(file_name)
                for i in range(len(answers)):
                    samples.append(Race2(
                        race_id = rid+":"+str(i),
                        context_sentence = text, 
                        start_ending = questions[i], 
                        ending_0 = options[i][0],
                        ending_1 = options[i][1],
                        ending_2 = options[i][2], 
                        ending_3 = options[i][3],
                        label = self.ansLabel_map[answers[i]]#ord(answers[i])-65
                        ))
        return samples   
    
    def convert_examples_to_features1(self, examples, tokenizer, max_seq_length,
                                 is_training):
        """Loads a data file into a list of `InputBatch`s."""

        # RACE is a multiple choice task like Swag. To perform this task using Bert,
        #
        # Each choice will correspond to a sample on which we run the
        # inference. For a given Race example, we will create the 4
        # following inputs:
        # - [CLS] context [SEP] choice_1 [SEP]
        # - [CLS] context [SEP] choice_2 [SEP]
        # - [CLS] context [SEP] choice_3 [SEP]
        # - [CLS] context [SEP] choice_4 [SEP]
        # The model will output a single value for each input. To get the
        # final decision of the model, we will run a softmax over these 4
        # outputs.
        features = []
        print("Length of Example: ", len(examples),examples[0])
        for example_index, example in enumerate(examples):
            context_tokens = tokenizer.tokenize(example.context_sentence) # tokenize the sentance
            start_ending_tokens = tokenizer.tokenize(example.start_ending) # question 

            choices_features = []
            for ending_index, ending in enumerate(example.endings): #extract options
                # We create a copy of the context tokens in order to be
                # able to shrink it according to ending_tokens
                context_tokens_choice = context_tokens[:] 
                ending_tokens = start_ending_tokens + tokenizer.tokenize(ending) # question + option convert to tokenize
                # Modifies `context_tokens_choice` and `ending_tokens` in
                # place so that the total length is less than the
                # specified length.  Account for [CLS], [SEP], [SEP] with
                # "- 3"
                self._truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
                
                # generate full token with label ( sentence+ qustion+ option)
                tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"] 
                #generate segment_id for represent sentence 0= context , 1 = question 
                segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(start_ending_tokens)) +[2] * (len(tokenizer.tokenize(ending)) + 1)
                    
                input_ids = tokenizer.convert_tokens_to_ids(tokens) #convert full sentance + optiom  into BERT input ids , input token related id
                input_mask = [1] * len(input_ids) # mask 1 for input sentenace , 0 for padding 

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding    # padding 0 the full sentence
                input_mask += padding   # 1 for 
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            if example_index < 5:
                print("*** Example ***")
                print("race_id: {}".format(example.race_id))
                for choice_idx, (tokens, input_ids, input_mask, segment_ids) in enumerate(choices_features):
                    print("choice: {}".format(choice_idx))
                    print("tokens: {}".format(' '.join(tokens)))
                    print("input_ids: {}".format(' '.join(map(str, input_ids))))
                    print("input_mask: {}".format(' '.join(map(str, input_mask))))
                    print("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                if is_training:
                    print("label: {}".format(label))
            if (example_index%5000 ==0): print(example_index)	
            features.append(
                InputFeatures1(
                    example_id = example.race_id,
                    choices_features = choices_features,
                    label = label
                    )
                )

        return features
    
    
    def convert_examples_to_features2(self, examples, tokenizer, article_len, question_len, option_len):
        """Loads a data file into a list of `InputBatch`s.
        Args:
            examples: RACE2
            tokenizer: 
            max_seq_length: max sentence length
        Returns:
            features:{
                example_id:  file id
                choices_features :[
                    {article + question + option1}, ... , {article + question + option2}
                    ]
            label: answer label
            }
        """
        features = []
        for example_index, example in enumerate(examples):
            article_features = self.get_features(
                tokenizer, example.article, article_len)
            question_features = self.get_features(
                tokenizer, example.question, question_len)

            option0_features = self.get_features(
                tokenizer, example.option_0, option_len)
            option1_features = self.get_features(
                tokenizer, example.option_1, option_len)
            option2_features = self.get_features(
                tokenizer, example.option_2, option_len)
            option3_features = self.get_features(
                tokenizer, example.option_3, option_len)

            assert len(option0_features) == 4
            label = example.label

            features.append(
                InputFeatures2(
                    example_id=example.race_id,
                    article_features=article_features,
                    question_features=question_features,
                    option0_features=option0_features,
                    option1_features=option1_features,
                    option2_features=option2_features,
                    option3_features=option3_features, 
                    label=label
                    )
                )

        return features
    
    
    
    def _truncate_seq(self, tokens,  max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while len(tokens) > max_length:
            tokens.pop()

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
    
    def selectField(self, features, field):
        return [
                [
                    choice[field]
                    for choice in feature.choices_features
                ]
                for feature in features
            ]
    
    # select field for 2
    def select_field(self, features, field):
        return [choice[field] for choice in features]
    
    def get_features(self, tokenizer, text, max_len):
        text_tokens = tokenizer.tokenize(text)
        self._truncate_seq(text_tokens, max_len - 2)
        tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
        
        segment_ids = [0] * (len(text_tokens) + 2) 
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        return (tokens, input_ids, input_mask, segment_ids)
    
    
    def get_tensor(self, features):
        all_input_ids = torch.tensor(
            self.select_field(features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(
            self.select_field(features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(
            self.select_field(features, 'segment_ids'), dtype=torch.long)
        
        return all_input_ids, all_input_mask, all_segment_ids
   
    def convert_features_to_tensors(self, features, batch_size):
        article_fetures = [feature.article_features for feature in features]
        article_tensors = self.get_tensor(article_fetures)

        question_features = [feature.question_features for feature in features]
        question_tensors = self.get_tensor(question_features)

        option0_features = [feature.option0_features for feature in features]
        option0_tensors = self.get_tensor(option0_features)

        option1_features = [feature.option1_features for feature in features]
        option1_tensors = self.get_tensor(option1_features)

        option2_features = [feature.option2_features for feature in features]
        option2_tensors = self.get_tensor(option2_features)

        option3_features = [feature.option3_features for feature in features]
        option3_tensors = self.get_tensor(option3_features)

        all_label_ids = torch.tensor(
            [f.label for f in features], dtype=torch.long)

        data = TensorDataset(
            article_tensors[0], article_tensors[1], article_tensors[2],
            question_tensors[0], question_tensors[1], question_tensors[2],
            option0_tensors[0], option0_tensors[1], option0_tensors[2],
            option1_tensors[0], option1_tensors[1], option1_tensors[2],
            option2_tensors[0], option2_tensors[1], option2_tensors[2],
            option3_tensors[0], option3_tensors[1], option3_tensors[2],
            all_label_ids)

        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

        return dataloader
    
    
    def accuracy(out, labels):
        outputs = np.argmax(out, axis=1)
        #print(outputs,outputs == labels)
        return np.sum(outputs == labels)
        
        
    def loadAllDataSet(self):    
        #find race file list
        self.raceMidTrainFileList = self.scanFileList(race_middle_train_file_path)
    
        self.raceMidTestFileList = self.scanFileList(race_middle_test_file_path)

        self.raceMidDevFileList = self.scanFileList(race_middle_dev_file_path)
        
        self.raceHighTrainFileList = self.scanFileList(race_high_train_file_path)
    
        self.raceHighTestFileList = self.scanFileList(race_high_test_file_path)

        self.raceHighDevFileList = self.scanFileList(race_high_dev_file_path) 
       
       # find MCTest file list
        self.mcTestFileList = self.scanFileList(MCTest_test_file_path)
    
        self.mcTestAnsFileList = self.scanFileList(MCTest_testAns_file_path)
        
        if(self.dataset == datasetsDREAM):
              self.dreamDevConventionSent, self.dreamDevQustAns , \
              self.dreamDevIndex  =  self.loadDreamJsonDataFile(dream_dev_file_path)
        
              self.dreamTrainConventionSent, self.dreamTrainQustAns , \
              self.dreamTrainIndex = self.loadDreamJsonDataFile(dream_train_file_path)
        
              self.dreamTestConventionSent, self.dreamTestQustAns , \
              self.dreamTestIndex = self.loadDreamJsonDataFile(dream_test_file_path)
        
    
        if(self.dataset == datasetsMCTEST):
            self.mcTestID = []
            self.mcTestArticle = []
            self.mcTestQuestions = []
            self.mcTestOption = []
            self.mcTestAnswer = []
            for file in self.mcTestFileList:
                # print(file)
                if file.endswith(".tsv"):
                    tempID, tempArt, tempQust, tempOpt =self.loadMCTestTsvDataFile(file)
                    self.mcTestID.append(tempID)
                    self.mcTestArticle.append(tempArt)
                    self.mcTestQuestions.append(tempQust)
                    self.mcTestOption.append(tempOpt)
                    
                elif file.endswith(".ans"):
                     tempAns = self.loadMCTestTsvAnsFile(file)
                     self.mcTestAnswer.append(tempAns)
    
    
        if(self.dataset == datasetsRACE):
            self.raceMidDevID = []
            self.raceMidDevArticle = []
            self.raceMidDevQuestions = []
            self.raceMidDevOption = []
            self.raceMidDevAnswer = []
            self.raceMidTrainID = []
            self.raceMidTrainArticle = []
            self.raceMidTrainQuestions = []
            self.raceMidTrainOption = []
            self.raceMidTrainAnswer = []
            self.raceHighDevID = []
            self.raceHighDevArticle = []
            self.raceHighDevQuestions = []
            self.raceHighDevOption = []
            self.raceHighDevAnswer = []
            for file in self.raceMidDevFileList:
                tempID , tempArt, tempQust, tempOpt, tempAns = self.loadRaceJsonDataFile(file)
                self.raceMidDevID.append(tempID)
                self.raceMidDevArticle.append(tempArt)
                self.raceMidDevQuestions.append(tempQust)
                self.raceMidDevOption.append(tempOpt)
                self.raceMidDevAnswer.append(tempAns)
            for file in self.raceMidTrainFileList:
                tempID , tempArt, tempQust, tempOpt, tempAns = self.loadRaceJsonDataFile(file)
                self.raceMidTrainID.append(tempID)
                self.raceMidTrainArticle.append(tempArt)
                self.raceMidTrainQuestions.append(tempQust)
                self.raceMidTrainOption.append(tempOpt)
                self.raceMidTrainAnswer.append(tempAns)     
                
                
            for file in self.raceHighDevFileList:
                tempID , tempArt, tempQust, tempOpt, tempAns = self.loadRaceJsonDataFile(file)
                self.raceHighDevID.append(tempID)
                self.raceHighDevArticle.append(tempArt)
                self.raceHighDevQuestions.append(tempQust)
                self.raceHighDevOption.append(tempOpt)
                self.raceHighDevAnswer.append(tempAns)
            # self.sample =  self.read_raceModify(race_raw_train_path)      
                
    
if __name__ == "__main__":
    
    max_seq_length = 256
    gradient_accumulation_steps = 3
    train_batch_size = 3
    numEpoch = 2
    warmup_proportion= 0.1
    learning_rate = 5e-5
    seed= 42
    
    train_batch_size = train_batch_size // gradient_accumulation_steps
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    mcaqTrainModel = MCQATrainModel(datasetsRACE) # inital 
    # mcaqTrainModel.loadAllDataSet()
    #getTrain sample 
    trainSamples =  mcaqTrainModel.read_raceModify(race_raw_train_path, 10)
    num_train_optimization_steps  = int(len(trainSamples) /train_batch_size / gradient_accumulation_steps) * numEpoch
    
    print("Optimzation Step: ", num_train_optimization_steps)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    model = BertForMultipleChoice.from_pretrained("bert-base-uncased", 
                                                  cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE)),
                                                                         num_choices=4)
    
    print("Freeze network")
    for name, param in model.named_parameters():
        ln = 24
        if name.startswith('bert.encoder'):
        	l = name.split('.')
        	ln = int(l[3])
      
        if name.startswith('bert.embeddings') or ln < 6:
        	print(name)  
        	param.requires_grad = False


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    
    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)
    
    
    trainFeatures = mcaqTrainModel.convert_examples_to_features1(trainSamples, tokenizer, max_seq_length, True)
    trainLen = len(trainFeatures)
    print("***** Running training *****")
    print("  Num examples = %d",len(trainSamples))
    print("  Batch size = %d", train_batch_size)
    print("  Num steps = %d", num_train_optimization_steps)
    
    #convert into tensor
    all_input_ids =  torch.tensor(mcaqTrainModel.selectField(trainFeatures, 'input_ids') ,dtype=torch.long)
    all_input_mask = torch.tensor(mcaqTrainModel.selectField(trainFeatures, 'input_mask') ,dtype=torch.long)
    all_segment_ids =  torch.tensor(mcaqTrainModel.selectField(trainFeatures, 'segment_ids') ,dtype=torch.long)
    all_label= torch.tensor([f.label for f in trainFeatures], dtype=torch.long)
    
    trainData = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    #use RandomSampler 
    train_sampler = RandomSampler(trainData)
    #user DistributedSamper
    #train_sampler = DistributedSampler(trainData)
    trainDataLoader = DataLoader(trainData, sampler=train_sampler, batch_size= train_batch_size)
    #model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')
   
    # setup GPU/CPU
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'
    # move model over to detected device
    model.to(device)
    
    model.train()
    
    
    # optim = AdamW(model.parameters(), lr=learning_rate)
    global_step = 0
    
    
    for epoch in range(numEpoch):
        tr_loss = 0
        last_tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(trainDataLoader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # print(batch)
            print("input ids: ", input_ids)
            print("input mask: ", input_mask)
            print("segment_ids: ", segment_ids)
            print("label_ids: ", label_ids)
            # optim.zero_grad()
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # start_positions = batch['start_positions'].to(device)
            # end_positions = batch['end_positions'].to(device)
            #outputs = model(input_ids, attention_mask=input_mask, start_positions=start_positions, end_positions=end_positions)
            loss = model(input_ids, segment_ids, input_mask, label_ids) # for pytorch_pretrained_bert only
            # loss = outputs[0]
            loss = loss / gradient_accumulation_steps 
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps +=1
            loss.backward()
            # optim.step()
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step +=1
            
            if nb_tr_examples % 512 ==0:
                loss_log = (tr_loss - last_tr_loss)* 1.0/512
                print(nb_tr_examples, loss_log)
                last_tr_loss = tr_loss
        
    
    
    
    
    
        