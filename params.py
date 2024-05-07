# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:34:33 2021

@author: GS63
"""

datasetsMCTEST = "MCTest"
datasetsRACE = "RACE"
datasetsDREAM = "DREAM"

race_raw_train_path = "./datasets/{}/train/".format(datasetsRACE)
race_raw_dev_path = "./datasets/{}/dev/".format(datasetsRACE)
race_raw_test_path = "./datasets/{}/test/".format(datasetsRACE)

race_high_train_file_path = "./datasets/{}/train/high/".format(datasetsRACE)
race_high_dev_file_path = "./datasets/{}/dev/high/".format(datasetsRACE)
race_high_test_file_path = "./datasets/{}/test/high/".format(datasetsRACE)


race_middle_train_file_path = "./datasets/{}/train/middle/".format(datasetsRACE)
race_middle_dev_file_path = "./datasets/{}/dev/middle/".format(datasetsRACE)
race_middle_test_file_path = "./datasets/{}/test/middle/".format(datasetsRACE)

dream_train_file_path = "./datasets/{}/train.json".format(datasetsDREAM)
dream_dev_file_path = "./datasets/{}/dev.json".format(datasetsDREAM)
dream_test_file_path = "./datasets/{}/test.json".format(datasetsDREAM)


MCTest_test_file_path = "./datasets/{}/MCTest/".format(datasetsMCTEST)
MCTest_testAns_file_path = "./datasets/{}/MCTestAnswers/".format(datasetsMCTEST)