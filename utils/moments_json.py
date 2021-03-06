from __future__ import print_function, division
import os
import sys
import json
import pandas as pd

def convert_csv_to_dict(csv_path, subset):
    with open(csv_path) as f:
       lines= f.readlines()

    label_name_dict = {}
    keys = []
    keys_labels = []
    for i in range(len(lines)):
        basename = lines[i].split('\n')[0].split(' ')[0].split('/')[1] 
        label  = lines[i].split('\n')[0].split('/')[0] 
        print("basename: ", basename) 
        print("label: ", label)    
        keys.append(basename)
        keys_labels.append(label)

  
    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        label = keys_labels[i]
        database[key]['annotations'] = {'label': label}

    return database

def load_labels(categories_path):
    with open(categories_path) as f:
       lines= f.readlines()

    labels = []
    for i in range(len(lines)):
        label = lines[i].split('\n')[0]
        labels.append(label)

    return labels

def convert_kinetics_csv_to_moments_json(categories_path, train_csv_path, val_csv_path, dst_json_path):
    labels = load_labels(categories_path)

    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
  

    dst_data = {}
    dst_data['labels'] = labels
    print("labels")
    print(labels) 
    print("train_database")
    print(train_database)
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database) 

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__=="__main__":
  categories_path = "./data/moments/moments_categories.txt"
  #train_csv_path = "/home/lili/Video/3D-ResNets-PyTorch/data/moments/moments_15_train_list.csv"
  train_csv_path  = "./data/moments/moments_raw_val_list.csv"
  val_csv_path = "./data/moments/moments_raw_val_list.csv"
  dst_json_path = "./data/moments.json"
 

  convert_kinetics_csv_to_moments_json(categories_path, train_csv_path, val_csv_path, dst_json_path)
