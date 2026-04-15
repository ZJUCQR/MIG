import io
import os
import json
import zipfile
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constant import *

def submission(model_name, zip_file):
    os.makedirs(model_name, exist_ok=True)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(model_name)
    upload_data = {}
    for file in os.listdir(model_name):
        if file.startswith('.') or file.startswith('__'):
            print(f"Skip the file: {file}")
            continue
        cur_file = os.path.join(model_name, file)
        if os.path.isdir(cur_file):
            for subfile in os.listdir(cur_file):
                if subfile.endswith('.json'):
                    with open(os.path.join(cur_file, subfile)) as ff:
                        cur_json = json.load(ff)
                        if isinstance(cur_json, dict):
                            for key in cur_json:
                                upload_data[key.replace('_', ' ')] = cur_json[key][0]
        elif cur_file.endswith('json'):
            with open(cur_file) as ff:
                cur_json = json.load(ff)
                if isinstance(cur_json, dict):
                    for key in cur_json:
                        upload_data[key.replace('_', ' ')] = cur_json[key][0]

        for key in TASK_INFO:
            if key not in upload_data:
                upload_data[key] = 0
    return upload_data

def get_final_score(score):
    values = [score[key] for key in TASK_INFO]
    return sum(values) / len(values)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Load submission file')
    parser.add_argument('--zip_file', type=str, required=True, help='Name of the zip file', default='evaluation_results.zip')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model', default='t2v_model')
    args = parser.parse_args()

    upload_dict = submission(args.model_name, args.zip_file)
    print(f"your submission info: \n{upload_dict} \n")
    final_score = get_final_score(upload_dict)
    print('+-------------------------------|------------------+')
    print(f'|                  complex plot|{upload_dict["Complex Plot"]}|')
    print(f'|             dynamic attribute|{upload_dict["Dynamic Attribute"]}|')
    print(f'|  dynamic spatial relationship|{upload_dict["Dynamic Spatial Relationship"]}|')
    print(f'|     motion order understanding|{upload_dict["Motion Order Understanding"]}|')
    print(f'|                    total score|{final_score}|')
    print('+-------------------------------|------------------+')
