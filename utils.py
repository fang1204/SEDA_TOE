import logging
import time
import pickle
import pandas as pd
import numpy as np
from operator import itemgetter
import data_loader


def get_logger(config):
    pathname = "./log/{}_{}_{}_{}_{}_{}_{}_{}.txt".format(config.dataset, config.seed, config.dilation, config.conv_hid_size, config.rounds, config.batch_size, config.alpha, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index):
    text = "-".join([str(i) for i in index])
    return text

def decode(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        forward_dict = {}
        head_dict = {}

        # # T,L
        for i in range(l):
            for j in range(l):
                for f in range(4):
                    if instance[i, j, f] > 0:
                        if f == 0 and j > i:
                            if instance[j, i, 1] > 0:
                                if i not in forward_dict:
                                    forward_dict[i] = [j]
                                else:
                                    forward_dict[i].append(j)
                                forward_dict[i] = list(set(forward_dict[i]))  
                        elif f == 2 and j >= i:
                            if i not in head_dict:
                                head_dict[i] = {j}
                            else:
                                head_dict[i].add(j)
                        elif f == 3 and j <= i:
                            if j not in head_dict:
                                head_dict[j] = {i}
                            else:
                                head_dict[j].add(i)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()
 
        for head in head_dict:
            find_entity(head, [], head_dict[head])


        predicts = set([convert_index_to_text(x) for x in predicts])
        ent_r += len(ent_set)
        ent_p += len(predicts)
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_r, ent_p, ent_c

def decode_pre(outputs, entities, length, sentence_batch,enhance_id,enhance_count,config):
    find_entity_ratio=[]
    find_area_ratio=[]
    predict_dataframe = pd.DataFrame(columns=["file_id", "label", "start_end", "entity", "other"])
    for index, (instance, ent_set, l,sentence_item) in enumerate(zip(outputs, entities, length,sentence_batch)):
        forward_dict = {}
        head_dict = {}
        label_list = []
# 80 81 82 81
        # # T,L
        for i in range(l):
            for j in range(l):
                for f in range(4): 
                    # 0 ,1,2,3
                    if instance[i, j, f] > 0:
                        if f == 0 and j > i:
                            if instance[j, i, 1] > 0:
                                if i not in forward_dict:
                                    forward_dict[i] = [j]
                                else:
                                    forward_dict[i].append(j)
                                forward_dict[i] = list(set(forward_dict[i]))  
                        elif f == 2 and j >= i:
                            if i not in head_dict:
                                head_dict[i] = {j}
                            else:
                                head_dict[i].add(j)
                            label_list.append((list(range(i,j+1))))
                        elif f == 3 and j <= i:
                            if j not in head_dict:
                                head_dict[j] = {i}
                            else:
                                head_dict[j].add(i)
                            label_list.append((list(range(j,i+1))))
        # 5,2  5,10 ht
        predicts = []
        dict_check={}
        res_list = []
        for item in label_list:
            x = tuple(item)
            if x not in dict_check:
                dict_check[x]=1
            else:
                dict_check[x]+=1
            if dict_check[x]==2 and list(x) not in res_list:
                res_list.append(list(x))
        label_list = res_list
        # label_list = {tuple(item): item for item in label_list}
        # label_list = list(label_list.values())

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()
 
        for head in head_dict:
            find_entity(head, [], head_dict[head])


        # if int(enhance_id)==enhance_count-1:
        if predicts!=[]:
            for p_item in predicts:  

                predict_index = [sentence_item['word_start_end'][ind] for ind in p_item]
                predict_index = ",".join(str(j) for i in predict_index for j in i)
                # out_f.write(str(predict_index)+"\t")                           #索引
                predict_word = [sentence_item['sentence'][ind] for ind in p_item]
                predict_word = ",".join(str(i) for i in predict_word)
                if "cadec" in config.dataset:
                    lable="adr"
                else:
                    lable="disorder"
                data_to_append = pd.DataFrame([{
                    "file_id": sentence_item['filename'], 
                    "label": lable, 
                    "start_end": predict_index,
                    "entity": predict_word, 
                    "other": ""
                }])
                predict_dataframe = predict_dataframe.append(data_to_append)

        if predicts!=[] or label_list!=[]:
            if label_list!=[]:
                predicts.extend(label_list)
                predicts = {tuple(item): item for item in predicts}
                predicts = list(predicts.values())
            # -------------#
            p_ner_start_end_dict ={}
            for p_item in predicts:  

                if p_item[0] not in p_ner_start_end_dict:
                    p_ner_start_end_dict[p_item[0]] = [p_item[0], p_item[-1]]
                else:
                    p_ner_start_end_dict[p_item[0]] = [p_item[0],max(p_ner_start_end_dict[p_item[0]][-1],p_item[-1])]
                
            p_merge = merge_span(p_ner_start_end_dict)
            
        if sentence_item['ner']!=[] and predicts!=[]:
            g_ner_start_end_dict={}
            for ner in sentence_item['ner']:
                if ner['index'][0] not in g_ner_start_end_dict:
                    g_ner_start_end_dict[ner['index'][0]] = [ner['index'][0], ner['index'][-1]]
                else:
                    g_ner_start_end_dict[ner['index'][0]] = [ner['index'][0],max(g_ner_start_end_dict[ner['index'][0]][-1],ner['index'][-1])]
            g_merge = merge_span(g_ner_start_end_dict)
            # g_list = list(range(g_merge[0][0],g_merge[-1][-1]))
            # p [4,10,16]  [1, 0, 0]
            # g 4 
            find = [np.isin(row[1], p_merge[:,1]).all() for row in g_merge]
            find_entity_ratio.append(sum(find)/p_merge.shape[0])

            g_list= []
            p_list= []
            for index, row in enumerate(g_merge):
                g_list.extend(list(range(row[0],row[-1]+1)))
            for index, row in enumerate(p_merge):
                p_list.extend(list(range(row[0],row[-1]+1)))
            find_area = np.isin(np.array(g_list), np.array(p_list))
            find_area_ratio.append(find_area.sum(-1)/len(p_list))
        elif sentence_item['ner']!=[] and predicts==[]:
            find_entity_ratio.append(0.0)
            find_area_ratio.append(0.0)

    return predict_dataframe, find_entity_ratio, find_area_ratio
    # return predict_dataframe,find_entity_ratio,find_area_ratio,predict_dataframe
    # return predict_dataframe,find_entity_ratio,find_area_ratio,predict_dataframe_all

def merge_span(ner_start_end_dict):
    new_item_s_e = []
    for k, v in ner_start_end_dict.items():
        new_item_s_e.append(v)
    new_item_s_e = sorted(new_item_s_e,key=itemgetter(0))
    #[12,20],[13,19],[14,21] => [12,21]
    mergedData = []
    start, end = new_item_s_e[0]
    for pair in new_item_s_e[1:]:
        if pair[0] <= end:
            end = max(end, pair[1])
        else:
            mergedData.append([start, end])
            start, end = pair

    mergedData.append([start, end])
    mergedDataArray= np.array(mergedData)
    return mergedDataArray

def decode_without_disconnect(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):

        ht_type_dict = {}
        predicts = []
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    predicts.append(list(range(i, j + 1)))

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

        ent_r += len(ent_set)
        ent_p += len(predicts)
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return ent_r, ent_p, ent_c