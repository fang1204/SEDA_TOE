import argparse
import random
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
import data_loader
import utils
from model import Model
import os
import torch.nn.functional as F
from enhance import enhance
import pandas as pd
from statistics import mean 
from operator import itemgetter
def merge_span(ner_start_end_dict):
    new_item_s_e = []
    for v in ner_start_end_dict:
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
    # mergedDataArray= np.array(mergedData)
    return mergedData
def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = LabelSmoothSoftmaxCEV1(lb_smooth=config.smoothing)

  
        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': config.bert_learning_rate,
                'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': config.bert_learning_rate,
                'weight_decay': 0.0},
            {'params': other_params,
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay},
        ]

        self.optimizer = optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.7)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    @staticmethod
    def multilabel_categorical_crossentropy(y_pred, y_true):
        """
        This function is a loss function for multi-label learning
        ref: https://kexue.fm/archives/7359

        y_pred: (batch_size_train, ... , type_size)
        y_true: (batch_size_train, ... , type_size)
        y_true and y_pred have the same shape，elements in y_true are either 0 or 1，
             1 tags positive classes，0 tags negtive classes(means tok-pair does not have this type of link).
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])# st - st
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return (neg_loss + pos_loss).mean()

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []
        pred_result_new = []
        label_result_new = []
        acc =0
        for i, data_batch in enumerate(tqdm(data_loader)):
            data_batch = [data.to(device) for data in data_batch[:-1]]

            word_inputs, bert_inputs, char_inputs, grid_labels, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d = data_batch
            
            outputs, outputs_new = model(word_inputs, bert_inputs, char_inputs, grid_mask2d, dist_inputs, pieces2word, word_mask2d)

            grid_mask2d = grid_mask2d.clone()

            loss = config.alpha * self.multilabel_categorical_crossentropy(outputs[grid_mask2d], grid_labels[grid_mask2d]) + (1 - config.alpha) * self.multilabel_categorical_crossentropy(outputs_new[grid_mask2d], grid_labels_new[grid_mask2d])
      
            loss/=config.accumulate_step
            loss.backward()
            acc+=1
            if acc == config.accumulate_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                acc=0
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            # self.optimizer.step()
            # self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            # outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)
            grid_labels_new = grid_labels_new[grid_mask2d].contiguous().view(-1)
            outputs_new = outputs_new[grid_mask2d].contiguous().view(-1)


            label_result.append(grid_labels)
            pred_result.append(outputs)
            label_result_new.append(grid_labels_new)
            pred_result_new.append(outputs_new)

            # self.scheduler.step()

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        label_result_new = torch.cat(label_result_new)
        pred_result_new = torch.cat(pred_result_new)
        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      np.int64(pred_result.cpu().detach().numpy()>0),
                                                      average="macro")

        p_new, r_new, f1_new, _ = precision_recall_fscore_support(label_result_new.cpu().numpy(),
                                                      np.int64(pred_result_new.cpu().detach().numpy()>0),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [(f1+f1_new)/2, (p+p_new)/2, (r+r_new)/2]])
        logger.info("\n{}".format(table))
        return (f1+f1_new)/2

    def predict(self, epoch, data_loader,data,now_dataset_format, generate_mode=False,enhance_id="0",enhance_count=-1):
        self.model.eval()

        pred_result = []
        label_result = []
        pred_result_new = []
        label_result_new = []
        loss_list = []
        find_entity_ratios = []
        find_area_ratios = []
        cc = 0
        # ent_r, ent_p, ent_c = 0, 0, 0
        predict_start_end = pd.DataFrame(columns=["file_id", "label", "start_end", "entity", "other"])

        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(data_loader)):
                sentence_batch = data[cc:cc+config.batch_size]
                entity_text = data_batch[-1]
                data_batch = [data.to(device) for data in data_batch[:-1]]
                word_inputs, bert_inputs, char_inputs, grid_labels, grid_labels_new, grid_mask2d, pieces2word, dist_inputs, word_mask2d = data_batch

                outputs, outputs_new = model(word_inputs, bert_inputs, char_inputs, grid_mask2d, dist_inputs, pieces2word, word_mask2d)
                length = word_inputs.ne(0).sum(dim=-1)
                loss = config.alpha * self.multilabel_categorical_crossentropy(outputs[grid_mask2d], grid_labels[grid_mask2d]) +\
                      (1 - config.alpha) * self.multilabel_categorical_crossentropy(outputs_new[grid_mask2d], grid_labels_new[grid_mask2d])
                loss_list.append(loss.cpu().item())

                grid_mask2d = grid_mask2d.clone()

                predict_dataframe,find_entity_ratio,find_area_ratio = utils.decode_pre(outputs.cpu().numpy(), entity_text, length.cpu().numpy(), sentence_batch,enhance_id,enhance_count,config)
                predict_start_end = pd.concat([predict_start_end,predict_dataframe])
                predict_start_end_all = predict_start_end
                find_entity_ratios += find_entity_ratio
                find_area_ratios += find_area_ratio
                # ent_r, ent_p, ent_c = utils.decode_pre(outputs.cpu().numpy(), entity_text, length.cpu().numpy(),sentence_batch)
                # total_ent_r += ent_r
                # total_ent_p += ent_p
                # total_ent_c += ent_c
                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)
                grid_labels_new = grid_labels_new[grid_mask2d].contiguous().view(-1)
                outputs_new = outputs_new[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels)
                pred_result.append(outputs)
                label_result_new.append(grid_labels_new)
                pred_result_new.append(outputs_new)
                cc += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        label_result_new = torch.cat(label_result_new)
        pred_result_new = torch.cat(pred_result_new)

        p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
                                                      np.int64(pred_result.cpu().detach().numpy()>0),
                                                      average="macro")

        p_new, r_new, f1_new, _ = precision_recall_fscore_support(label_result_new.cpu().numpy(),
                                                      np.int64(pred_result_new.cpu().detach().numpy()>0),
                                                      average="macro")
        if "cadec" in config.dataset:
            answer_start_end_path = "answer/cadec/cadec_"+ now_dataset_format +"_answer.json"
           
        elif "share13" in config.dataset:
            answer_start_end_path = "answer/share13/share13_"+ now_dataset_format +"_answer_our.json"
        else:
            answer_start_end_path = "answer/share14/share14_"+ now_dataset_format +"_all_answer_our.json"
        entity_set= {}
        entity_bound_set= {}
        with open(answer_start_end_path, 'r', encoding='utf-8') as json_file:
            answer_start_end = json.load(json_file)
            for file_name,start_end in answer_start_end.items():
                entity_set[file_name] = set(start_end)
                start_end = [[int(s_e.split(',')[0]),int(s_e.split(',')[-1])] for s_e in start_end]
                start_end = merge_span(start_end)

                entity_bound_set[file_name] = set([",".join(str(s_e)) for s_e in start_end])

        predict_start_end = predict_start_end.drop_duplicates(subset=['file_id','start_end'])
        predict_start_end['first_start'] = predict_start_end['start_end'].apply(lambda x: int(x.split(',')[0]))
        predict_start_end = predict_start_end.sort_values(by=['file_id','first_start']).drop('first_start', axis=1)
        predict_start_end['entity_bound'] = predict_start_end['start_end'].apply(lambda x: ",".join([x.split(',')[0],x.split(',')[-1]]))


        predict_start_end_all = predict_start_end_all.drop_duplicates(subset=['file_id','start_end'])
        predict_start_end_all['first_start'] = predict_start_end_all['start_end'].apply(lambda x: int(x.split(',')[0]))
        predict_start_end_all = predict_start_end_all.sort_values(by=['file_id','first_start']).drop('first_start', axis=1)
        predict_start_end_all['entity_bound'] = predict_start_end_all['start_end'].apply(lambda x: ",".join([x.split(',')[0],x.split(',')[-1]]))

        predict_start_end_group = predict_start_end.groupby(['file_id'])
        predict_start_end_all_group = predict_start_end_all.groupby(['file_id'])
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0

        total_entity_bound_r = 0
        total_entity_bound_p = 0
        total_entity_bound_c = 0
        for name in entity_set.keys():
            answer_entity = entity_set[name]
            ans_entity_bound = entity_bound_set[name]
            try:
                file_group = predict_start_end_group.get_group(name)
                predict_entity = set(file_group['start_end'].tolist())

                file_all_group = predict_start_end_all_group.get_group(name)
                predict_entity_bound = file_all_group['entity_bound'].tolist()
                predict_entity_bound = [list(map(int, s.split(','))) for s in predict_entity_bound]
                predict_entity_bound = merge_span(predict_entity_bound)
                # predict_entity_bound = set([s_e[-1] for s_e in predict_entity_bound])
                predict_entity_bound = set([",".join(map(str, s_e)) for s_e in predict_entity_bound])
                # ans_entity_bound = set(merge_span(ans_entity_bound))


                total_ent_r += len(answer_entity)
                total_ent_p += len(predict_entity)
                predict_in_answer = answer_entity.intersection(predict_entity)
                total_ent_c += len(predict_in_answer)

                total_entity_bound_r += len(ans_entity_bound)
                total_entity_bound_p += len(predict_entity_bound)
                predict_in_entity_bound = ans_entity_bound.intersection(predict_entity_bound)
                total_entity_bound_c += len(predict_in_entity_bound)
            except:
                total_ent_r+=len(answer_entity)
                total_ent_p+=0
                total_ent_c+=0
                total_entity_bound_r+=len(ans_entity_bound)
                total_entity_bound_p+=0
                total_entity_bound_c+=0

        if total_ent_r == 0 or total_ent_p == 0:
            e_f1, e_p, e_r = 0, 0, 0
        else:
            e_r = total_ent_c / total_ent_r
            e_p = total_ent_c / total_ent_p
            if total_ent_c == 0 or total_ent_r == 0:
                e_f1 = 0
            else:
                e_f1 = (2 * e_p * e_r / (e_p + e_r))
 
        entity_bound_r = total_entity_bound_c / total_entity_bound_r
        entity_bound_p = total_entity_bound_c / total_entity_bound_p
        entity_bound_f1 = (2 * entity_bound_p * entity_bound_r / (entity_bound_p + entity_bound_r))

        title = now_dataset_format.upper()
        logger.info('{} Label F1 {}'.format(title, 0.5 * (f1_score(label_result.cpu().numpy(),
                                                            np.int64(pred_result.cpu().detach().numpy()>0),
                                                            average=None) + f1_score(label_result_new.cpu().numpy(),
                                                            np.int64(pred_result_new.cpu().detach().numpy()>0),
                                                            average=None))))

        avg_find_entity_ratios,avg_find_area_ratios = mean(find_entity_ratios),mean(find_area_ratios)
        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall","Loss","find_entity_ratios","find_area_ratios"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [(f1+f1_new)/2, (p+p_new)/2, (r+r_new)/2]]+["{:.4f}".format(np.mean(loss_list))]+[entity_bound_f1,avg_find_area_ratios])
        table.add_row(["Entity"] + ["{:3.4f}".format(x) for x in [e_f1, e_p, e_r]]+["{:.4f}".format(np.mean(loss_list))]+[entity_bound_f1,avg_find_area_ratios])
        logger.info("\n{}".format(table))
        if generate_mode:
            if not os.path.exists("./ann/"+config.enhance_project+"/"+now_dataset_format):
                os.makedirs("./ann/"+config.enhance_project+"/"+now_dataset_format)
            predict_start_end.to_csv("./ann/"+config.enhance_project+"/"+now_dataset_format+"/toe_"+now_dataset_format+"_"+enhance_id+".ann", sep='\t', columns = ["file_id", "label", "start_end", "entity", "other"], index=False,header=None)
        return e_f1,entity_bound_f1,avg_find_area_ratios

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path, device=1):
        self.model.load_state_dict(torch.load(path, map_location=f'cuda:{device}'),strict=False)

def seed_torch(seed=3306):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import time
import json
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/cadec_fix_length.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--cr', type=int)


    args = parser.parse_args()
    
    bert_config = config.Config(args, is_bert=True)
    config = config.Config(args)
    
    logger = utils.get_logger(config)
    logger.info(config)
    config.logger = logger
    if torch.cuda.is_available():
        device = f"cuda:{args.device}"
    else:
        device = "cpu"

    seed_torch(config.seed)

    for enhance_id in range(config.enhance_count):
        logger.info("Loading Data")
        datasets, ori_data = data_loader.load_data_bert(config)

        # updates_total = 1
        updates_total = len(datasets[0]) // config.batch_size * config.epochs
        train_loader = DataLoader(dataset=datasets[0],
                    batch_size=config.batch_size,
                    shuffle=True,
                    collate_fn=data_loader.collate_fn,
                    num_workers=8,
                    )
        dev_loader = DataLoader(dataset=datasets[1],
                    batch_size=config.batch_size,
                    collate_fn=data_loader.collate_fn,
                    num_workers=8,
                    )
        test_loader = DataLoader(dataset=datasets[2],
                    batch_size=config.batch_size,
                    collate_fn=data_loader.collate_fn,
                    num_workers=8,
                    )

        logger.info("Building Model")
        model = Model(config, bert_config)

        model = model.to(device)

        trainer = Trainer(model)

        best_f1 = 0
        best_test_f1 = 0
        best_area_score = 0
        for i in range(config.epochs):
            logger.info("Epoch: {}".format(i))
            trainer.train(i, train_loader)
            f1,avg_find_entity_ratios,avg_find_area_ratios = trainer.predict(i, dev_loader, ori_data[1], "dev",generate_mode=False,enhance_id=enhance_id,enhance_count=config.enhance_count)
            test_f1,test_avg_entity_ratios,test_avg_area_ratios = trainer.predict(i, test_loader, ori_data[-1],"test",generate_mode=False,enhance_id=enhance_id,enhance_count=config.enhance_count)
            score = avg_find_entity_ratios
            # f1 = trainer.predict(i, dev_loader)
            # test_f1 = trainer.predict(i, test_loader, is_test=True)
            if enhance_id==config.enhance_count-1:
                # if f1 > best_f1:
                if score > best_area_score:
                # if test_f1 > best_test_f1:
                # if score > best_test_score:
                    best_area_score = score
                    best_f1 = f1
                    best_test_f1 = test_f1
                    trainer.save(config.save_path)
            else:
                if score > best_area_score:
                    best_area_score = score
                    best_f1 = f1
                    best_test_f1 = test_f1
                    trainer.save(config.save_path)
        logger.info("Best DEV F1: {:3.4f}".format(best_f1))
        logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
        logger.info("Best area F1: {:3.4f}".format(best_area_score))
        enhance_id = str(enhance_id)
        enhace_train_loader = DataLoader(dataset=datasets[0],
                    batch_size=config.batch_size,
                    collate_fn=data_loader.collate_fn,
                    num_workers=8,
                )
        trainer.load(config.save_path,args.device)
        trainer.predict("Final", enhace_train_loader, ori_data[0],"train",generate_mode=config.generate_mode,enhance_id=enhance_id,enhance_count=config.enhance_count)
        trainer.predict("Final", dev_loader, ori_data[1],"dev",generate_mode=config.generate_mode,enhance_id=enhance_id,enhance_count=config.enhance_count)
        trainer.predict("Final", test_loader, ori_data[-1],"test",generate_mode=config.generate_mode,enhance_id=enhance_id,enhance_count=config.enhance_count)
        if config.generate_mode:
            enhance_train_file = enhance(enhance_id,config,logger,"train","our")
            config.train_file = enhance_train_file

            enhance_dev_file = enhance(enhance_id,config,logger,"dev","our")
            config.dev_file = enhance_dev_file

            enhance_test_file = enhance(enhance_id,config,logger,"test","our")
            config.test_file = enhance_test_file
            if "cadec" in config.dataset:
                config.batch_size = 8                
            else:
                if "share14" in config.dataset:
                    config.warm_factor=0.1
                    config.dilation = [1]
                else:
                    config.epochs = 20

                config.weight_decay=0.4
                config.batch_size = 10
            config.accumulate_step = 2