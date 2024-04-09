
import time
import datetime
import numpy as np 
import Constants
import torch
from torch.utils.data import DataLoader
from dataLoader import datasets, Read_data, Split_data

from tqdm import tqdm
from utils.parsers import parser
from utils.util import *
from utils.EarlyStopping import *
from utils.Metrics import Metrics
from utils.graphConstruct import ConRelationGraph, ConHypergraph

from model.model import MIMIDP

metric = Metrics()
opt = parser.parse_args() 

def init_seeds(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_performance(crit, pred, gold):

    loss = crit(pred, gold.contiguous().view(-1))
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()
    return loss, n_correct

   
def model_training(model, train_loader, epoch):
    ''' model training '''
    
    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    print('start training: ', datetime.datetime.now())
    model.train()
    for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(train_loader):

        n_words = label.data.ne(Constants.PAD).sum().float().item()
        n_total_words += n_words

        model.zero_grad()
        cascade_item = trans_to_cuda(cascade_item.long())
        tar = trans_to_cuda(label.long())
        cascade_time = trans_to_cuda(cascade_time.long())
        label_time = trans_to_cuda(label_time.long())

        output_past, future_output, past_dist, futrue_dist, past_emb, future_emb = model(cascade_item, tar, cascade_time, label_time)
        
        loss, n_correct = get_performance(model.loss_function, output_past, tar)
        future_loss, fure_n_correct = get_performance(model.loss_function, future_output, cascade_item)

        loss2 = model.kl_loss(past_dist, futrue_dist)
        loss3 = model.seq2seqloss(past_emb, future_emb, cascade_item)
        loss = loss +  opt.beta* future_loss + opt.beta2 * loss2 + opt.beta3 * loss3
    
        loss.backward()
        model.optimizer.step()
        model.optimizer.update_learning_rate()

        if torch.isinf(model.user_embedding.weight).any():
            print(0)
        
        if torch.isnan(model.user_embedding.weight).any():
                print(0)

        total_loss += loss.item()
        n_total_correct += n_correct

        torch.cuda.empty_cache()

    print('\tTotal Loss:\t%.3f' % total_loss)

    return total_loss, n_total_correct/n_total_words

def model_testing(model, test_loader, k_list=[10, 50, 100]):
    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0

    n_total_words = 0.0
    n_correct = 0.0
    total_loss = 0.0

    print('start predicting: ', datetime.datetime.now())
    model.eval()

    with torch.no_grad():
        for step, (cascade_item, label, cascade_time, label_time, cascade_len) in enumerate(test_loader):

            cascade_item = trans_to_cuda(cascade_item.long())
            cascade_time = trans_to_cuda(cascade_time.long())

            y_pred = model.model_prediction(cascade_item, cascade_time)

            y_pred = y_pred.detach().cpu()
            tar = label.view(-1).detach().cpu()

            pred = y_pred.max(1)[1]
            gold = tar.contiguous().view(-1)
            correct = pred.data.eq(gold.data)
            n_correct = correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

            scores_batch, scores_len = metric.compute_metric(y_pred, tar, k_list)
            n_total_words += scores_len

            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

        for k in k_list:
            scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
            scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words

        return scores, n_correct/n_total_words

def train_test(epoch, model, train_loader, val_loader, test_loader):

    total_loss, _ = model_training(model, train_loader, epoch)
    val_scores, val_accuracy = model_testing(model, val_loader)
    test_scores, test_accuracy = model_testing(model, test_loader)

    return total_loss, val_scores, test_scores, val_accuracy.item(), test_accuracy.item()

def main(data_path, seed=2023):

    init_seeds(seed)
    if opt.preprocess:
        Split_data(data_path, train_rate=opt.train_rate, valid_rate=opt.valid_rate, load_dict=True)
    train, valid, test, user_size = Read_data(data_path)

    train_data = datasets(train, opt.max_lenth)
    val_data = datasets(valid, opt.max_lenth)
    test_data = datasets(test, opt.max_lenth)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    opt.n_node = user_size
    relation_graph = ConRelationGraph(data_path)
    HG_Item, HG_User = ConHypergraph(opt.data_name, opt.n_node, opt.window)
    save_model_path = opt.save_path
    early_stopping = EarlyStopping(patience=opt.patience, verbose=True, path=save_model_path)
    model = trans_to_cuda(MIMIDP(graph = relation_graph,  hypergraphs=[HG_Item, HG_User], opt = opt, reverse = True, dropout=opt.dropout))

    top_K = [10, 50, 100]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    validation_history = 0.0
    print(opt)
    for epoch in range(opt.epoch):
        print('\n[ Epoch', epoch, ']')
        total_loss, val_scores, test_scores, val_accuracy, test_accuracy = train_test(epoch, model, train_loader, val_loader, test_loader)

        if validation_history <= sum(val_scores.values()):
            validation_history = sum(val_scores.values())
        
        print('  - ( Validation )) ')
        for metric in val_scores.keys():
            print(metric + ' ' + str(val_scores[metric]))

            for K in top_K:
                test_scores['hits@' + str(K)] = test_scores['hits@' + str(K)] * 100
                test_scores['map@' + str(K)] = test_scores['map@' + str(K)] * 100

                best_results['metric%d' % K][0] = test_scores['hits@' + str(K)]
                best_results['epoch%d' % K][0] = epoch
                best_results['metric%d' % K][1] = test_scores['map@' + str(K)]
                best_results['epoch%d' % K][1] = epoch


        early_stopping(-sum(list(val_scores.values())), model)
        if early_stopping.early_stop:
            print("Early_Stopping")
            break
    print(" -(Finished!!) \n parameter settings: ")
    print("--------------------------------------------")    
    print(opt)

    print(" -(Finished!!) \n test scores: ")
    print("--------------------------------------------")
    for K in top_K:
        print('Recall@%d: %.4f\tMAP@%d: %.4f\tEpoch: %d,  %d' %
              (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
               best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))



if __name__ == "__main__": 
    main(opt.data_name, opt.seed)



