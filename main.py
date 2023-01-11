
import numpy as np
import time
from LSTF_model import *
from function import *

def train():
    seed=123
    kfold=5
    numbers, Rs, Rs_indexes, Graphs, Graphs_indexes, id_mappings,genenames = load_data()  #读取数据
    inter_pairs = Rs[0]
    inter_pairs, noninter_pairs, pos_tests, neg_tests=cross_divide(kfold, inter_pairs, numbers[0], seed=seed)
    print("Data initlized!")

    gra=-10
    nn=4
    lamda_s=-10
    lamda_g=-10
    lamda_p=-8
    lamda_o=-10
    theta, maxiter = 2.0 ** (-3), 1000
    rank=160

    auc_pair, aupr_pair, recall_pair, precision_pair, accuracy_pair, f_measure_pair, MCC_pair = [], [], [], [], [], [], []
    t = time.time()
    pos_x, pos_y = inter_pairs[:, 0], inter_pairs[:, 1]

    # kfold折交叉验证
    for fold in range(kfold):
        pos_test=pos_tests[fold]
        neg_test=neg_tests[fold]

        pos_test_x,pos_test_y=inter_pairs[pos_test, 0],inter_pairs[pos_test, 1]
        neg_test_x,neg_test_y=noninter_pairs[neg_test, 0],noninter_pairs[neg_test, 1]


        model = LSTF_model(ranks=[160], nn_size=16, lamda_g=2**lamda_g, lamda_s=2**lamda_s,
                          lamda_gs=[2**lamda_p,2**lamda_o], numbers=numbers, theta=theta, max_iter=maxiter)

        print(str(model))
        SL_Mat = np.zeros((numbers[0], numbers[0]))
        SL_Mat[pos_x, pos_y] = 1
        SL_Mat[pos_y, pos_x] = 1
        SL_Mat[pos_test_x,pos_test_y]=0
        SL_Mat[pos_test_y, pos_test_x] = 0
        Rs[0] = SL_Mat


        model.train(Rs, Rs_indexes, Graphs, Graphs_indexes=Graphs_indexes, inter_pairs=inter_pairs,
                    noninter_pairs=noninter_pairs, pos_test=pos_test, neg_test=neg_test, Kfold=fold)

        reconstruct=model.reconstruct
        auc_val, aupr_val,recall_val,precision_val,accuracy_val,f_measure_val,MCC_val= evalution_all(reconstruct, inter_pairs[pos_test, :],noninter_pairs[neg_test, :])

        auc_pair.append(auc_val)
        aupr_pair.append(aupr_val)
        recall_pair.append(recall_val)
        precision_pair.append(precision_val)
        accuracy_pair.append(accuracy_val)
        f_measure_pair.append(f_measure_val)
        MCC_pair.append(MCC_val)
        # 打印当前结果并保存
        print(
            "metrics over protein pairs: auc %.6f, aupr %.6f,recall %.6f,precision %.6f,accuracy %.6f,f_measure %.6f,MCC %.6f, time: %f\n" % (
                auc_val, aupr_val, recall_val, precision_val, accuracy_val, f_measure_val, MCC_val, time.time() - t))

    m1, sdv1 = mean_confidence_interval(auc_pair)
    m2, sdv2 = mean_confidence_interval(aupr_pair)
    m3, sdv3 = mean_confidence_interval(recall_pair)
    m4, sdv4 = mean_confidence_interval(precision_pair)
    m5, sdv5 = mean_confidence_interval(accuracy_pair)
    m6, sdv6 = mean_confidence_interval(f_measure_pair)
    m7, sdv7 = mean_confidence_interval(MCC_pair)
    print("Average metrics over pairs: auc_mean:%.6f, aupr_mean:%.6f,recall_mean:%.6f,precision_mean:%.6f,accuracy_mean:%.6f,f_measure_mean:%.6f,MCC_mean:%.6f \n" % (
            m1, m2, m3, m4, m5, m6, m7))



if __name__ == "__main__":
    train()
