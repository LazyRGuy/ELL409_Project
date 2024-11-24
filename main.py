import torch
import numpy as np
from torch import optim as optim
from nasadataset import Nasadataset
from torch.utils.data import DataLoader
from model import Model
from train import Trainer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    trainset = Nasadataset(mode='train',
                               dataset='./CMAPSSData/train_FD001_normed.txt')
    num_feat = trainset.num_cols
    to_drop = trainset.dropped
    trainset,valset = train_test_split(trainset,test_size=0.05,random_state=42)
    train_loader = DataLoader(dataset=trainset, batch_size=100, shuffle=True, num_workers=2)
    testset = Nasadataset(mode='test',
                              dataset='./CMAPSSData/test_FD001_normed.txt',
                              rul_result='./CMAPSSData/RUL_FD001.txt',dropped_cols = to_drop)
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=2)
    val_loader = DataLoader(dataset=valset,batch_size=64,shuffle=False,num_workers=2)
    print('dataset load successfully!')

    best_score_list = []
    best_RMSE_list = []
    test_score_list = []
    test_RMSE_list = []
    for iteration in range(4):
        print('---Iteration: {}---'.format(iteration + 1))
        model = Model(num_features = num_feat)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        epochs = 10
        trainer = Trainer(model=model,
                          model_optimizer=optimizer,
                          print_every=50,
                          epochs=epochs,
                          prefix='FD001')
        best_score, best_RMSE,test_score,test_RMSE = trainer.train(train_loader, test_loader,val_loader, iteration)
        best_score_list.append(best_score)
        best_RMSE_list.append(best_RMSE)
        test_score_list.append(test_score)
        test_RMSE_list.append(test_RMSE)

    best_score_list = np.array(best_score_list).reshape(-1)
    best_RMSE_list = np.array(best_RMSE_list).reshape(-1)
    test_score_list = np.array(test_score_list).reshape(-1)
    test_RMSE_list = np.array(test_RMSE_list).reshape(-1)
    print(best_score_list.shape,best_RMSE_list.shape)
    print(np.mean(best_score_list),np.mean(best_RMSE_list))
    print(np.mean(test_score_list),np.mean(test_RMSE_list))
    # result = np.concatenate(best_score_list, best_RMSE_list).reshape(2, 1)
    # np.savetxt('./{}_result.txt'.format(trainer.prefix), result, fmt='%.4f')
