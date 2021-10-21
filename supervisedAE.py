import numpy as np
import pickle as pkl
import time
import itertools
import pprint
import scipy

import torch
from torch.utils.data import TensorDataset, DataLoader

from aec import Autoencoder

class SupervisedAE:

    def __init__(self, input_dim, emb_dim, num_targets, num_chunks=5, num_layers=None, \
        learning_rate=None, weight_decay=None, convg_thres=1e-5, max_epochs=500, activation=None, \
        sup_loss_weight=1, is_last_linear=False, supervision='clf', is_gpu=False, \
        gpu_ids = "0"):
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_targets = num_targets
        self.num_chunks = num_chunks
        self.num_layers = num_layers
        assert supervision in ['clf', 'reg'], 'Provide \'reg\' or \'clf\' as supervision inputs'
        self.activation = activation
        self.supervise = supervision
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.convg_thres = convg_thres
        self.max_epochs = max_epochs
        self.sup_loss_weight = sup_loss_weight
        self.is_last_linear=is_last_linear
        self.is_gpu = is_gpu
        self.gpu_ids = gpu_ids
        
        self.print_params()

    def print_params(self):
        print("#")
        print("SAE: ")
        print("#")
        print("input_dim: ",self.input_dim)
        print("emb_dim: ",self.emb_dim)
        print("num_targets: ",self.num_targets)
        print("num_layers: ",self.num_layers)
        print("learning_rate: ",self.learning_rate)
        print("weight_decay: ",self.weight_decay)
        print("convg_thres: ",self.convg_thres)
        print("max_epochs: ",self.max_epochs)
        print("num_chunks: ",self.num_chunks)
        print("activation: ",self.activation)
        print("Last Layer Linear: ", self.is_last_linear)
        print("Supervision: ", self.supervise)
        print("is_gpu: ",self.is_gpu)
        print("gpu_ids: ",self.gpu_ids)
        print("#")

    def __is_converged(self,prev_cost,cost,convg_thres):
        diff = (prev_cost - cost)
        if (abs(diff)) < convg_thres:
            return True

    def __input_transformation(self, X, y):
        print("__input_transformation - start")
        print("#")
         #chunking
        assert X.shape[1] == self.input_dim, 'The input features shape is not equal to input dim.'
        #assert that num_chunks is not > number of datapoints
        assert self.num_chunks <= X.shape[0], 'The num_chunks must be <= minimum entity size in the setting.'
        #warn if the encoding length k is not > than minimum of the feature lengths
        assert X.shape[1] >= self.emb_dim, 'The encoding length k is larger than minimum entity feature size.'
        X_temp = torch.from_numpy(X).float()
        X_chunks_list = torch.chunk(X_temp,self.num_chunks,dim=0)
        print("X_chunks_list length: ",len(X_chunks_list))
        print("X_chunks_list[0].shape: ",X_chunks_list[0].shape)
        print("creating pytorch variables of target variables...")
        if self.supervise == 'reg':
            y_temp = torch.from_numpy(y).float()
        else:
            y_temp = torch.from_numpy(y).long()
        
        y_chunks_list = torch.chunk(y_temp,self.num_chunks,dim=0)
        print("y_chunks_list length: ",len(y_chunks_list))
        print("y_chunks_list[0].shape: ",y_chunks_list[0].shape)
             
        print("#")
        print("__input_transformation - end")
        return (X_chunks_list, y_chunks_list)

    def __network_construction(self):
        #for each entity construct an autoencoder
        #Building aec_dict
        print("__network_construction - start") 
        device = torch.device("cuda:"+str(self.gpu_ids) if self.is_gpu else "cpu")
        self.AEC = Autoencoder(self.input_dim,self.emb_dim, self.num_layers,self.activation,self.num_targets,self.is_last_linear)
        self.AEC = self.AEC.to(device)
        
        print("__network_construction - end")
    
    def __getCriterion(self):
        if self.supervise == 'clf':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = torch.nn.MSELoss()

        return criterion
    
    def fit(self, X, y):
        #dcmf model construction
        print("SAE - model construction - start")
        X_chunks_list, y_chunks_list = self.__input_transformation(X, y)
        self.__network_construction()
        print("SAE - model construction - end")
        print("#")
        print("SAE.fit - start")
        device = torch.device("cuda:"+str(self.gpu_ids) if self.is_gpu else "cpu")
        #opt algo setup
        criterion = torch.nn.MSELoss()
        supervise_criterion = self.__getCriterion()
        
        model_params = []
        params_temp = list(self.AEC.parameters())
        model_params+=params_temp
        optimizer = torch.optim.Adam(model_params, lr=self.learning_rate, weight_decay=self.weight_decay)
        #main loop - training - start
        
        dict_epoch_loss = {}
        dict_epoch_aec_rec_loss = {}
        dict_epoch_supervise_loss = {}
        epoch = 1
        prev_loss_epoch = 0
        while True:
            if epoch > self.max_epochs:
                break
            #epoch - start
            s = time.time()
            loss_epoch = 0
            loss_aec_rec_epoch = 0
            loss_supervise_epoch = 0
            U_chunks_list = []
            #chunks processing - start
            for i in np.arange(self.num_chunks):
                optimizer.zero_grad()
                #load current batch data 
                X_chunk_batch = X_chunks_list[i]
                y_chunk_batch = y_chunks_list[i]
                X_chunk_batch = X_chunk_batch.to(device)
                y_chunk_batch = y_chunk_batch.to(device)
                #train
                #autoencoder reconstruction - chunk
                X_chunk_rec,U_chunk, y_chunk_pred = self.AEC(X_chunk_batch)
                #loss
                #autoencoder reconstruction loss - chunk
                loss_aec_rec = criterion(X_chunk_rec,X_chunk_batch)
                ## supervision loss
                loss_supervise = supervise_criterion(torch.squeeze(y_chunk_pred), y_chunk_batch)
                #sum all losses
                loss = loss_aec_rec + self.sup_loss_weight*loss_supervise
                #backprop
                loss.backward()
                optimizer.step()
                #update - loss info
                loss_epoch += loss.item()
                loss_aec_rec_epoch += loss_aec_rec.item()
                loss_supervise_epoch += loss_supervise.item()
                
                U_chunks_list.append(U_chunk)
            #chunks processing - end
            dict_epoch_loss[epoch] = loss_epoch
            dict_epoch_aec_rec_loss[epoch] = loss_aec_rec_epoch
            dict_epoch_supervise_loss[epoch] = loss_supervise_epoch
            #epoch - end
            e = time.time()
            # print("AEC loss:", loss_aec_rec_epoch, 
            #     "Supervision loss:", loss_supervise_epoch)
            print("epoch: ",epoch," total loss L: ",loss_epoch," Took ",round(e-s,1)," secs.")
            #update - counter
            epoch+=1
            if self.__is_converged(prev_loss_epoch,loss_epoch,self.convg_thres):
                print("**train converged**")
                break
            prev_loss_epoch = loss_epoch
        #main loop - training - end
        #Build entity representations
        self.loss = loss_epoch
        X_emb = torch.cat(U_chunks_list)
        print("#")
        print("SAE.fit - end")
        return X_emb

    def predict(self, X, num_chunks=1):
        assert X.shape[1] == self.input_dim, 'The input features shape is not equal to input dim.'
        U_list = []
        y_pred_list = []
        device = torch.device("cuda:"+str(self.gpu_ids) if self.is_gpu else "cpu")
        X_chunk_list = torch.chunk(torch.from_numpy(X).float(), num_chunks, dim=0)
        with torch.no_grad():
            for i in range(num_chunks):
                X_temp = X_chunk_list[i]        
                X_temp = X_temp.to(device)
                _, U, y_pred = self.AEC(X_temp)
                y_pred_list.append(y_pred.cpu())
                U_list.append(U.cpu())
        
        result = (torch.squeeze(torch.cat(y_pred_list)), torch.cat(U_list))
        return result

    def save(self, path):
        torch.save(self, path)

    def set_params(self, params):
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        
        if 'weight_decay' in params:
            self.weight_decay = params['weight_decay']

        if 'num_layers' in params:
            self.num_layers = params['num_layers']

        if 'convg_thres' in params:
            self.convg_thres = params['convg_thres'] 

        if 'activation' in params:
            self.activation = params['activation']   
