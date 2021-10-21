############################################################################
# Supervised Autoencoder for Language Detection                            #                     
#                                                                          # 
# Authors:                                                                 #
#         Shantipriya Parida (Idiap Research Institute, Switzerland),      #
#         Sajith Kumar (IITKGP, India)                                     #
#         Esau villatoro tello (Idiap Research Institute, Switzerland      #
#                               and Universidad Autónoma Metropolitana     #
#                               campus Cuajimalpa (UAM-C) in Mexico City.  #
############################################################################

import numpy as np
import collections
import sys
import torch
from torch import nn

class Autoencoder(nn.Module):
   
    def get_actf(self,actf_name):
        if actf_name is "relu":
            A = nn.ReLU()
        elif actf_name is "sigma":
            A = nn.Sigmoid()
        elif actf_name is "tanh":
            A = nn.Tanh()
        elif actf_name is "lrelu":
            A = nn.LeakyReLU()
        else:
            print("Unknown activation function: ",actf_name)
            sys.exit(1)
        return A
    
    def get_encoder(self):
        return self.encoder
    
    def get_encoder_params(self):
        params_list = []
        for temp in self.encoder.parameters():
            params_list.append(temp.cpu().data.numpy())
        return params_list

    def get_aec_params(self):
        params_list = []
        for temp in self.parameters():
            params_list.append(temp.cpu().data.numpy())
        return params_list        
    
    def __init__(self,input_dim, emb_dim, num_layers, activation, num_target, is_last_linear=False): 
        super(Autoencoder, self).__init__()

        step = round((input_dim - emb_dim)/ num_layers, -1)
        layers = []
        start = input_dim
        k = 1
        while k < num_layers:
            stop = start-step
            layers.append((start, stop))
            start = stop
            k+=1
        layers.append((start, emb_dim))
        
        #encoding layers
        enc_layers_dict = collections.OrderedDict()
        for i in range(num_layers):
            k1, k2 = layers[i]
            temp_layer = nn.Linear(int(k1), int(k2))
            enc_layers_dict["enc-"+str(i)] = temp_layer
            if not is_last_linear:
                enc_layers_dict["act-"+str(i)] = self.get_actf(activation) 
            else:
                if i != (num_layers-1):
                    enc_layers_dict["act-"+str(i)] = self.get_actf(activation)

        #decoding layers
        dec_layers_dict = collections.OrderedDict()
        layers.reverse()
        for i in range(num_layers):
            k2, k1 = layers[i]
            temp_layer = nn.Linear(int(k1), int(k2))
            dec_layers_dict["dec-"+str(i)] = temp_layer
            if i != (num_layers-1):
                dec_layers_dict["act-"+str(i)] = self.get_actf(activation)

        #supervised layer
        fc_dict = collections.OrderedDict()
        fc_layer = nn.Linear(int(emb_dim), int(num_target))
        fc_dict['fc-1'] = fc_layer
            
        self.encoder = nn.Sequential(enc_layers_dict)
        self.decoder = nn.Sequential(dec_layers_dict)
        self.fullyconnected = nn.Sequential(fc_dict)

        print("encoder: ")
        print(self.encoder)
        print("decoder: ")
        print(self.decoder)
        print("supervised: ")
        print(self.fullyconnected)
  

    def forward(self, x):
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        x_fc = self.fullyconnected(x_enc)

        return x_dec,x_enc,x_fc

