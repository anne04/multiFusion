import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from multiFusion_train_util import shuffle_data #, data_to_tensor

class multiFusion(torch.nn.Module):
    def __init__(self, 
                channel_count_adc:int,
                channel_count_hbv:int,
                channel_count_t2w:int,

                adc_H: int,
                adc_W: int,

                hbv_H: int,
                hbv_W: int,

                t2w_H: int,
                t2w_W: int,

                filter_count_layer1:int = 4, 
                filter_count_layer2:int = 8,  

                kernel_size1:int = 4,
                kernel_size2:int = 2,

                hidden_size_predictor_layer1:int = 128,
                hidden_size_predictor_layer2:int = 64 
                ):
        """
        This will initialize the model and return it.
        Args:
        input_size: concat size of gene embedding & protein embedding for a lig/rec gene
        hidden_size_fusion and output_size_fusion: hidden/output layer dimensions for emb fusion
        hidden_size_predictor_layer1 & hidden_size_predictor_layer2: hidden layers for ppi pred
        """
        super().__init__() # error happens without this 

        # Branch: adc --> should be replaced by the pretrained feature extractor
        self.adc_feature_layer = nn.Sequential(
            nn.Conv2d(in_channels = channel_count_adc, out_channels = filter_count_layer1, kernel_size = kernel_size1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = filter_count_layer1, out_channels = filter_count_layer2, kernel_size = kernel_size2, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Branch: hbv --> is there any pretrained model for hbv type?
        self.hbv_feature_layer = nn.Sequential(
            nn.Conv2d(in_channels = channel_count_hbv, out_channels = filter_count_layer1, kernel_size = kernel_size1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels = filter_count_layer1, out_channels = filter_count_layer2, kernel_size = kernel_size2, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )

        # Branch: t2w --> should be replaced by the pretrained feature extractor
        self.t2w_feature_layer = nn.Sequential(
            nn.Conv2d(in_channels = channel_count_t2w, out_channels = filter_count_layer1, kernel_size = kernel_size1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels = filter_count_layer1, out_channels = filter_count_layer2, kernel_size = kernel_size2, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten()
        )
        '''
        # Branch: clinical (input like a feature vector)
        self.clinical_feature_layer = nn.Sequential(
          nn.Linear(input_size, hidden_size_fusion),
          nn.BatchNorm1d(hidden_size_fusion),
          nn.ReLU(),
          nn.Dropout(0.5),
          nn.Linear(hidden_size_fusion, output_size_fusion),
          nn.BatchNorm1d(output_size_fusion),
          nn.ReLU(),
          nn.Dropout(0.5)
        )        
        '''

        # calculate the flatten size for multi modal branches before infusion
        flatten_size_adc = filter_count_layer2 * (adc_H/2)/2 * (adc_W/2)/2 
        flatten_size_hbv = filter_count_layer2 * (hbv_H/2)/2 * (hbv_W/2)/2 
        flatten_size_t2w = filter_count_layer2 * (t2w_H/2)/2 * (t2w_W/2)/2 

        print(flatten_size_adc)
        print(flatten_size_hbv)
        print(flatten_size_t2w)

        input_size_fusion = int(flatten_size_adc + flatten_size_hbv + flatten_size_t2w) 
        print('input_size_fusion %d'%input_size_fusion)
        print('hidden_size_predictor_layer1 %d'%hidden_size_predictor_layer1)
        # fuse all and pass through MLP classification layer 
        self.fusionNpredict_layer = nn.Sequential(
            nn.Linear(input_size_fusion, hidden_size_predictor_layer1), 
            # use output_size_fusion*5 if clinical feature is used
            nn.BatchNorm1d(hidden_size_predictor_layer1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_predictor_layer1, hidden_size_predictor_layer2),
            nn.BatchNorm1d(hidden_size_predictor_layer2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size_predictor_layer2, 1),
            nn.Sigmoid() # if <0.5 --> no recurrence, >=0.5 --> yes recurrence
        )

    def forward(self, 
                adc_ftr: torch.Tensor, 
                hbv_ftr: torch.Tensor,
                t2w_ftr: torch.Tensor, 
                #clinical_ftr: torch.Tensor, 
                )-> torch.Tensor:
        
        adc_emb = self.adc_feature_layer(adc_ftr)
        hbv_emb = self.hbv_feature_layer(hbv_ftr)
        t2w_emb = self.t2w_feature_layer(t2w_ftr)

        #clinical_emb = self.clinical_feature_layer(clinical_ftr)

        #print(adc_emb.shape)
        #print(hbv_emb.shape)
        #print(t2w_emb.shape)
        # try with and without normalizing (usually normalizing is preferred before infusion)
        adc_emb = F.normalize(adc_emb, p=2) 
        hbv_emb = F.normalize(hbv_emb, p=2) 
        t2w_emb = F.normalize(t2w_emb, p=2) 
        #clinical_emb = F.normalize(adc_emb, p=2) 
        #print(adc_emb.shape)
        #print(hbv_emb.shape)
        #print(t2w_emb.shape)

        concat_emb = torch.cat((adc_emb, hbv_emb, t2w_emb), dim=1)
        #print(concat_emb.shape)
        #concat_emb = torch.cat((adc_emb, hbv_emb, t2w_emb, clinical_emb), dim=1)
        recur_prediction = self.fusionNpredict_layer(concat_emb)
        return recur_prediction




def train_multiFusion(
    args,
    metadata_adc,
    metadata_hbv, 
    metadata_t2w,
    training_set,
    validation_set, 
    epoch = 2000,
    batch_size = 32,
    learning_rate =  1e-4,
    threshold_score = 0.5,
    ):
    """
    args:
    training_set: torch.Tensor of training samples (80%)
    validation_set: torch.Tensor of validation samples (20%)
    val_class: list() of validation samples but with binary label (0/1)
    threshold_score: some cutoff to set binary labels
    """
    
    # CHECK = set a manual seed?
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize the model
    model_multiFusion = multiFusion(
                channel_count_adc = metadata_adc[0],
                channel_count_hbv = metadata_hbv[0],
                channel_count_t2w = metadata_t2w[0],

                adc_H = metadata_adc[1],
                adc_W = metadata_adc[2],

                hbv_H = metadata_hbv[1],
                hbv_W = metadata_hbv[2],

                t2w_H = metadata_t2w[1],
                t2w_W = metadata_t2w[2]
                ).to(device) 
    
    print(model_multiFusion)
    # set the loss function
    loss_function = nn.CrossEntropyLoss() #nn.BCELoss()

    # set optimizer
    optimizer = torch.optim.Adam(model_multiFusion.parameters(), lr=learning_rate)
    epoch_interval = 20 # CHECK
    #### for plotting loss curve ########
    loss_curve = np.zeros((epoch//epoch_interval+1, 4))
    loss_curve_counter = 0
    ######################################
    total_training_samples = training_set[0].shape[0]
    total_batch = total_training_samples//batch_size
    min_loss = 100000 # just a big number to initialize
    for epoch_indx in range (0, epoch):
        # shuffle the training set and split into multiple branches
        training_adc_ftr, training_hbv_ftr, training_t2w_ftr, training_prediction = shuffle_data(training_set)        
        model_multiFusion.train() # training mode
        total_loss = 0
        for batch_idx in range(0, total_batch):
            optimizer.zero_grad() # clears the grad, otherwise will add to the past calculations
            # get the batch of the training features and move to GPU
            batch_adc_ftr = training_adc_ftr[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :].to(device)
            batch_hbv_ftr = training_hbv_ftr[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :].to(device)
            batch_t2w_ftr = training_t2w_ftr[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :].to(device)
            batch_target = training_prediction[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            # run the model and get prediction
            batch_prediction = model_multiFusion(batch_adc_ftr, batch_hbv_ftr, batch_t2w_ftr)

            ### for debug purpose to see if the weights are changing ############
            if batch_idx == 0 and epoch_indx%epoch_interval == 0:
                print('debug: training:')
                print(batch_target[0:10])
                print(list(batch_prediction.flatten().cpu().detach().numpy())[0:10])

            #####################################################################
            # get the loss and backpropagate 
            loss = loss_function(batch_prediction.flatten(), batch_target.flatten())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # all batches are done     
        avg_loss = total_loss/total_batch
        ## if min loss found, log it ##
        if epoch_indx%epoch_interval == 0:
            #print('Epoch %d/%d, Training loss: %g'%(epoch_indx, epoch, avg_loss))            
            # run validation
            # CHECK: if you use dropout layer, you might need to set some flag during inference step 
            validation_adc_ftr = validation_set[0]
            validation_hbv_ftr = validation_set[1]
            validation_t2w_ftr = validation_set[2]
            validation_prediction = validation_set[3]
            # .to(device) to transfer to GPU
            batch_adc_ftr = validation_adc_ftr.to(device)
            batch_hbv_ftr = validation_hbv_ftr.to(device)
            batch_t2w_ftr = validation_t2w_ftr.to(device)
            batch_target = validation_prediction.to(device)
            ##### run the prediction ##########
            model_multiFusion.eval()
            batch_prediction = model_multiFusion(batch_adc_ftr, batch_hbv_ftr, batch_t2w_ftr)
            validation_loss = loss_function(batch_prediction.flatten(), batch_target.flatten())
            if epoch_indx==0:
                min_loss = validation_loss
            print('Epoch %d/%d, Training loss: %g, val loss: %g'%(epoch_indx, epoch, avg_loss, validation_loss))
            if validation_loss <= min_loss:
                min_loss = validation_loss
                # state save
                torch.save(model_multiFusion, args.model_name)  
                # model = torch.load("my_model.pickle")
                #
                # torch.save(model_multiFusion.state_dict(), "model/my_model_multiFusion_state_dict.pickle")
                # model = nn.Sequential(...)
                # model.load_state_dict(torch.load("my_model.pickle"))
                print('*** min loss found! %g ***'%validation_loss)


            #########
            batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
            batch_target = list(batch_target.flatten().cpu().detach().numpy())
            ### just for debug ###
            #print(batch_prediction[0:10])
            #print(batch_target[0:10])
            #######################

            for i in range(0, len(batch_prediction)):
                if batch_prediction[i]>= threshold_score:
                    batch_prediction[i] = 1
                else:
                    batch_prediction[i] = 0

            
            pred_class = batch_prediction

            TP = TN = FN = FP = 0
            P = N = 0
            for i in range (0, len(batch_target)):
                if batch_target[i] == 1 and pred_class[i] == 1:
                    TP = TP + 1
                    P = P + 1
                elif batch_target[i] == 1 and pred_class[i] == 0:
                    FN = FN + 1
                    P = P + 1
                elif batch_target[i] == 0 and pred_class[i] == 1:
                    FP = FP + 1
                    N = N + 1
                elif batch_target[i] == 0 and pred_class[i] == 0:
                    TN = TN + 1
                    N = N + 1

            #print("P %d"%P)
            print('TP/P = %g, TN/N=%g '%(TP/P, TN/N))


            ######## update the loss curve #########
            loss_curve[loss_curve_counter][0] = avg_loss
            loss_curve[loss_curve_counter][1] = validation_loss
            loss_curve[loss_curve_counter][2] = TP/P
            loss_curve[loss_curve_counter][3] = TN/N

            loss_curve_counter = loss_curve_counter + 1
            logfile=open(args.model_name+'_loss_curve.csv', 'wb')
            np.savetxt(logfile,loss_curve, delimiter=',')
            logfile.close()
            ############################


def test_multiFusion(model_name, test_set, threshold_score = 0.5, total_batch=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model
    model_multiFusion = torch.load(model_name)
    model_multiFusion.to(device)

    # batch_size = len(dataset)//total_batch # no need as the test set is already small

    # get the multi branches of data
    test_adc_ftr = test_set[0]
    test_hbv_ftr = test_set[1]
    test_t2w_ftr = test_set[2]
    test_prediction = test_set[3]
    # .to(device) to transfer to GPU
    batch_adc_ftr = test_adc_ftr.to(device)
    batch_hbv_ftr = test_hbv_ftr.to(device)
    batch_t2w_ftr = test_t2w_ftr.to(device)
    #batch_target = test_prediction.to(device)
    ##### run the prediction ##########
    model_multiFusion.eval()
    batch_prediction = model_multiFusion(batch_adc_ftr, batch_hbv_ftr, batch_t2w_ftr)
    batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
    test_prediction = list(test_prediction.flatten().cpu().detach().numpy())
    ### just for debug ###
    #print(batch_prediction[0:10])
    #print(batch_target[0:10])
    #######################
    for i in range(0, len(batch_prediction)):
        if batch_prediction[i]>= threshold_score:
            batch_prediction[i] = 1
        else:
            batch_prediction[i] = 0

    pred_class = batch_prediction
    TP = TN = FN = FP = 0
    P = N = 0
    for i in range (0, len(test_prediction)):
        #####################################
        if test_prediction[i] == 1:
            P = P + 1 
            if pred_class[i] == 1:
                TP = TP + 1
            else:
                FN = FN + 1
        ####################################
        elif test_prediction[i] == 0:
            N = N + 1
            if pred_class[i] == 1:
                FP = FP + 1
            else:
                TN = TN + 1
    
    print('TP/P = %g, TN/N=%g '%(TP/P, TN/N))
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0, 0] = TP
    confusion_matrix[0, 1] = FN
    confusion_matrix[1, 0] = FP
    confusion_matrix[1, 1] = TN
     
    return confusion_matrix



# https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/