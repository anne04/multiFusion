import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from multiFusion_train_util import shuffle_data #, data_to_tensor
from lifelines.utils import concordance_index
import wandb

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


        input_size_fusion = int(flatten_size_adc + flatten_size_hbv + flatten_size_t2w) 
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
            nn.Linear(hidden_size_predictor_layer2, 1)
            # no sigmoid or other activation since it is a regression problem
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


        # try with and without normalizing (usually normalizing is preferred before infusion)
        adc_emb = F.normalize(adc_emb, p=2) 
        hbv_emb = F.normalize(hbv_emb, p=2) 
        t2w_emb = F.normalize(t2w_emb, p=2) 
        #clinical_emb = F.normalize(adc_emb, p=2) 

        concat_emb = torch.cat((adc_emb, hbv_emb, t2w_emb), dim=1)

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
    print_model_flag = 0,
    wandb_project_name = '',
    fold = 0
    ):
    """
    args:
    training_set: torch.Tensor of training samples (80%)
    validation_set: torch.Tensor of validation samples (20%)
    val_class: list() of validation samples but with binary label (0/1)
    threshold_score: some cutoff to set binary labels
    """

    wandb.init(project=wandb_project_name, mode="offline", name="fold-"+str(fold))
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
      
    if print_model_flag == 1:
        print(model_multiFusion)
    # set the loss function
    loss_function = nn.MSELoss()

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
        training_adc_ftr, training_hbv_ftr, training_t2w_ftr, training_target = shuffle_data(training_set)        
        model_multiFusion.train() # training mode
        total_loss = 0
        for batch_idx in range(0, total_batch):
            optimizer.zero_grad() # clears the grad, otherwise will add to the past calculations
            # get the batch of the training features and move to GPU
            batch_adc_ftr = training_adc_ftr[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :].to(device)
            batch_hbv_ftr = training_hbv_ftr[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :].to(device)
            batch_t2w_ftr = training_t2w_ftr[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :].to(device)
            batch_target = training_target[batch_idx*batch_size: (batch_idx+1)*batch_size, :].to(device)
            # run the model and get prediction
            batch_prediction = model_multiFusion(batch_adc_ftr, batch_hbv_ftr, batch_t2w_ftr)

            ### for debug purpose to see if the weights are changing ############
            '''
            if batch_idx == 0 and epoch_indx%epoch_interval == 0:
                print('debug: training:')
                print(batch_target[0:10])
                print(list(batch_prediction.flatten().cpu().detach().numpy())[0:10])            
            
            '''
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
            validation_target = validation_set[3]
            # .to(device) to transfer to GPU
            batch_adc_ftr = validation_adc_ftr.to(device)
            batch_hbv_ftr = validation_hbv_ftr.to(device)
            batch_t2w_ftr = validation_t2w_ftr.to(device)
            batch_target = validation_target.to(device)
            ##### run the prediction ##########
            model_multiFusion.eval()
            batch_prediction = model_multiFusion(batch_adc_ftr, batch_hbv_ftr, batch_t2w_ftr)
            validation_loss = loss_function(batch_prediction.flatten(), batch_target.flatten())
            batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
            batch_target = list(batch_target.flatten().cpu().detach().numpy())
            validation_Cindex = concordance_index(batch_target, batch_prediction)

            if epoch_indx==0:
                min_loss = validation_loss
                max_cindex = validation_Cindex

            print('Epoch %d/%d, Training loss: %g, val loss: %g, c-index: %g'%(epoch_indx, epoch, avg_loss, validation_loss, validation_Cindex))
            wandb.log({
                "epoch": epoch_indx,
                "train_loss": avg_loss,
                "val_loss": validation_loss,
                "val_c_index": validation_Cindex
            })
            
            if validation_loss <= min_loss:
                min_loss = validation_loss
                # state save
                torch.save(model_multiFusion, args.model_name)  
                print('*** min loss found! %g ***'%validation_loss) 

            '''
            if validation_Cindex > max_cindex:
                max_cindex = validation_Cindex
                # state save
                torch.save(model_multiFusion, args.model_name)  
                print('*** max c-index found! %g ***'%validation_Cindex)              
            
            '''


            #########
            ### just for debug ###
            #print(batch_prediction[0:10])
            #print(batch_target[0:10])
            #######################

            ######## update the loss curve #########
            loss_curve[loss_curve_counter][0] = avg_loss
            loss_curve[loss_curve_counter][1] = validation_loss
            loss_curve[loss_curve_counter][2] = validation_Cindex


    loss_curve_counter = loss_curve_counter + 1
    logfile=open(wandb_project_name+'_fold'+ str(fold) +'_loss_curve.csv', 'wb')
    np.savetxt(logfile,loss_curve, delimiter=',')
    logfile.close()
    wandb.finish()
            ############################


def test_multiFusion(model_name, test_set, fold,  threshold_score = 0.5, total_batch=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model
    model_multiFusion = torch.load(model_name)
    model_multiFusion.to(device)

    # batch_size = len(dataset)//total_batch # no need as the test set is already small

    # get the multi branches of data
    test_adc_ftr = test_set[0]
    test_hbv_ftr = test_set[1]
    test_t2w_ftr = test_set[2]
    test_target = test_set[3]
    # .to(device) to transfer to GPU
    batch_adc_ftr = test_adc_ftr.to(device)
    batch_hbv_ftr = test_hbv_ftr.to(device)
    batch_t2w_ftr = test_t2w_ftr.to(device)
    #batch_target = test_prediction.to(device)
    ##### run the prediction ##########
    model_multiFusion.eval()
    batch_prediction = model_multiFusion(batch_adc_ftr, batch_hbv_ftr, batch_t2w_ftr)
    batch_prediction = list(batch_prediction.flatten().cpu().detach().numpy())
    test_target= list(test_target.flatten().cpu().detach().numpy())
    test_Cindex = concordance_index(test_target, batch_prediction) 

    print('fold '+ str(fold) +' C-index is %g'%test_Cindex)
    '''
    wandb.log({
        "fold": fold,
        "test_c_index": test_Cindex
    })    

    '''

    ### just for debug ###
    #print(batch_prediction[0:10])
    #print(batch_target[0:10])
    #######################
     
    return test_Cindex



# https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/
