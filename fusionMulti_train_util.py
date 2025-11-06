import torch
import numpy as np

cellNEST_dimension = 512
geneEmb_dimension = 256 
proteinEmb_dimension = 1024


def split_branch(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction_column = data.size(1)-1
    rcvr_dimension_total = sender_dimension_total = cellNEST_dimension + geneEmb_dimension + proteinEmb_dimension
    sender_emb = data[:, cellNEST_dimension:sender_dimension_total]    
    rcv_emb = data[:, sender_dimension_total+cellNEST_dimension:sender_dimension_total+rcvr_dimension_total]
    prediction = data[:, prediction_column]
    return sender_emb, rcv_emb, prediction 

def shuffle_data(
    training_set: torch.Tensor 
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shuffles the training data
    """
    # Generate random permutation of row indices
    sample_count = training_set.size(0)
    #prediction_column = training_set.size(1)-1
    row_perm = torch.randperm(sample_count)

    # Shuffle the rows using advanced indexing
    training_set = training_set[row_perm]
    #print(training_set)
    training_sender_emb, training_rcv_emb, training_prediction = split_branch(training_set)
    return training_sender_emb, training_rcv_emb, training_prediction 
    
def data_to_tensor(
    training_set, #: list()
    remove_set=None
    ):
    """
    training_set = list of [sender_emb, rcvr_emb, pred]
    """
    add_set = []
    rcvr_dimension_total = sender_dimension_total = 512 + 256 + 1024
    training_set_matrix = np.zeros((len(training_set), sender_dimension_total + rcvr_dimension_total + 1 )) # 1=prediction column
    for i in range(0, len(training_set)):
        if remove_set != None:
            if training_set[i][3]+'_to_'+training_set[i][4] in remove_set:
                add_set.append(training_set[i])
                continue

        training_set_matrix[i, 0:sender_dimension_total] = np.concatenate((training_set[i][0][0].flatten(),training_set[i][0][1], training_set[i][0][2]), axis=0)

        training_set_matrix[i, sender_dimension_total:sender_dimension_total+rcvr_dimension_total] = np.concatenate((training_set[i][1][0].flatten(),training_set[i][1][1], training_set[i][1][2]), axis=0)

        training_set_matrix[i, sender_dimension_total+rcvr_dimension_total] = training_set[i][2]

    # convert to tensor
    training_set = torch.tensor(training_set_matrix, dtype=torch.float)

    return training_set, add_set

