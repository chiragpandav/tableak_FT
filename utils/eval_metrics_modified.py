import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import numpy as np


def get_acc_and_bac(network, X, y):
    """
    Returns the accuracy and the balanced accuracy score of a given neural network on the dataset X, y.

    :param network: (nn.Module) The torch model of which we wish to measure the accuracy of.
    :param X: (torch.tensor) The input features.
    :param y: (torch.tensor) The true labels corresponding to the input features.
    :return: (tuple) The accuracy score and the balanced accuracy score.
    """
    with torch.no_grad():
        _, all_pred = torch.max(network(X).data, 1)
        acc = accuracy_score(np.array(y.cpu()), np.array(all_pred.cpu()))
        bac = balanced_accuracy_score(np.array(y.cpu()), np.array(all_pred.cpu()))
    return acc, bac


def feature_wise_accuracy_score(true_data, reconstructed_data, tolerance_map, train_features):
    """
    Calculates the categorical accuracy and in-tolerance-interval accuracy for continuous features per feature.

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
    :param reconstructed_data: (np.ndarray) The reconstructed mixed-type feature vector.
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param train_features: (dict) A dictionary of the feature names per column.
    :return: (dict) A dictionary with the features and their corresponding error.
    """
    feature_errors = {}
    for feature_name, true_feature, reconstructed_feature, tol in zip(train_features.keys(), true_data, reconstructed_data, tolerance_map):
        if tol == 'cat':
            feature_errors[feature_name] = 0 if str(true_feature) == str(reconstructed_feature) else 1
        else:
            feature_errors[feature_name] = 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
    return feature_errors


def batch_feature_wise_accuracy_score(true_data, reconstructed_data, tolerance_map, train_features):
    """

    :param true_data: (np.ndarray) The true/reference mixed-type feature matrix.
    :param reconstructed_data:
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param train_features: (dict) A dictionary of the feature names per column.
    :return: (dict) A dictionary with the features and their corresponding error.
    """

    assert len(true_data.shape) == 2, 'This function requires a batch of data'

    batch_size = true_data.shape[0]
    feature_errors = {feature_name: 0 for feature_name in train_features.keys()}
    for true_data_line, reconstructed_data_line in zip(true_data, reconstructed_data):
        line_feature_errors = feature_wise_accuracy_score(true_data_line, reconstructed_data_line, tolerance_map, train_features)
        for feature_name in feature_errors.keys():
            feature_errors[feature_name] += 1/batch_size * line_feature_errors[feature_name]
    return feature_errors


# def _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
#     """

#     :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
#     :param reconstructed_data:
#     :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
#         corresponding to a numerical feature in the data should contain a floating point value marking the
#         reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
#         has to contain the entry 'cat'.
#     :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
#         features and by continuous features separately.
#     :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
#         If the flag 'detailed' is set to True the reconstruction errors of the categorical and the continuous features
#         are returned separately.
#     """
#     cat_score = 0
#     cont_score = 0
#     num_cats = 0
#     num_conts = 0

#     for true_feature, reconstructed_feature, tol in zip(true_data, reconstructed_data, tolerance_map):
#         if tol == 'cat':
#             cat_score += 0 if str(true_feature) == str(reconstructed_feature) else 1
#             num_cats += 1
#         elif not isinstance(tol, str):
#             cont_score += 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
#             num_conts += 1
#         else:
#             raise TypeError('The tolerance map has to either contain numerical values to define tolerance intervals or '
#                             'the string >cat< to mark the position of a categorical feature.')
#     if detailed:
#         if num_cats < 1:
#             num_cats = 1
#         if num_conts < 1:
#             num_conts = 1
#         return (cat_score + cont_score)/(num_cats + num_conts), cat_score/num_cats, cont_score/num_conts
#     else:
#         return (cat_score + cont_score)/(num_cats + num_conts)


# def _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
#     """

#     :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
#     :param reconstructed_data:
#     :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
#         corresponding to a numerical feature in the data should contain a floating point value marking the
#         reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
#         has to contain the entry 'cat'.
#     :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
#         features and by continuous features separately.
#     :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
#         If the flag 'detailed' is set to True the reconstruction errors of the categorical and the continuous features
#         are returned separately.
#     """
#     cat_score = 0
#     cont_score = 0
#     num_cats = 0
#     num_conts = 0

#     # print("true_data:::",true_data)
#     # print("called")
#     gender_collector=[]
#     race_collector=[]
#     male_collector=[]
#     female_collector=[]

#     male_rec_counter=0
#     female_rec_counter=0
#     # print("true_data", true_data)
#     # for true_feature, reconstructed_feature, tol in zip(true_data, reconstructed_data, tolerance_map):
#     for index, (true_feature, reconstructed_feature, tol) in enumerate(zip(true_data, reconstructed_data, tolerance_map)):

#         # print(tol)
#         if tol == 'cat':            
#             cat_score += 0 if str(true_feature) == str(reconstructed_feature) else 1
#             num_cats += 1
#         elif not isinstance(tol, str):
#             print(str(true_feature))

#             # for my information --- Reconstr
#             # need to check this logic
#             if  str(true_feature) == str(reconstructed_feature):
                
#                 # print("elif111", str(true_feature) ,":::",str(reconstructed_feature),":::", type(str(reconstructed_feature)))

#                 # if  str(reconstructed_feature) == 1 or str(reconstructed_feature) == 2:

#                 # print("elif222", str(true_feature) ,":::",str(reconstructed_feature))

#                 if index ==8:
#                     # print("Index ", index,str(true_feature), " ::: ",str(reconstructed_feature))
                    
#                     # print("gender_collector",gender_collector)
#                     if str(reconstructed_feature) == 1:
#                         male_rec_counter +=1

#                     if str(reconstructed_feature) == 2:
#                         female_rec_counter +=1

#                     gender_collector.append(reconstructed_feature)
#                     male_collector.append(male_rec_counter)

#                 if index==9:
#                     # print("Index ", index,str(true_feature), " ::: ",str(reconstructed_feature))
#                     race_collector.append(reconstructed_feature)
#                     # print("race_collector",race_collector)

#             cont_score += 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
#             num_conts += 1

#         else:
#             raise TypeError('The tolerance map has to either contain numerical values to define tolerance intervals or '
#                             'the string >cat< to mark the position of a categorical feature.')
        


#     if detailed:
#         # print("IFFFF")
#         if num_cats < 1:
#             num_cats = 1
#         if num_conts < 1:
#             num_conts = 1
#         return (cat_score + cont_score)/(num_cats + num_conts), cat_score/num_cats, cont_score/num_conts, gender_collector, race_collector
#     else:
#         # print("ELSEEE")
#         return (cat_score + cont_score)/(num_cats + num_conts), gender_collector, race_collector
    

def _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
    """

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
    :param reconstructed_data:
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
        features and by continuous features separately.
    :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
        If the flag 'detailed' is set to True the reconstruction errors of the categorical and the continuous features
        are returned separately.
    """
    cat_score = 0
    cont_score = 0
    num_cats = 0
    num_conts = 0

    # print("true_data:::",true_data)
    # print("called")
    gender_collector=[]
    race_collector=[]
    male_collector=[]
    female_collector=[]
    white_collector=[]
    black_collector=[]

    male_rec_counter=1
    female_rec_counter=2

    white_rec_counter=1
    black_rec_counter=2

    # print("true_data", true_data) # [1,2,3,4,5,5,6,7,1,9,0]
    # for true_feature, reconstructed_feature, tol in zip(true_data, reconstructed_data, tolerance_map):
    for index, (true_feature, reconstructed_feature, tol) in enumerate(zip(true_data, reconstructed_data, tolerance_map)):

        # print(tol) 

        if tol == 'cat':            
            cat_score += 0 if str(true_feature) == str(reconstructed_feature) else 1
            num_cats += 1
        elif not isinstance(tol, str):
            # print(str(true_feature))

            # for my information --- Reconstr
            # Need to check this logic
            
            
            # print("elif111", str(true_feature) ,":::",str(reconstructed_feature),":::", type(str(reconstructed_feature)))

            # if str(reconstructed_feature) == 1 or str(reconstructed_feature) == 2:

            # print("elif222", str(true_feature) ,":::",str(reconstructed_feature))

            if  str(true_feature) == str(reconstructed_feature) :                   

                if float(str(reconstructed_feature)) != 0:

                    # print(str(reconstructed_feature))  
                    # print("---------------------------")                
                    # print("true_data",true_data)   
                    # print("---------------------------")                
                    if true_data[8] == 1:                        
                        male_collector.append(male_rec_counter)
                    if true_data[8] == 2:                        
                        female_collector.append(female_rec_counter)
                    if true_data[9] == 1:                        
                        white_collector.append(white_rec_counter)
                    if true_data[9] == 2:                        
                        black_collector.append(black_rec_counter)

                    # Need to confirm

                    # if index==8 and float(str(true_feature)) == 1:                          
                    #     male_collector.append(male_rec_counter)

                    # if index==8 and float(str(true_feature)) == 2:
                    #     female_collector.append(female_rec_counter)

                    # if index==9 and float(str(true_feature)) == 1:                          
                    #         white_collector.append(white_rec_counter)

                    # if index==9 and float(str(true_feature)) == 2:
                    #         black_collector.append(black_rec_counter)
                        
                        # if str(reconstructed_feature) == str(1):
                        #     # print("str(reconstructed_feature)",str(reconstructed_feature))
                        #     male_collector.append(male_rec_counter)

                        # if str(reconstructed_feature) == str(2):
                        #     female_collector.append(female_rec_counter)

                    gender_collector.append(reconstructed_feature)


                if index==9:
                    # print("Index ", index,str(true_feature), " ::: ",str(reconstructed_feature))
                    race_collector.append(reconstructed_feature)
                    # print("race_collector",race_collector)

                # if index ==3:
                #     print("you are here::",str(true_feature) , "::",str(reconstructed_feature))

            cont_score += 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
            num_conts += 1

        else:
            raise TypeError('The tolerance map has to either contain numerical values to define tolerance intervals or '
                            'the string >cat< to mark the position of a categorical feature.')
        


    if detailed:
        # print("IFFFF")
        if num_cats < 1:
            num_cats = 1
        if num_conts < 1:
            num_conts = 1
        return (cat_score + cont_score)/(num_cats + num_conts), cat_score/num_cats, cont_score/num_conts, gender_collector, race_collector,male_collector,female_collector,white_collector,black_collector
    else:
        # print("ELSEEE")
        return (cat_score + cont_score)/(num_cats + num_conts), gender_collector, race_collector,male_collector,female_collector,white_collector,black_collector
   


def categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, detailed=False):
    """
    Calculates an error score between the true mixed-type datapoint and a reconstructed mixed-type datapoint. For each
    categorical feature we count a 0-1 error by the rule of the category being reconstructed correctly. For each
    continuous feature we count a 0-1 error by the rule of the continuous variable being reconstructed within a
    symmetric tolerance interval around the true value. The tolerance parameters are set by 'tolerance_map'.

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector or matrix if comprising more than
        datapoint.
    :param reconstructed_data: (np.ndarray) The reconstructed mixed-type feature vector/matrix.
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param detailed: (bool) Set to True if you want additionally to calculate the error rate induced by categorical
        features and by continuous features separately.
    :return: (float or tuple of floats) The accuracy score with respect to the given tolerance of the reconstruction.
        If a batch of data is given, then the average accuracy of the batch is returned. Additionally, if the flag
        'detailed' is set to True the reconstruction errors of the categorical and the continuous features are returned
        separately.
    """
    assert true_data.shape == reconstructed_data.shape
    score = 0
    cat_score = 0
    cont_score = 0
    gen_1=[]
    rac_1=[]
    man_1=[]
    female_1=[]
    white_1=[]
    black_1=[]

    if len(true_data.shape) > 1:
        # print("IFFF")
        for true_data_line, reconstructed_data_line in zip(true_data, reconstructed_data):
            assert len(true_data_line) == len(tolerance_map)
            if detailed:
                scores = _categorical_accuracy_continuous_tolerance_score(true_data_line, reconstructed_data_line,
                                                                          tolerance_map, True)
                score += 1/true_data.shape[0] * scores[0]
                cat_score += 1 / true_data.shape[0] * scores[1]
                cont_score += 1 / true_data.shape[0] * scores[2]
            else:
                score += 1/true_data.shape[0] * _categorical_accuracy_continuous_tolerance_score(true_data_line,
                                                                                                 reconstructed_data_line,
                                                                                                 tolerance_map)
                
                # print("true_data_line::",true_data_line)
                # print("reconstructed_data_line::",reconstructed_data_line)
    else:
        # print("ELSE")
        # this is being called.......
        assert len(true_data) == len(tolerance_map)
        if detailed:
            # print("detailed")
            # Modified----
            scores= _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map, True)            
            score += scores[0]
            cat_score += scores[1]
            cont_score += scores[2]
            gen_1.append(scores[3])
            rac_1.append(scores[4])
            man_1.append(scores[5])
            female_1.append(scores[6])
            white_1.append(scores[7])
            black_1.append(scores[8])

        else:
            # print("not detailed")
            score = _categorical_accuracy_continuous_tolerance_score(true_data, reconstructed_data, tolerance_map)

    if detailed:
        # print("detailed")
        # gend=scores[3]
        # rac=scores[4]

        # print(gen_1,"::::",rac_1)
        return score, cat_score, cont_score,gen_1[0],rac_1[0],man_1[0],female_1[0],white_1[0],black_1[0]
    else:
        return score
