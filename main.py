

import training_pipeline as tp
import evaluation_pipeline as ep
import log_experiment as lgx
from training_config import tconfig


# ------------ DATA PIPELINE -----------
# load raw json files into train, val, test datasets and make eda plots
#import data_pipeline as dp
# dp.run_data_pipeline()


# ------------ TRAINING PIPELINE -----------
#Transform some initial features: old label, new label, transformation
# initfeat_transform = [{'old':'T', 'new':'Tinv', 'func':lambda x: 1/x}]
# initfeat_transform = []


#********** Manual selection of some runs *********
# training_runs = [('Autofeat_2', 50, 100), ('Autofeat_2', 100, 99), ('Autofeat_4', 50, 1),
#                 ('Autofeat_4', 50, 21), ('Autofeat_4', 50, 30), ('Autofeat_4', 50, 69),
#                 ('Autofeat_4', 100, 2), ('Autofeat_4', 100, 21), ('Autofeat_4', 100, 31)]
# models_to_skip = list(tconfig['models'].keys())
# inside loop: tconfig['skip'] = [model_name for model_name in models_to_skip if model_name != trun[0]]

# ********* ORDINARY EN_MASS RUNS ***********
# data_sizes = [50, 100, 250, 400]
# random_seeds = [2, 35, 42, 69, 79]

# for rnds in random_seeds:
#     for dtsz in data_sizes:
        
#         tconfig['data']['subsample'] = dtsz
#         tconfig['data']['random_state'] = rnds


#create new results directory
current_results_path = lgx.create_experiment_directory(path = tconfig['data']['path_results'], 
                                                        dataset_size = tconfig['data']['subsample'], 
                                                        random_state = tconfig['data']['random_state'], 
                                                        append_label = 'selected_alldata_noy0')
#train models and save as pickle
try:
    trained_models = tp.train_models(tconfig)  

    lgx.pickle_model_results(path=current_results_path, 
                            data_dict=trained_models)

    # #save config file
    lgx.save_to_json(path=current_results_path,
                    data_dict=tconfig, 
                    label='training_config')

    # # ------------ EVALUATION PIPELINE -----------

    #evaluate results and save file
    evaluated_models = ep.run_evaluation_pipeline(trained_models, 
                                                    path = current_results_path, 
                                                    tconfig=tconfig,
                                                    initfeat_transform = [])
    #save evaluation file
    lgx.save_to_json(path=current_results_path, 
                    data_dict=evaluated_models, 
                    label='model_and_errors')

#Inf or Nans during training
except ValueError as ve:
    # #save error
    lgx.save_to_json(path=current_results_path,
                    data_dict={'Error in training': ve}, 
                    label='training_error')



  



