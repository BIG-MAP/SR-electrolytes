import pickle
import workflows as wf
import copy
import pandas as pd



def subsample_df(df, subsample: int,
                random_state: int):
        # Subsample training set
    if subsample == 0:
        return df
    elif (subsample > 0) & (subsample < df.shape[0]):
        return df.sample(subsample, random_state =random_state)
    else:
        raise Exception("""the subsample size must be either 0 (all samples) or a positive number 
                            smaller than the total number of training samples {}""".format(df.shape[0]))


def transform_df_column(df, transformations: list[dict]):

    new_df = df.copy(deep=True)

    for transform_dict in transformations:
        new_df[transform_dict['new']] = new_df[transform_dict['old']].apply(transform_dict['func'])
        new_df.drop(columns=[transform_dict['old']])
    
    return new_df





def get_training_dataset(load_from: str):

    #load training set
    with open(load_from,'rb') as pickle_file:
        dataset_train = pickle.load(pickle_file)

    #------------TEMP: ALL DATA------------------------------
    # with open('./data/dataset_val.pkl','rb') as pickle_file:
    #     dataset_val = pickle.load(pickle_file)

    # with open('./data/dataset_test.pkl','rb') as pickle_file:
    #     dataset_test = pickle.load(pickle_file)

    # dataset_train = pd.concat([dataset_train, dataset_val, dataset_test])
    # print(dataset_train.shape)
    #------------TEMP: ALL DATA------------------------------

    return dataset_train




def train_models(tconfig, initfeat_transform=[]):

    #retrieve training configuration
    RANDOM_SATE = copy.deepcopy(tconfig['data']['random_state'])
    PATH_DATASET_TRAIN = copy.deepcopy(tconfig['data']['path_data_train'])
    TRAINING_SUBSAMPLE = copy.deepcopy(tconfig['data']['subsample'])
    MODELS = copy.deepcopy(tconfig['models'])
    SKIP_MODELS = copy.deepcopy(tconfig['skip'])


    #instantiate empy container for results
    trained_models = {}
    
    
    # Load training data
    dataset_train = get_training_dataset(PATH_DATASET_TRAIN)


    #subsample
    dataset_train = subsample_df(dataset_train, 
                                subsample = TRAINING_SUBSAMPLE,
                                random_state = RANDOM_SATE)

    #tranform features
    if len(initfeat_transform)>0:
        dataset_train = transform_df_column(dataset_train, transformations = initfeat_transform)


    # delete models skipped
    for model_label in SKIP_MODELS:
        if model_label in MODELS:
            MODELS.pop(model_label)

    
    # instantiate workflow for every non-skipped model defined inside model_attrs
    for model_label, model_specs in MODELS.items():


            if model_specs['workflow'] == 'NoGen':
                workflow = wf.WorkflowNoGen(xtrain = dataset_train[model_specs['features']], 
                                            ytrain = dataset_train['σ_mean'], 
                                            scaling_type = model_specs['scaling'],
                                            stdalpha =  model_specs['stdalpha'],
                                            rejection_thresshold = model_specs['thresshold'], 
                                            fit_intercept = model_specs['fit_intercept'])
            
            elif model_specs['workflow'] == 'Poly':
                workflow = wf.WorkflowPoly(poly_order =model_specs['poly_order'],
                                            xtrain = dataset_train[model_specs['features']], 
                                            ytrain = dataset_train['σ_mean'], 
                                            scaling_type = model_specs['scaling'],
                                            stdalpha =  model_specs['stdalpha'], 
                                            rejection_thresshold = model_specs['thresshold'], 
                                            fit_intercept = model_specs['fit_intercept']) 

            elif model_specs['workflow'] == 'NoGenArrh':
                workflow = wf.WorkflowNoGenArrh(xtrain = dataset_train[model_specs['features']], 
                                            ytrain = dataset_train['σ_mean'], 
                                            scaling_type = model_specs['scaling'],
                                            stdalpha =  model_specs['stdalpha'], 
                                            rejection_thresshold = model_specs['thresshold'], 
                                            fit_intercept = model_specs['fit_intercept']) 

            elif model_specs['workflow'] == 'PolyArrh':
                workflow = wf.WorkflowPolyArrh(poly_order =model_specs['poly_order'],
                                            xtrain = dataset_train[model_specs['features']], 
                                            ytrain = dataset_train['σ_mean'], 
                                            scaling_type = model_specs['scaling'],
                                            stdalpha =  model_specs['stdalpha'], 
                                            rejection_thresshold = model_specs['thresshold'], 
                                            fit_intercept = model_specs['fit_intercept']) 

            elif model_specs['workflow'] == 'AF':
                workflow = wf.WorkflowAF(feateng_steps = model_specs['feateng_steps'],
                                        units =  model_specs['units'],
                                        featsel_runs = model_specs['featsel_runs'],
                                        transformations = model_specs['transformations'],
                                        xtrain = dataset_train[model_specs['features']], 
                                        ytrain = dataset_train['σ_mean'], 
                                        scaling_type = model_specs['scaling'],
                                        stdalpha =  model_specs['stdalpha'], 
                                        rejection_thresshold = model_specs['thresshold'], 
                                        fit_intercept = model_specs['fit_intercept']) 


            elif model_specs['workflow'] == 'SelectedTerms':
                workflow = wf.WorkflowSelectedTerms(xtrain = dataset_train[model_specs['features']], 
                                                    ytrain = dataset_train['σ_mean'], 
                                                    scaling_type = model_specs['scaling'],
                                                    stdalpha =  model_specs['stdalpha'], 
                                                    rejection_thresshold = model_specs['thresshold'], 
                                                    selected_terms=model_specs['selected terms'], 
                                                    fit_intercept = model_specs['fit_intercept']) 

            else:
                raise Exception('Workflow not supported')
            
            #run workflow and return trained workflow
            print('----------- WORKFLOW START MODEL: {} -----------'.format(model_label))
            trained_workflow = workflow.run_workflow()
            trained_models.update({model_label: trained_workflow})


    return trained_models




if __name__ == '__main__': 

    trained_models = train_models()

