import numpy as np
import pandas as pd
import sympy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso
from autofeat import AutoFeatRegressor

import trained_workflows as tf






class Workflow:
    

    """
    Base class for Workflows

    A workflow is instantiated from:
    * xtrain: pandas dataframe of shape (samples, features). Predictors. The column names are used as terms in the symbolic expression
    * ytrain: pandas dataframe of shape (samples)
    * scaling_type

    Generate first, standardize later.
    This enables to recover statistical summaries from generated x
    If standardization is done before generation, the summaries of the original generated data cannot be recovered.
    Also, generation from original data presevres the domain of features (e.g. non-negative) and enables autofeat to explore the right operators.
    """

    def __init__(self,  xtrain: pd.DataFrame, 
                        ytrain: pd.DataFrame,
                        scaling_type: str,
                        stdalpha: float, 
                        rejection_thresshold: int, 
                        fit_intercept = True):

        #Parameters
        self.scaling_type = scaling_type
        self.stdalpha = stdalpha
        self.rejection_thresshold = rejection_thresshold
        self.fit_intercept = fit_intercept
        #Data
        self.xtrain = xtrain
        self.ytrain = ytrain
        #Others
        self.dataset_checking()
        self.initialize_defaults()


    def initialize_defaults(self):
        #Derived data
        self.xtrain_gen = None
        self.xtrain_gen_stand = None
        #Models
        self.optimal_alpha = 0.0
        self.lasso_regressor = Lasso(alpha=self.optimal_alpha, max_iter=100000,
                                     fit_intercept= self.fit_intercept)
        self.lassocv_regressor =  LassoCV(alphas=np.logspace(-6, 6, 13), cv=10, max_iter=100000, 
                                        fit_intercept= self.fit_intercept)
        #Results
        self.coefficients = None
        self.intercept = None
        self.intercept_corr = None
        self.ytrain_hat = None
        self.coeff_table = None

    def dataset_checking(self):
        if self.xtrain.empty:
            raise Exception('Dataset of initial predictors cannot be empty')
        if self.ytrain.empty:
            raise Exception('Dataset of targets cannot be empty')
        if self.xtrain.shape[0] != self.ytrain.shape[0]:
            raise Exception('Dataset of initial predictors must have same number of samples as targets')



    def standardize(self):

        if self.scaling_type == 'standard':    
            scaler_obj = StandardScaler() 
        elif self.scaling_type == 'standard_nomean':
            scaler_obj = StandardScaler(with_mean=False)
        elif self.scaling_type == 'none':
            scaler_obj = None
        else:
            raise Exception("Scaler not implemented")


        if not scaler_obj:
            self.xtrain_gen_stand =  self.xtrain_gen

        else:
            df_columns = list(self.xtrain_gen.columns)        
            scaler_obj.fit(self.xtrain_gen.to_numpy())
            
            self.xtrain_gen_stand = pd.DataFrame(scaler_obj.transform(self.xtrain_gen.to_numpy()), columns = df_columns)



    def sparsify_lasso(self):
        self.lassocv_regressor.fit(self.xtrain_gen_stand, self.ytrain)


    
    def find_optimal_alpha(self):

        mean_mses = np.mean(self.lassocv_regressor.mse_path_, axis=1) #mean mse across the cv path for every alpha 
        std_mses = np.std(self.lassocv_regressor.mse_path_, axis=1)   #std mse across the cv path for every alpha  
        idx_min_mse = np.argmin(mean_mses)    #index of lowest mse

        #indexes of mean_mses values that are within std away from the minimum mse
        idxs_mean_mses_within_std = np.where((mean_mses < mean_mses[idx_min_mse]+self.stdalpha*std_mses[idx_min_mse]) & (mean_mses > mean_mses[idx_min_mse]-self.stdalpha*std_mses[idx_min_mse]))
        
        self.optimal_alpha = np.amax(self.lassocv_regressor.alphas_[idxs_mean_mses_within_std])



    def regress_lasso(self):
        self.lasso_regressor.set_params(alpha = self.optimal_alpha)
        self.lasso_regressor.fit(self.xtrain_gen_stand, self.ytrain)
        self.ytrain_hat = self.lasso_regressor.predict(self.xtrain_gen_stand)
        self.coefficients = self.lasso_regressor.coef_
        self.intercept = self.lasso_regressor.intercept_

    
    def generate_coeff_table(self):

        coeffs = {'mean':[],
                'stdev': [],
                'coeff':[],
                'coeff stdev':[],
                'coeff |t|':[]}
        
        try:
            squared_error_y_hat = np.sum((self.ytrain-self.ytrain_hat)**2)


            for coef, col in zip(self.coefficients, self.xtrain_gen_stand):   
                squared_dev_x_mean = np.sum((self.xtrain_gen_stand[col]-self.xtrain_gen_stand[col].mean())**2)
                n_minus_2 = len(self.xtrain_gen_stand[col])-2        
                coef_stdev = np.sqrt(squared_error_y_hat/(n_minus_2*squared_dev_x_mean))

                coeffs['mean'].append(self.xtrain_gen[col].mean())
                coeffs['stdev'].append(self.xtrain_gen[col].std())
                coeffs['coeff'].append(coef)
                coeffs['coeff stdev'].append(coef_stdev)
                coeffs['coeff |t|'].append(np.abs(coef/coef_stdev))

            
            self.coeff_table = pd.DataFrame(coeffs, index=self.xtrain_gen_stand.columns)
            self.coeff_table.sort_values(by=['coeff |t|'], ascending=False, inplace=True) 
          
            
        except MemoryError:
            print("Numpy cannot allocate enough memory to calculate y_hat")
            self.coeff_table = pd.DataFrame()


    
    def discard_features(self):

        neglected_features = list(self.coeff_table.loc[self.coeff_table['coeff |t|']<=self.rejection_thresshold,:].index)

        if len(neglected_features) ==0:
            return False
        else:             
            self.xtrain_gen_stand.drop(columns=neglected_features, inplace=True)
            return True




    def correct_coeffs(self):
        '''
        Transforms standardized regression coefficients into unstandardized versions.

        When standardization is 'standard':
        #b = b_stand/stdev(x)    and    y0 = y0_stand - sum(b*<x>, all x)

        When standardization is 'standard, no mean':
        #b = b_stand/stdev(x)    and    y0 = y0_stand

        When standardization is 'none':
        #b = b_stand    and    y0 = y0_stand

        See for reference: 
        https://www.real-statistics.com/multiple-regression/standardized-regression-coefficients/
        
        '''

        if self.scaling_type == 'standard':    
            self.coeff_table['coeff_corr'] = self.coeff_table['coeff']/self.coeff_table['stdev']
            self.intercept_corr = self.intercept - np.dot(self.coeff_table['coeff_corr'],self.coeff_table['mean'])


        elif self.scaling_type == 'standard_nomean':
            self.coeff_table['coeff_corr'] = self.coeff_table['coeff']/self.coeff_table['stdev']
            self.intercept_corr = self.intercept

        elif self.scaling_type == 'none':
            self.coeff_table['coeff_corr'] = self.coeff_table['coeff']
            self.intercept_corr = self.intercept

        if not self.fit_intercept:
            self.intercept = 0.0
            self.intercept_corr = 0.0


    def get_trained_workflow(self):

        if self.xtrain_gen_stand.empty:

            return tf.TrainedWorkflow(coeff_table = pd.DataFrame(),
                                    initial_features = list(self.xtrain.columns),
                                    intercept = 0)
        
        else:
            return tf.TrainedWorkflow(coeff_table = self.coeff_table,
                                    initial_features = list(self.xtrain.columns),
                                    intercept = self.intercept_corr)



    def run_workflow(self):

        self.generate_features()
        self.standardize()

        features_were_discarded = True
        loopn = 0
        while features_were_discarded:
                print('[Sparsification] Loop: {}. Current number of features: {}'.format(loopn, self.xtrain_gen_stand.shape[1]))
                self.sparsify_lasso()
                self.find_optimal_alpha()
                self.regress_lasso()
                self.generate_coeff_table()
                features_were_discarded = self.discard_features()
                loopn += 1                
                if self.xtrain_gen_stand.empty:
                    break #break loop if sparsification resulted in no predictors
                
        
        self.correct_coeffs()

        print('TRAINING COMPLETE')

        return self.get_trained_workflow()





class WorkflowAF(Workflow):

    def __init__(self, feateng_steps, units, featsel_runs, transformations, **kwargs):            
        self.feateng_steps = feateng_steps
        self.units= units
        self.featsel_runs = featsel_runs
        self.transformations = transformations

        super().__init__(**kwargs)
        self.instantiate_featgen_model()
        

    def instantiate_featgen_model(self):
        self.af_model = AutoFeatRegressor(verbose=1, 
                                        feateng_steps=self.feateng_steps, 
                                        units=self.units, 
                                        max_gb=7, 
                                        featsel_runs=self.featsel_runs,
                                        transformations=self.transformations)

    def generate_features(self):

        self.af_model.fit(self.xtrain, self.ytrain)
        self.xtrain_gen = self.af_model.transform(self.xtrain)






class WorkflowSelectedTerms(Workflow):
    def __init__(self, selected_terms: list[str], **kwargs):            
        super().__init__(**kwargs)
        self.selected_terms = selected_terms


    def initialize_defaults(self):
        #remove unused attributes 
        self.__dict__.pop('stdalpha',None) 
        self.__dict__.pop('rejection_thresshold',None) 
        #Derived data
        self.xtrain_gen = pd.DataFrame()
        self.xtrain_gen_stand = None
        #Model
        self.lassocv_regressor = LassoCV(alphas=np.logspace(-6, 6, 13), cv=10, 
                                        max_iter=100000, fit_intercept= self.fit_intercept)
        #Results
        self.coefficients = None
        self.intercept = None
        self.intercept_corr = None
        self.ytrain_hat = None
        self.coeff_table = None
        

    def generate_features(self):

        #tranform column names (str) into sympy terms
        symbols = {i:sp.Symbol(i) for i in self.xtrain.columns}
        
        for term in self.selected_terms:
            
            #if selected term is in original data
            if term in self.xtrain.columns:
               self.xtrain_gen[term] =  self.xtrain[term]

            #if selected term is not in original data
            else:                
                #transform term into sympy expression
                sympy_expression = sp.sympify(term,symbols)
                #transform expression into function
                sympy_function = sp.lambdify(args=list(symbols.values()),
                                            expr=sympy_expression)
                #apply function to original features to get transformed feature
                self.xtrain_gen[term] = sympy_function(*[feature_col.to_numpy() for _,feature_col in self.xtrain.iteritems()])



    def regress_lasso(self):
        # Fit selected features
        self.lassocv_regressor.fit(self.xtrain_gen_stand, self.ytrain)
        self.ytrain_hat = self.lassocv_regressor.predict(self.xtrain_gen_stand)
        self.coefficients = self.lassocv_regressor.coef_
        self.intercept = self.lassocv_regressor.intercept_


    def run_workflow(self):
        self.generate_features()
        self.standardize()
        self.regress_lasso()
        self.generate_coeff_table()
        self.correct_coeffs()

        return self.get_trained_workflow()










