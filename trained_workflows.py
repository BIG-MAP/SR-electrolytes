import pandas as pd
import sympy as sp
import numpy as np



class TrainedWorkflow:
    """
    Generates trained symbolic models.

    Parameters
    ----------
    coeff_table: pd.DataFrame
        Pandas dataframe with coefficients resulting from training
    initial_features: list[str]
        list of initial predictors (e.g. [T, c, r])
    intercept: float
        value of the trained intercept parameter. intercept = 0 for contrained models.

    Returns
    --------
    trained_model object.

    """

    def __init__(self, coeff_table: pd.DataFrame, 
                initial_features: list,
                intercept: float):
        #Data
        self.coeff_table = coeff_table
        self.intercept = intercept
        self.initial_features = initial_features
        #Results
        self.eqn = None

        self.__generate_symbolic_eqn()



    def __generate_symbolic_eqn(self):

        if self.coeff_table.empty:
            self.eqn = sp.Integer(0)

        else:

            eqn_string = str(self.intercept)
            symbols = {i:sp.Symbol(i) for i in self.initial_features}

            for index, row in self.coeff_table.iterrows():
                eqn_string += ' + {}*{}'.format(row['coeff_corr'],index)

            self.eqn = sp.sympify(eqn_string,symbols)


    @property
    def nfeatures(self):
        return self.coeff_table.shape[0]
        

    def predict(self, x: pd.DataFrame):
        
        if list(x.columns) == self.initial_features: 

            if self.coeff_table.empty:
                return np.zeros(x.shape[0])

            else:            
                #Lambda functions cannot be pickled. Solution: Keep lmabdification inside of predict. 
                #In this way the lambda function is only generated when calling predict. Predict can be pickled.
                function_from_eqn =  sp.lambdify(args=[sp.Symbol(i) for i in self.initial_features],
                                        expr=self.eqn)   

                y_hat = function_from_eqn(*[feature_col.to_numpy() for _,feature_col in x.iteritems()]) 

                if isinstance(y_hat, float): #happens when equation does not have variables, only intercept
                    return y_hat*np.ones(x.shape[0]) #ensures an array is returned
                elif len(y_hat) == x.shape[0]:
                    return y_hat
                else:
                    raise Exception('The lenght of the predicted array is not compatible with the lenght of the predictors dataset')

        else:
            raise Exception("""The dataframe input does not have the same columns 
                                (different names or different order) as the dataframe 
                                used for training, i.e: [{}]""".format(', '.join(self.initial_features)))



