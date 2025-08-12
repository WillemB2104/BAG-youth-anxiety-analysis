# This works - but is v slow for TreeBased methods (i.e. 10min). Its 3x times as fast using the default (BR)

# Can report this for methods:
# The baseline data included 2.3% overall missing predictor values, with specific variable missingness ranging 
# from 2 to 12.5% (see Table 1 for percentage of missing data). For missing data estimation, we used 
# IterativeImputer (scikit-learn package (Pedregosa et al., 2011)). Specifically, BayesianRidge, the default 
# estimator used for IterativeImputer, imputes missing values for any predictor as a function of all other '
# predictors by using regularized linear regression. Imputation begins with the predictor with the least 
# missing data and progresses to the predictor with the most missing data.
# Strategy called multiple imputation by chained equations (MICE) [54], relies on a regression model (we use a 
# Bayesian  ridge regression) to recursively estimate the missing features.

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

class MICE_IterativeImputer(IterativeImputer):
    def __init__(self, 
                 add_indicator=False,
                 estimator=BayesianRidge(), 
#                  estimator=ExtraTreesRegressor(n_jobs=20, max_features='sqrt'), 
                 *,
                 imputation_order='ascending', 
                 initial_strategy='mean',
                 max_iter=10, 
                 max_value=None, 
                 min_value=None,
                 missing_values=np.nan, 
                 n_nearest_features=None, 
                 random_state=0,
                 sample_posterior=False, 
                 skip_complete=False, 
                 tol=0.001,
                 verbose=2
                ):
        super().__init__(estimator)
        self.verbose=2
