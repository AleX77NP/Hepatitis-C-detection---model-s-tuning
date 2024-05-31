from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Class used to return specific ML model
class ModelSelector:
    # Init all models with specified hyperparams
    def __init__(self, hyperparams: dict) -> None:
        self.__models = {
            'logistic': self.__get_logistic_model(hyperparams=hyperparams['logistic']) if hyperparams.get('logistic') else None,
            'g_boost': self.__get_g_boost(hyperparams=hyperparams['g_boost']) if hyperparams.get('g_boost') else None,
            'rand_forest': self.__get_random_forest(hyperparams=hyperparams['rand_forest']) if hyperparams.get('rand_forest') else None
        }
    
    def select_model(self, model_name: str):
        return self.__models[model_name]
    
    def __get_logistic_model(self, hyperparams: dict):
        return LogisticRegression(multi_class='multinomial', solver=hyperparams['solver'], 
                max_iter=hyperparams['max_iter'], C=hyperparams['C'])
    
    def __get_g_boost(self, hyperparams: dict):
        return GradientBoostingClassifier(n_estimators=hyperparams['n_estimators'], learning_rate=hyperparams['l_rate'])
    
    def __get_random_forest(self, hyperparams: dict):
        return RandomForestClassifier(n_estimators=hyperparams['n_estimators'], 
            max_depth=hyperparams['max_depth'], criterion=hyperparams['criterion'])
    
    
    