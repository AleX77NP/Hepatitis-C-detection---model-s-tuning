from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Class used to return specific ML model
class ModelSelector:
    # Init all models with specified hyperparams
    def __init__(self, hyperparams: dict) -> None:
        self.__models = {
            'logistic': self.__get_logistic_model(hyperparams=hyperparams['logistic']) if hyperparams.get('logistic') else None,
            'n_bayes': self.__get_naive_bayes(hyperparams=hyperparams['n_bayes']) if hyperparams.get('n_bayes') else None,
            'rand_forest': self.__get_random_forest(hyperparams=hyperparams['rand_forest']) if hyperparams.get('rand_forest') else None
        }
    
    def select_model(self, model_name: str):
        return self.__models[model_name]
    
    def __get_logistic_model(self, hyperparams: dict):
        return LogisticRegression(multi_class='multinomial', solver=hyperparams['solver'], 
                max_iter=hyperparams['max_iter'], penalty=hyperparams['penalty'], C=hyperparams['C'])
    
    def __get_naive_bayes(self, hyperparams: dict):
        return MultinomialNB(alpha=hyperparams['alpha'])
    
    def __get_random_forest(self, hyperparams: dict):
        return RandomForestClassifier(n_estimators=hyperparams['n_estimators'], 
            max_depth=hyperparams['max_depth'], criterion=hyperparams['criterion'])
    
    
    