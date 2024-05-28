from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Class used to return specific ML model
class ModelSelector:
    # Init all models with specified hyperparams
    def __init__(self, hyperparams: dict) -> None:
        self.models = {
            'logistic': self.__get_logistic_model(hyperparams=hyperparams['logistic']),
            'n_bayes': self.__get_naive_bayes(hyperparams=hyperparams['n_bayes']),
            'rand_forest': self.__get_random_forest(hyperparams=hyperparams['rand_forest'])
        }
    
    def select_model(self, model_name: str):
        return self.models[model_name]
    
    def __get_logistic_model(self, hyperparams: dict):
        return LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=hyperparams['max_iter'])
    
    def __get_naive_bayes(self, hyperparams: dict):
        return MultinomialNB(alpha=hyperparams['alpha'])
    
    def __get_random_forest(self, hyperparams: dict):
        return RandomForestClassifier(n_estimators=hyperparams['n_estimators'], random_state=hyperparams['rand_state'])
    
    
    