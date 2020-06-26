import numpy as np

class GaussianNBClassifier:

    def __init__(self):
        pass

    def separate_classes(self, X, y):
        """
        Separates the dataset in a subset of data for each class.

        Parameters:
        ----------
        X - array-like, list of features
        y - list, target variable

        Returns:
        A dictionnary with y as keys, and the assigned X as values
        """
        separated_classes = {}
        for i in range(len(X)):
            feature_values = X[i]
            class_name = y[i]
            if class_name not in separated_classes:
                separated_classes[class_name] = []
            separated_classes[class_name].append(feature_values)
        return separated_classes
    
    def summarize(self, X):
        """
        """
        for feature in zip(*X):
            yield {
                'stdev' : np.std(feature),
                'mean' : np.mean(feature)
            }
          
    def fit(self, X, y):
        """
        """
        separated_classes = self.separate_classes(X, y)
        self.class_summary = {}
        for class_name, feature_values in separated_classes.items():
            self.class_summary[class_name] = {
                'prior_proba': len(feature_values)/len(X),
                'summary': [i for i in self.summarize(feature_values)],
            }     
        return self.class_summary
    
    def gauss_distribution_function(self, x, mean, stdev):
        """
        """
        exponent = np.exp(-((x-mean)**2 / (2*stdev**2)))
        proba = exponent / (np.sqrt(2*np.pi)*stdev)
        return proba
    
    def predict(self, X):
        """
        """
        MAPs = []
        for row in X:
            joint_proba = {}
            for target, features in self.class_summary.items():
                total_features = len(features['summary'])
                likelihood = 1
                for idx in range(total_features):
                    feature = row[idx]
                    mean = features['summary'][idx]['mean']
                    stdev = features['summary'][idx]['stdev']
                    normal_proba = self.gauss_distribution_function(feature, mean, stdev)
                    likelihood *= normal_proba
                prior_proba = features['prior_proba']
                joint_proba[target] = prior_proba * likelihood
            posterior_proba = {}
            marginal_proba = sum(joint_proba.values())
            for target, joint_p in joint_proba.items():
                posterior_proba[target] = joint_p / marginal_proba
            MAP = max(posterior_proba, key=posterior_proba.get)
            MAPs.append(MAP)
        return MAPs
    
    def accuracy(self, y_test, y_pred):
        """
        """
        true_true = 0
        for y_t, y_p in zip(y_test, y_pred):
            if y_t == y_p:
                true_true += 1
        return true_true / len(y_test)
