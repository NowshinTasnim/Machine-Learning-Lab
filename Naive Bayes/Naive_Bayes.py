from collections import Counter, defaultdict
import numpy as np
class NaiveBayes(object):
    def __init__(self):
        self.classes = 0


    def occurrences(self, y):
        len_of_y = len(y)
        prob = dict(Counter(y))
        for key in prob.keys():
            prob[key] = prob[key] / float(len_of_y)
        return prob

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.rows, self.cols = np.shape(X_train)
        self.likelihoods = {}
        self.y_train = y_train
        for cls in self.classes:
            self.likelihoods[cls] = defaultdict(list)

        for cls in self.classes:
            row_indices = np.where(y_train == cls)[0]
            subset = X_train[row_indices, :]
            r, c = np.shape(subset)
            for j in range(0, c):
                self.likelihoods[cls][j] += list(subset[:, j])

        for cls in self.classes:
            for j in range(0, c):
                self.likelihoods[cls][j] = self.occurrences(self.likelihoods[cls][j])

        return self

    def predict(self, X_test):
        Binary_results = []
        for single_sample in X_test:
            results = {}
            class_probabilities = self.occurrences(self.y_train)

            for cls in self.classes:
                class_probability = class_probabilities[cls]
                for i in range(0, len(single_sample)):
                    relative_feature_values = self.likelihoods[cls][i]
                    if single_sample[i] in relative_feature_values.keys():
                        class_probability *= relative_feature_values[single_sample[i]]
                    else:
                        class_probability *= 0

                    results[cls] = class_probability
            Binary_results.append(max(results, key=lambda key: results[key]))
        return Binary_results


if __name__ == "__main__":
    X_train = np.asarray(((1, 0, 1, 1), (1, 1, 0, 0), (1, 0, 2, 1), (0, 1, 1, 1), (0, 0, 0, 0), (0, 1, 2, 1), (0, 1, 2, 0), (1, 1, 1, 1)));
    y_train = np.asarray((0, 1, 1, 1, 0, 1, 0, 1))
    X_test = np.asarray(((1,0,1,0),(1,0,1,0),(1,0,1,0)))
    nb = NaiveBayes()
    nb.fit(X_train,y_train)
    print(nb.predict(X_test))