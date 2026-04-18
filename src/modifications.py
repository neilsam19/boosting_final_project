import numpy as np
from sklearn.tree import DecisionTreeClassifier
class AdaBoostClipped:
    def __init__(self, T=50, cap=0.01):
        self.T = T
        self.cap = cap
        self.models = []
        self.alphas = []
        self.max_weights = []
        #tracking (for visualization)
        self.training_errors = []
        self.weight_history = []

    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n

        for i in range(self.T):
            #save weights for visualization
            self.weight_history.append(w.copy())

            # 1. Train weak learner (basic decision tree should be fine)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)

            # 2. Predict
            pred = stump.predict(X)

            # 3. Compute weighted error
            err = np.sum(w * (pred != y))

            # Avoid dividing by 0
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # 4. Compute alpha
            alpha = 0.5 * np.log((1 - err) / err)

            # 5. Update weights
            w *= np.exp(-alpha * y * pred)


            #NEW: weight clipping
            w = np.minimum(w, self.cap)

            # 6. Normalize
            self.max_weights.append(np.max(w))
            w /= np.sum(w)
            #print(f"Round {i+1}: max weight = {np.max(w):.6f}")
            # Store model and alpha
            self.models.append(stump)
            self.alphas.append(alpha)

            # Track training error of ensemble
            ensemble_pred = self._predict_internal(X)
            train_err = np.mean(ensemble_pred != y)
            self.training_errors.append(train_err)

    def _predict_internal(self, X):
        """Used during training"""
        final = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final += alpha * model.predict(X)
        return np.sign(final)

    def predict(self, X):
        final = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final += alpha * model.predict(X)
        return np.sign(final)
    #this is basically the entire weighted sum stuff



class AdaBoostPersistent:
    def __init__(self, T=50, threshold=2, damp_factor=0.6):
        self.T = T
        self.threshold = threshold
        self.damp_factor = damp_factor
        self.models = []
        self.alphas = []
        self.max_weights = []
        self.training_errors = []
        self.weight_history = []

    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n
        self.misclassified_counts = np.zeros(n)

        for i in range(self.T):
            #save weights for visualization
            self.weight_history.append(w.copy())

            # 1. Train weak learner (basic decision tree should be fine)
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)

            # 2. Predict
            pred = stump.predict(X)

            incorrect = (pred != y)
            self.misclassified_counts = 0.9 * self.misclassified_counts + incorrect
            # 3. Compute weighted error
            err = np.sum(w * incorrect)

            # Avoid dividing by 0
            err = np.clip(err, 1e-10, 1 - 1e-10)

            # 4. Compute alpha
            alpha = 0.5 * np.log((1 - err) / err)

            # 5. Update weights but with loop:
            for j in range(n):
                if incorrect[j] and self.misclassified_counts[j] >= self.threshold:
                    w[j] *= np.exp(-alpha * y[j] * pred[j] * self.damp_factor)
                else:
                    w[j] *= np.exp(-alpha * y[j] * pred[j])



            # 6. Normalize
            self.max_weights.append(np.max(w))
            w /= np.sum(w)
            #print(f"Round {i+1}: max weight = {np.max(w):.6f}")
            # Store model and alpha
            self.models.append(stump)
            self.alphas.append(alpha)

            # Track training error of ensemble
            ensemble_pred = self._predict_internal(X)
            train_err = np.mean(ensemble_pred != y)
            self.training_errors.append(train_err)

    def _predict_internal(self, X):
        """Used during training"""
        final = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final += alpha * model.predict(X)
        return np.sign(final)

    def predict(self, X):
        final = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final += alpha * model.predict(X)
        return np.sign(final)
    #this is basically the entire weighted sum stuff

class AdaBoostSoft:
    def __init__(self, T=50, beta=0.5):
        self.T = T
        self.beta = beta
        self.models = []
        self.alphas = []
        self.max_weights = []
        self.training_errors = []
        self.weight_history = []

    def fit(self, X, y):
        n = len(y)
        w = np.ones(n) / n

        for t in range(self.T):
            self.weight_history.append(w.copy())

            # Train weak learner
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)

            pred = stump.predict(X)

            incorrect = (pred != y)
            err = np.sum(w * incorrect)
            err = np.clip(err, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - err) / err)

            # SOFT UPDATE
            w *= np.exp(-self.beta * alpha * y * pred)

            self.max_weights.append(np.max(w))
            w /= np.sum(w)

            self.models.append(stump)
            self.alphas.append(alpha)

            ensemble_pred = self._predict_internal(X)
            train_err = np.mean(ensemble_pred != y)
            self.training_errors.append(train_err)

    def _predict_internal(self, X):
        final = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final += alpha * model.predict(X)
        return np.sign(final)

    def predict(self, X):
        final = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final += alpha * model.predict(X)
        return np.sign(final)