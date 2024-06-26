from os import path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pd.set_option("future.no_silent_downcasting", True)


class FloodPredictor:
    def __init__(self, config):
        self.config = config

        self.results_dir = self.config['results_dir']
        self.data_dir = self.config['data_dir']
        self.test_size = self.config['test_size']

    def read_data(self, print_info=False):
        df = pd.read_csv(path.join(self.data_dir, "kerala.csv"))
        if print_info:
            print(df.drop(columns=["SUBDIVISION", "FLOODS"]).corr())
            print("HEAD", df.head(5))
            print("INFO", df.info())
            print("DESCRIBE", df.describe())
            print("Correlation matrix")

        df.drop(columns=['SUBDIVISION'], inplace=True)
        df.replace({'FLOODS': {"YES": 1, "NO": 0}}, inplace=True)
        df = df.apply(pd.to_numeric)

        self.df = df
        self.shape = df.shape

    def preprocess_data(self):
        self.feature_selecion()
        self.split_dataset()

    def feature_selecion(self):
        X = self.df.iloc[:, 1:14]
        self.y = self.df.iloc[:, -1]

        # Find top 3 features related to floods
        X_fit = SelectKBest(score_func=chi2, k=3).fit(X, self.y)
        df_features = pd.DataFrame({
            'Features': X.columns,
            'Score': X_fit.scores_
        })
        print(df_features.sort_values(by='Score'))

        # Extract top features
        self.X = self.df[['SEP', 'JUN', 'JUL']]

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X, self.y, test_size=self.test_size, random_state=100)

    def train(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def validate_model(self):
        # Common model performance metrics
        print("Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred))
        print("Recall: ", metrics.recall_score(self.y_test, self.y_pred, zero_division=1))
        print("Precision:", metrics.precision_score(self.y_test, self.y_pred, zero_division=1))
        print("CL Report:", metrics.classification_report(self.y_test, self.y_pred, zero_division=1))

        self.roc_curve()

    def roc_curve(self):
        # Plot Reciever operating characteristic (ROC) curve
        y_pred_prob = self.model.predict_proba(self.X_test)[::, 1]
        false_pos, true_pos, _ = metrics.roc_curve(self.y_test, y_pred_prob)

        auc = metrics.roc_auc_score(self.y_test, y_pred_prob)

        plt.plot(false_pos, true_pos, label=f"AUC={auc:.3f}")
        plt.title('ROC Curve')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    config = {
        'test_size': 0.4,  # [0,1]
        'results_dir': 'results',
        'data_dir': 'data',
        'max_iter': 50,
        'lr': 0.001,
        'nb_units': 50
    }

    model = FloodPredictor(config)
    model.read_data()
    model.preprocess_data()
    model.train()
    model.predict()
    model.validate_model()
