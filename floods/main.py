from os import path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pd.set_option("future.no_silent_downcasting", True)
sns.set_theme()


class FloodPredictor:
    def __init__(self, config):
        self.config = config

        self.results_dir = self.config['results_dir']
        self.data_dir = self.config['data_dir']
        self.test_size = self.config['test_size']
        self.k_features = self.config['k_features']
        self.use_normalization = self.config['use_normalization']

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
        self.X = self.df.iloc[:, 1:14]
        if self.use_normalization:
            scaler = MinMaxScaler()
            X_scaler = scaler.fit(self.X)
            self.X = X_scaler.transform(self.X)

        self.y = self.df.iloc[:, -1]

        # Find top 3 features related to floods
        X_fit = SelectKBest(score_func=chi2, k=self.k_features).fit(self.X, self.y)

        # Extract top features
        self.X = self.df[['SEP', 'JUN', 'JUL']]

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(self.X, self.y, test_size=self.test_size, random_state=100)

    def train(self, model="LR"):
        if model == "LR":
            self.model = LogisticRegression()
        elif model == "KNN":
            self.model = KNeighborsClassifier()
        elif model == "SVC":
            self.model = SVC(kernel="rbf", probability=True)
        elif model == "DT":
            self.model = DecisionTreeClassifier()
        elif model == "RFC":
            self.model = RandomForestClassifier(max_depth=3, random_state=0)
        else:
            print(f"{model} is not a valid model.")
            exit()

        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def validate_model(self, model):
        # Common model performance metrics
        accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        recall = metrics.recall_score(self.y_test, self.y_pred, zero_division=1)
        precision = metrics.precision_score(self.y_test, self.y_pred, zero_division=1)
        print("CL Report:", metrics.classification_report(self.y_test, self.y_pred, zero_division=1))

        self.roc_curve(model)

        return accuracy, recall, precision

    def roc_curve(self, model="LR", show_plot=False):
        # Plot Reciever operating characteristic (ROC) curve
        y_pred_prob = self.model.predict_proba(self.X_test)[::, 1]
        false_pos, true_pos, _ = metrics.roc_curve(self.y_test, y_pred_prob)

        auc = metrics.roc_auc_score(self.y_test, y_pred_prob)
        if show_plot:
            plt.plot(false_pos, true_pos, label=f"Model={model}, AUC={auc:.3f}")
            plt.title('ROC Curve')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend()
            plt.show()


def plot_results(models, accuracies, recalls, precisions):
    results = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'Recall': recalls,
        'Precision': precisions
    })
    results.set_index('Model', inplace=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    sns.barplot(results, x='Model', y="Accuracy", hue="Model", ax=ax1, palette="rocket")
    sns.barplot(results, x='Model', y="Recall", hue="Model", ax=ax2, palette="rocket")
    sns.barplot(results, x='Model', y="Precision", hue="Model", ax=ax3, palette="rocket")

    ax1.set_title("Accuracy")
    ax2.set_title("Recall")
    ax3.set_title("Precision")
    plt.show()


def main(config):
    models = ['LR', 'KNN', 'SVC', 'DT', 'RFC']
    accuracies = []
    recalls = []
    precisions = []

    model = FloodPredictor(config)
    model.read_data()
    model.preprocess_data()

    for model_name in models:
        model.train(model_name)
        model.predict()
        accuracy, recall, precision = model.validate_model(model_name)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)

    plot_results(models, accuracies, recalls, precisions)


if __name__ == '__main__':
    config = {
        'test_size': 0.4,  # [0,1]
        'results_dir': 'results',
        'data_dir': 'data',
        'k_features': 4,
        'use_normalization': False
    }
    main(config)
