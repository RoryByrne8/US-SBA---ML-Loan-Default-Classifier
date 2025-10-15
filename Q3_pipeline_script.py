import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import numpy as np
import matplotlib.pyplot as plt
import datetime
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def preprocess_data(df, target_col='Default', test_size=0.3, random_state=42, smote = True):
    # Function to process our data, runs test train split, scales data and allows us to use SMOTE to balance the data

    df = df.copy()
    df = df.map(lambda x: int(x) if isinstance(x, bool) else x)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
   
    if smote == True:
        sm = SMOTE(random_state=42)
        X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
        print(" Applied SMOTE: Resampled class distribution:", dict(pd.Series(y_train).value_counts()))

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def tune_hyperparams(X, y,n_iter, scoring, cv=5, model= "rf"):
    # Function to tune hyperparameters for different models using RandomizedSearchCV

    if model == "rf":
        model = RandomForestClassifier(random_state=42)
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model == "xgb":
        param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
        }
        model= XGBClassifier(eval_metric='logloss', random_state=42)
    
    elif model == "svm":
        model = SVC(kernel='linear', probability=True, random_state=42)
        param_dist = {
            'C': [0.01, 0.1, 1, 10]
        }
    elif model == "knn":
        model = KNeighborsClassifier()
        param_dist = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],     #  odd numbers to avoid ties
        }
    elif model == "bag":
        model = BaggingClassifier(random_state=42)
        param_dist = {
            'n_estimators': [10, 20, 30],
            'max_samples': [0.5, 0.75, 1.0],
            'max_features': [0.5, 0.75, 1.0]
        }
    else:
        print("Invalid model type. Choose 'rf','xgb', 'svm', 'knn'or 'bag'.")
        return None, None
 
  
    search = RandomizedSearchCV(model, param_dist, n_iter= n_iter, scoring=scoring, cv=cv, n_jobs=-1, random_state=42, verbose = 2)
    print("Tuning combinations received, moving to model fitting ...\n")

    search.fit(X, y)
    print("Tuning model search complete ...\n")
    return search.best_estimator_, search.best_params_



def train_model(model_type, X_train, y_train, n_iter = None, tune=False, cv=5, scoring ="f1"):
    # Function to train the model, either using hyperparameter tuning or default parameters

    if tune:
        
        model, best_params = tune_hyperparams(X_train, y_train, cv=cv, model=model_type, n_iter=n_iter, scoring =scoring)
        print(" Best Model Parameters:", best_params)

    elif tune ==False:

        if model_type == "rf":
            model = RandomForestClassifier(random_state=42)
        elif model_type == "xgb":
            model = XGBClassifier( eval_metric='logloss', random_state=42)
        elif model_type == "svm":
            model = SVC(kernel='linear', probability=True, random_state=42, C=1)
        elif model_type == "knn":
            model = KNeighborsClassifier()
        elif model_type == "bag":
            model = BaggingClassifier(random_state=42)
            
        print("Fitting model on full training set ...\n")
        model.fit(X_train, y_train)
        print("Model fitting complete...\n")
    return model


def cross_validate_thresholds(model, X, y, thresholds=[0.1, 0.2, 0.35, 0.5], cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Prepare a dictionary to store lists of metrics for each threshold
    results = {t: {'f1': [], 'recall': [], 'precision': [], 'fnr': [], 'fpr': [], 'error': []} for t in thresholds}
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
        print(" Applied SMOTE: Resampled class distribution:", dict(pd.Series(y_train).value_counts()))
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_val)[:, 1]

        # Loops through thresholds and runs 5-Fold cross validation on each one 
        for t in thresholds:
            y_pred = (y_probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            f1 = f1_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            err = (fp + fn) / (tp + tn + fp + fn)

            results[t]['f1'].append(f1)
            results[t]['recall'].append(rec)
            results[t]['precision'].append(prec)
            results[t]['fnr'].append(fnr)
            results[t]['fpr'].append(fpr)
            results[t]['error'].append(err)

    # Construct DataFrame from averaged results
    df = pd.DataFrame({
        'Threshold': thresholds,
        'F1_mean': [np.mean(results[t]['f1']) for t in thresholds],
        'Recall_mean': [np.mean(results[t]['recall']) for t in thresholds],
        'Precision_mean': [np.mean(results[t]['precision']) for t in thresholds],
        'FNR_mean': [np.mean(results[t]['fnr']) for t in thresholds],
        'FPR_mean': [np.mean(results[t]['fpr']) for t in thresholds],
        'Error_mean': [np.mean(results[t]['error']) for t in thresholds],
    })

    print("\nCross-Validated Metrics across Thresholds:")
    print(df.round(4))
    
    return df.round(4)


def plot_roc_train_vs_test(model_name, model, X_train, y_train, X_test, y_test):
    plt.figure(figsize=(8, 6))
    for label, X, y in [('Train', X_train, y_train), ('Test', X_test, y_test)]:
        probs = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve: Train vs Test')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_thresholds(model, X_test, y_test, thresholds=[0.1, 0.2, 0.35, 0.5]):
    y_probs = model.predict_proba(X_test)[:, 1]
    results = []

    # Loop through each threshold and calculate metrics
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = precision_score(y_test, y_pred)
        error = (fp + fn) / (tp + tn + fp + fn)
        g_score = np.sqrt(tpr * (1 - fpr))
        f1 = f1_score(y_test, y_pred)
        fnr = fn / (tp + fn) 
        results.append({
            'Threshold': t, 'TPR': tpr, 'FPR': fpr, 'Precision': precision,
            'Error Rate': error, 'G Score': g_score, 'F1 Score': f1, "FNR": fnr
        })
    df = pd.DataFrame(results)
    print("\n Metrics across thresholds:")
    print(df.round(4))
    return df


def get_knn_metrics(model, X_test, y_test, k=None):
    y_pred = model.predict(X_test)

    # We use this function within a loop in our notebook to calculate the metrics and plot confusion matrix for each value of k

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
    g_score = np.sqrt(rec * tnr)
    err = (fp + fn) / (tp + tn + fp + fn)

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    title = f"Confusion Matrix (k={k})" if k is not None else "Confusion Matrix"
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.grid(False)
    plt.show()

    result = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall (TPR)': rec,
        'F1 Score': f1,
        'FNR': fnr,
        'FPR': fpr,
        'TNR': tnr,
        'G-Score': g_score,
        'Error Rate': err
    }

    if k is not None:
        result['k'] = k

    return result


def cross_validate_across_k(X, y, k_values=[1, 3, 5, 7, 9], cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []

    # Loop through each k value and perform 5-fold cross-validation
    for k in k_values:
        metrics_list = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
            print(" Applied SMOTE: Resampled class distribution:", dict(pd.Series(y_train).value_counts()))
    
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)

            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
            g_score = np.sqrt(rec * tnr)
            err = (fp + fn) / (tp + tn + fp + fn)

            metrics_list.append({
                'Accuracy': acc,
                'Precision': prec,
                'Recall (TPR)': rec,
                'F1 Score': f1,
                'FNR': fnr,
                'FPR': fpr,
                'TNR': tnr,
                'G-Score': g_score,
                'Error Rate': err
            })

        # Average across folds
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        avg_metrics['k'] = k
        results.append(avg_metrics)


    return pd.DataFrame(results)[['k'] + [col for col in results[0].keys() if col != 'k']]

def plot_metrics_vs_thresholds(model_name,df, mode="threshold"):

    # Choose x-axis label and column based on mode, just allows for functionality for K and Thresholds 
    if mode == "threshold":
        x_col = "Threshold"
        x_label = "Threshold"
        title = f"{model_name} Performance vs Thresholds"
    elif mode == "knn":
        x_col = "k" if "k" in df.columns else "K"
        x_label = "K Value"
        title = f"{model_name} Performance vs K Neighbors"
    else:
        raise ValueError("Mode must be 'threshold' or 'knn'")

    # Automatically detect which metric columns exist
    metric_cols = [m for m in ['TPR', 'FPR', 'Precision', 'Error Rate', 'G Score', 'F1 Score'] if m in df.columns]
    
    custom_palette = ["#092327", "#0B5351", "#00A9A5", "#4E8098", "#90C2E7",  "#6CC4A1"]
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metric_cols):
     plt.plot(df[x_col], df[metric], marker='o', label=metric, color=custom_palette[i % len(custom_palette)])

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_feature_importance(model_name ,model, feature_names, top_n=None):
    # Function to plot feature importances

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort by importance descending
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, color='skyblue')
    plt.title(f"Top Feature Importances using {model_name}", fontsize=14, weight='bold')
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Name")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_roc_with_threshold(model_name,model, X_test, y_test, threshold=0.5):

    # Function to plot ROC curve with a specific threshold marked on the curve 

    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # Get closest threshold point
    idx = np.argmin(np.abs(thresholds - threshold))
    plt.scatter(fpr[idx], tpr[idx], color='red', label=f"Threshold = {threshold}", zorder=10)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve at the {threshold} Threshold")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_at_threshold(model, X_test, y_test, threshold=0.5):
    # Function to plot confusion matrix at a specific threshold, used in a loop in our notebook to plot confusion matrix for each threshold
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred_thresh = (y_probs >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_thresh)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix at Threshold {threshold}")
    plt.grid(False)
    plt.show()



def knn_error_plot(X_train, X_test, y_train, y_test, k_values=[1,2,3,4,5,6,7,8,9,10,11,12]):

    error_rates = []
    
    # Loop through each k value to calculate and plot  error rate
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        error = 1 - accuracy_score(y_test, preds)
        error_rates.append(error)

    df = pd.DataFrame({
        'k': list(k_values),
        'Error Rate': error_rates
    })

    best_k_error = df.loc[df['Error Rate'].idxmin()]

    plt.figure(figsize=(12, 6))

    # Plot error rate
    plt.subplot(1, 2, 1)
    plt.plot(df['k'], df['Error Rate'], marker='o', color='black')
    plt.scatter(best_k_error['k'], best_k_error['Error Rate'], color='red', label=f"Min Error (k={int(best_k_error['k'])})")
    plt.xlabel('k (Neighbours)')
    plt.ylabel('Error Rate')
    plt.title('Test Error Rate vs k')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return df