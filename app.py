import os
import json
from datetime import datetime
from pyexpat import model
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
from numpy.fft import rfft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import plotly.express as px
import plotly.io as pio

import lightgbm as lgb
from lightgbm import LGBMClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib
matplotlib.use("Agg")  # evita erro de thread/tkinter
import matplotlib.pyplot as plt

import optuna

app = Flask(__name__)
app.secret_key = "troque-isto-por-uma-chave-secreta"

# ==============================
# Diretórios
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
HIST_DIR = os.path.join(BASE_DIR, "historico")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

# ==============================
# FUNÇÕES AUXILIARES
# ==============================
def FFT_coluna(serie, tempo):
    y = np.array(serie)
    N = len(y)
    if N < 2:
        return np.array([])
    Y = rfft(y)
    mag = np.abs(Y) / N
    if N % 2 == 0:
        mag[1:-1] *= 2.0
    else:
        mag[1:] *= 2.0
    return mag

def processar_classe(dados_classe, bloco=50):
    dados_T = dados_classe.T
    n_rows, n_cols = dados_T.shape
    n_segmentos = n_cols // bloco
    tempo = np.linspace(0.0001, 1.0, n_rows)
    segmentos_fft = []
    for i in range(n_segmentos):
        seg = dados_T[:, i*bloco:(i+1)*bloco]
        fft_cols = [FFT_coluna(seg[:, c], tempo) for c in range(seg.shape[1])]
        vetor_fft = np.hstack(fft_cols)
        segmentos_fft.append(vetor_fft)
    if len(segmentos_fft) == 0:
        return np.empty((0, 0))
    return np.array(segmentos_fft)

def processar_eixo_global(dados, labels, nome="Eixo", bloco=50):
    classes = np.unique(labels)
    all_feats, all_labels = [], []
    for classe in classes:
        dados_classe = dados[labels == classe][:10000, :200]
        feats = processar_classe(dados_classe, bloco=bloco)
        if feats.size == 0:
            continue
        all_feats.append(feats)
        all_labels.append(np.full(feats.shape[0], classe))

    if not all_feats:
        return np.empty((0, 0)), np.array([])

    eixo_feats = np.vstack(all_feats)
    eixo_labels = np.concatenate(all_labels)

    scaler = StandardScaler()
    eixo_feats_norm = scaler.fit_transform(eixo_feats)

    return eixo_feats_norm, eixo_labels

import uuid
def gerar_plots_and_save(fft_xyz_norm, labels_xyz, y_test, y_preds_dict, hid):
    run_dir = os.path.join(IMAGES_DIR, hid)
    os.makedirs(run_dir, exist_ok=True)

    # === PCA 2D (matplotlib) ===
    pca_2d = PCA(n_components=2)
    fft_xyz_pca_2d = pca_2d.fit_transform(fft_xyz_norm)
    plt.figure(figsize=(10, 7))
    for c in np.unique(labels_xyz):
        plt.scatter(
            fft_xyz_pca_2d[labels_xyz == c, 0],
            fft_xyz_pca_2d[labels_xyz == c, 1],
            label=f"Classe {c}", alpha=0.5, s=10
        )
    plt.title("PCA 2D - FFT concatenado (X, Y, Z)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    pca_2d_path = os.path.join(run_dir, "pca_2d.png")
    plt.tight_layout()
    plt.savefig(pca_2d_path)
    plt.close()

    pca_3d = PCA(n_components=3)
    fft_xyz_pca_3d = pca_3d.fit_transform(fft_xyz_norm)
    fig = px.scatter_3d(
        x=fft_xyz_pca_3d[:,0],
        y=fft_xyz_pca_3d[:,1],
        z=fft_xyz_pca_3d[:,2],
        color=[str(c) for c in labels_xyz],
        labels={"color":"Classe"},
        title="PCA 3D - FFT concatenado (X, Y, Z)"
        )
    pca_3d_path = os.path.join(run_dir, "pca_3d.html")
    pio.write_html(fig, file=pca_3d_path, auto_open=False)

    # === Model Accuracies ===
    accuracies = {name: accuracy_score(y_test, y_pred) for name, y_pred in y_preds_dict.items()}
    
    plt.figure(figsize=(8, 5))
    plt.bar(list(accuracies.keys()), list(accuracies.values()))
    plt.title("Model Accuracies")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    rank_path = os.path.join(run_dir, "model_accuracies.png")
    plt.tight_layout()
    plt.savefig(rank_path)
    plt.close()

    # === Confusion Matrices ===
    conf_paths = {}
    for name, y_pred in y_preds_dict.items():
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {name}")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        cm_path = os.path.join(run_dir, f"confusion_{name}.png")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        conf_paths[name] = f"images/{hid}/confusion_{name}.png"

    return {
        "pca_2d": f"images/{hid}/pca_2d.png",
        "pca_3d": f"images/{hid}/pca_3d.html",  # link para abrir no navegador
        
        "accuracies": f"images/{hid}/model_accuracies.png",
        "confusions": conf_paths
    }

# ==============================
# Rotas
# ==============================
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/run", methods=["POST"])
def run_pipeline():
    bloco = int(request.form.get("bloco", 50))
    n_trials = int(request.form.get("n_trials", 5))
    hid = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ==== carregar dados ====
    try:
        labels = np.load(os.path.join(BASE_DIR, "dados", "gt_1500_10.npy"))
        x = np.load(os.path.join(BASE_DIR, "dados", "x_1500_10.npy"))
        y = np.load(os.path.join(BASE_DIR, "dados", "y_1500_10.npy"))
        z = np.load(os.path.join(BASE_DIR, "dados", "z_1500_10.npy"))
    except Exception as e:
        flash(f"Erro ao carregar arquivos numpy: {e}", "danger")
        return redirect(url_for("index"))

    n, m = 50000, 200
    labels, x, y, z = labels[:n], x[:n, :m], y[:n, :m], z[:n, :m]   

    fft_x_norm, labels_x = processar_eixo_global(x, labels, "Eixo X", bloco=bloco)
    fft_y_norm, labels_y = processar_eixo_global(y, labels, "Eixo Y", bloco=bloco)
    fft_z_norm, labels_z = processar_eixo_global(z, labels, "Eixo Z", bloco=bloco)

    if fft_x_norm.size == 0 or fft_y_norm.size == 0 or fft_z_norm.size == 0:
        flash("Processamento devolveu arrays vazios.", "danger")
        return redirect(url_for("index"))

    fft_xyz_norm = np.hstack([fft_x_norm, fft_y_norm, fft_z_norm])
    labels_xyz = labels_x

    X_train, X_test, y_train, y_test = train_test_split(
        fft_xyz_norm, labels_xyz, test_size=0.2, random_state=42, stratify=labels_xyz
    )

    resultados = {}
    y_preds_dict = {}

    # =========================
    # Modelos + Otimização
    # =========================

    # SVM
    def objective_svm(trial):
        C = trial.suggest_loguniform('C', 1e-3, 1e3)
        gamma = trial.suggest_loguniform('gamma', 1e-1, 1e3)
        kernel = trial.suggest_categorical('kernel', [ 'poly', 'sigmoid'])
        model = SVC(C=C, gamma=gamma, kernel=kernel, class_weight="balanced")
        return cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1).mean()

    study_svm = optuna.create_study(direction="maximize")
    study_svm.optimize(objective_svm, n_trials=n_trials)
    best_svm = SVC(**study_svm.best_params, class_weight="balanced").fit(X_train, y_train)
    y_pred_svm = best_svm.predict(X_test)
    resultados["SVM"] = {
        "best_params": study_svm.best_params,
        "best_score_cv": study_svm.best_value,
        "classification_report": classification_report(y_test, y_pred_svm, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_svm).tolist()
    }
    y_preds_dict["SVM"] = y_pred_svm

    # # Random Forest
    def objective_rf(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 150)
        max_depth = trial.suggest_int("max_depth", 20, 60)
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        return cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1).mean()

    study_rf = optuna.create_study(direction="maximize")
    study_rf.optimize(objective_rf, n_trials=n_trials)
    best_rf = RandomForestClassifier(**study_rf.best_params, random_state=42).fit(X_train, y_train)
    y_pred_rf = best_rf.predict(X_test)
    resultados["RandomForest"] = {
        "best_params": study_rf.best_params,
        "best_score_cv": study_rf.best_value,
        "classification_report": classification_report(y_test, y_pred_rf, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_rf).tolist()
    }
    y_preds_dict["RandomForest"] = y_pred_rf

    # KNN
    # def objective_knn(trial):
    #     n_neighbors = trial.suggest_int("n_neighbors", 2, 100)   # ampliar range
    #     weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    #     leaf_size = trial.suggest_int("leaf_size", 10, 100)      # range mais prático
    #     p = trial.suggest_int("p", 1, 2)  # 1=Manhattan, 2=Euclidean
    #     metric = trial.suggest_categorical("metric", ["minkowski", "chebyshev", "cosine"])

    #     model = KNeighborsClassifier(
    #         n_neighbors=n_neighbors,
    #         weights=weights,
    #         leaf_size=leaf_size,
    #         p=p if metric == "minkowski" else 2,  # p só vale para Minkowski
    #         metric=metric
    #     )

    #     return cross_val_score(
    #         model,
    #         X_train,
    #         y_train,
    #         cv=3,                  # CV=3 mais estável que 2
    #         scoring="f1_macro",    # f1_macro lida melhor com desbalanceamento
    #         n_jobs=-1
    #     ).mean()


    # # Estudo com Optuna
    # study_knn = optuna.create_study(direction="maximize")
    # study_knn.optimize(objective_knn, n_trials=n_trials)

    # # Melhor modelo
    # best_knn = KNeighborsClassifier(**study_knn.best_params).fit(X_train, y_train)

    # y_pred_knn = best_knn.predict(X_test)

    # resultados["KNN"] = {
    #     "best_params": study_knn.best_params,
    #     "best_score_cv": study_knn.best_value,
    #     "classification_report": classification_report(y_test, y_pred_knn, output_dict=True),
    #     "confusion_matrix": confusion_matrix(y_test, y_pred_knn).tolist()
    # }
    # y_preds_dict["KNN"] = y_pred_knn
    

    # def objective_xgb(trial):
    #     n_estimators = trial.suggest_int("n_estimators", 50, 100)
    #     max_depth = trial.suggest_int("max_depth", 2, 10)
    #     learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        
    #     model = XGBClassifier(
    #         n_estimators=n_estimators,
    #         max_depth=max_depth,
    #         learning_rate=learning_rate,
    #         tree_method="hist",   # usa método hist
    #         device="cuda",        # indica GPU
    #         eval_metric="mlogloss",
    #         random_state=42
    #     )
        
    #     return cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1).mean()

    # study_xgb = optuna.create_study(direction="maximize")
    # study_xgb.optimize(objective_xgb, n_trials=n_trials)

    # best_xgb = XGBClassifier(
    #     **study_xgb.best_params,
    #     tree_method="hist",
    #     device="cuda",
    #     eval_metric="mlogloss",
    #     random_state=42
    # ).fit(X_train, y_train)

    # y_pred_xgb = best_xgb.predict(X_test)

    # resultados["XGB"] = {
    #     "best_params": study_xgb.best_params,
    #     "best_score_cv": study_xgb.best_value,
    #     "classification_report": classification_report(y_test, y_pred_xgb, output_dict=True),
    #     "confusion_matrix": confusion_matrix(y_test, y_pred_xgb).tolist()
    # }

    # y_preds_dict["XGB"] = y_pred_xgb


    # =========================
  



    # Logistic Regression
    # def objective_lr(trial):
    #     # Menor faixa de C para forçar regularização
    #     C = trial.suggest_loguniform("C", 1e-3, 1)  
        
    #     # Permitindo 'l1' e 'l2' com sag/liblinear
    #     solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
    #     penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    #     # Compatibilidade solver x penalty
    #     if solver == "liblinear" and penalty == "l1":
    #         penalty = "l1"
    #     elif solver == "liblinear" and penalty == "l2":
    #         penalty = "l2"
    #     elif solver == "saga":
    #         penalty = penalty

    #     model = LogisticRegression(
    #         C=C,
    #         solver=solver,
    #         penalty=penalty,
    #         max_iter=500,
    #         tol=1e-4,           # mais sensível a variação
    #         class_weight="balanced"  # evita aprender "na marra" classes desbalanceadas
    #     )
        
    #     return cross_val_score(
    #         model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1
    #     ).mean()


    # study_lr = optuna.create_study(direction="maximize")
    # study_lr.optimize(objective_lr, n_trials=n_trials)

    # # Treino final com os melhores hiperparâmetros encontrados
    # best_lr = LogisticRegression(
    #     **study_lr.best_params, max_iter=500
    # ).fit(X_train, y_train)

    # y_pred_lr = best_lr.predict(X_test)

    # resultados["LogisticRegression"] = {
    #     "best_params": study_lr.best_params,
    #     "best_score_cv": study_lr.best_value,
    #     "classification_report": classification_report(y_test, y_pred_lr, output_dict=True),
    #     "confusion_matrix": confusion_matrix(y_test, y_pred_lr).tolist()
    # }

    # y_preds_dict["LogisticRegression"] = y_pred_lr


    # Gradient Boosting
    # def objective_gb(trial):
    #     n_estimators = trial.suggest_int("n_estimators", 5, 10)
    #     learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    #     max_depth = trial.suggest_int("max_depth", 2, 10)
    #     model = GradientBoostingClassifier(
    #         n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42
    #     )
    #     return cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1).mean()

    # study_gb = optuna.create_study(direction="maximize")
    # study_gb.optimize(objective_gb, n_trials=n_trials)
    # best_gb = GradientBoostingClassifier(**study_gb.best_params, random_state=42).fit(X_train, y_train)
    # y_pred_gb = best_gb.predict(X_test)
    # resultados["GradientBoosting"] = {
    #     "best_params": study_gb.best_params,
    #     "best_score_cv": study_gb.best_value,
    #     "classification_report": classification_report(y_test, y_pred_gb, output_dict=True),
    #     "confusion_matrix": confusion_matrix(y_test, y_pred_gb).tolist()
    # }
    # y_preds_dict["GradientBoosting"] = y_pred_gb

    

    # ==== gerar plots e salvar histórico ====
    plot_paths = gerar_plots_and_save(fft_xyz_norm, labels_xyz, y_test, y_preds_dict, hid)

    meta = {"plots": plot_paths, "params": {"bloco": bloco, "n_trials": n_trials}}
    historico_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resultados": resultados,
        "meta": meta,
        "detalhes": {
            "bloco": bloco,
            "n_trials": n_trials,
            "model_params": {name: info["best_params"] for name, info in resultados.items()}
        }
    }

    filepath = os.path.join(HIST_DIR, hid + ".json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(historico_data, f, indent=4, ensure_ascii=False)

    return render_template("done.html", hid=hid)

@app.route("/historico")
def historico():
    arquivos = sorted(os.listdir(HIST_DIR), reverse=True)
    items = []
    for f in arquivos:
        path = os.path.join(HIST_DIR, f)
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        items.append({"id": f.replace(".json", ""), "timestamp": data.get("timestamp", "sem data")})
    return render_template("historico.html", historicos=items)

@app.route("/results/<hid>")
def results(hid):
    filepath = os.path.join(HIST_DIR, hid + ".json")
    if not os.path.exists(filepath):
        flash("Histórico não encontrado.", "danger")
        return redirect(url_for("historico"))

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    resultados, meta = data["resultados"], data["meta"]

    ranking = []
    for name, info in resultados.items():
        cr = info.get("classification_report", {})
        acc = cr.get("accuracy") or cr.get("macro avg", {}).get("recall", 0)
        ranking.append({"model": name, "accuracy": acc})
    ranking = sorted(ranking, key=lambda x: x["accuracy"], reverse=True)

    return render_template("results.html", resultados=resultados, meta=meta, ranking=ranking)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
