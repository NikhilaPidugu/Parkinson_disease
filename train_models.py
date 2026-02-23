# =========================================================
# PARKINSON'S DISEASE DETECTION – TRAINING PIPELINE
# BASE ML + ANN + CNN + CNN+GRAPH + GRU + TRANSFORMER
# AUTOENCODER + ANOMALY + SHAP
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten,
    GRU, Input, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)

import shap

# ================================
# 1. LOAD DATASET (LOCAL FILE)
# ================================
df = pd.read_csv("parkinsons.csv")  # <-- put your CSV here

if 'name' in df.columns:
    df.drop('name', axis=1, inplace=True)

print("\nDATASET PREVIEW")
print(df.head())

# ================================
# 2. PREPROCESSING
# ================================
X = df.drop('status', axis=1).values
y = df['status'].values
feature_names = df.drop('status', axis=1).columns.tolist()

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# ================================
# 3. BASE ML MODELS
# ================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
results['Logistic Regression'] = accuracy_score(y_test, lr.predict(X_test))

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
results['SVM'] = accuracy_score(y_test, svm.predict(X_test))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
results['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))

print("\nBASE ML ACCURACIES")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

# ================================
# 4. ANN
# ================================
ann = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

ann_acc = ann.evaluate(X_test, y_test, verbose=0)[1]
results['ANN'] = ann_acc
print(f"ANN Accuracy: {ann_acc:.4f}")

# ================================
# 5. CNN (1D)
# ================================
X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y
)

cnn = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(Xc_train, yc_train, epochs=30, batch_size=16, verbose=0)

cnn_acc = cnn.evaluate(Xc_test, yc_test, verbose=0)[1]
results['CNN'] = cnn_acc
print(f"CNN Accuracy: {cnn_acc:.4f}")

# ================================
# 6. CNN + GRAPH
# ================================
corr_matrix = np.corrcoef(X_train.T)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix[:15, :15], cmap='coolwarm')
plt.title("Feature Correlation Matrix (Graph Basis)")
plt.show()

X_graph = X_train @ corr_matrix
Xg_train, Xg_test, yg_train, yg_test = train_test_split(
    X_graph, y_train, test_size=0.2, random_state=42, stratify=y_train
)

cnn_graph = Sequential([
    Dense(128, activation='relu', input_shape=(Xg_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

cnn_graph.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_graph.fit(Xg_train, yg_train, epochs=30, batch_size=16, verbose=0)

graph_acc = cnn_graph.evaluate(Xg_test, yg_test, verbose=0)[1]
results['CNN + Graph'] = graph_acc
print(f"CNN + Graph Accuracy: {graph_acc:.4f}")

# ================================
# 7. GRU
# ================================
X_seq = X.reshape(X.shape[0], X.shape[1], 1)
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    X_seq, y, test_size=0.2, random_state=42, stratify=y
)

gru = Sequential([
    GRU(32, input_shape=(X_seq.shape[1], 1)),
    Dense(1, activation='sigmoid')
])

gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru.fit(Xs_train, ys_train, epochs=30, batch_size=16, verbose=0)

gru_acc = gru.evaluate(Xs_test, ys_test, verbose=0)[1]
results['GRU'] = gru_acc
print(f"GRU Accuracy: {gru_acc:.4f}")

# ================================
# 8. TRANSFORMER
# ================================
def build_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

X_tr = X.reshape(X.shape[0], X.shape[1], 1)
Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X_tr, y, test_size=0.2, random_state=42, stratify=y
)

transformer = build_transformer((X_tr.shape[1], 1))
transformer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
transformer.fit(Xt_train, yt_train, epochs=30, batch_size=16, verbose=0)

trans_acc = transformer.evaluate(Xt_test, yt_test, verbose=0)[1]
results['Transformer'] = trans_acc
print(f"Transformer Accuracy: {trans_acc:.4f}")

# ================================
# 9. AUTOENCODER + CLASSIFIER
# ================================
inp = Input(shape=(X_train.shape[1],))
enc = Dense(64, activation='relu')(inp)
enc = Dense(32, activation='relu')(enc)
dec = Dense(64, activation='relu')(enc)
dec = Dense(X_train.shape[1], activation='linear')(dec)

autoencoder = Model(inp, dec)
encoder = Model(inp, enc)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=16, verbose=0)

X_train_enc = encoder.predict(X_train)
X_test_enc = encoder.predict(X_test)

ae_clf = Sequential([
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

ae_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ae_clf.fit(X_train_enc, y_train, epochs=30, batch_size=16, verbose=0)

ae_acc = ae_clf.evaluate(X_test_enc, y_test, verbose=0)[1]
results['Autoencoder + Classifier'] = ae_acc
print(f"AE + Classifier Accuracy: {ae_acc:.4f}")

# ================================
# 10. AUTOENCODER ANOMALY DETECTION
# ================================
X_healthy = X_train[y_train == 0]
autoencoder.fit(X_healthy, X_healthy, epochs=100, batch_size=16, verbose=0)

recon = autoencoder.predict(X_test)
recon_error = np.mean(np.square(X_test - recon), axis=1)

anomaly_auc = roc_auc_score(y_test, recon_error)
print(f"Autoencoder Anomaly ROC-AUC: {anomaly_auc:.4f}")

# ================================
# 11. SHAP (CNN + GRAPH)
# ================================
explainer = shap.KernelExplainer(cnn_graph.predict, Xg_train[:50])
shap_values = explainer.shap_values(Xg_test[:50])
shap.summary_plot(shap_values, Xg_test[:50])

# ================================
# 12. FINAL MODEL COMPARISON
# ================================
plt.figure(figsize=(12,5))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Final Model Accuracy Comparison")
plt.ylim(0,1)
plt.show()

print("\n--- FULL PIPELINE COMPLETED SUCCESSFULLY ---")

# ================================
# 13. SELECT BEST MODEL & SAVE
# ================================
model_objects = {
    'Logistic Regression': lr,
    'SVM': svm,
    'Random Forest': rf,
    'ANN': ann,
    'CNN': cnn,
    'CNN + Graph': cnn_graph,
    'GRU': gru,
    'Transformer': transformer,
    'Autoencoder + Classifier': ae_clf
}

best_model_name = max(results, key=results.get)
best_model = model_objects[best_model_name]

print("Best Model:", best_model_name)
print("Best Accuracy:", results[best_model_name])

pickle.dump(best_model, open("parkinson_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Saved: parkinson_model.pkl and scaler.pkl")