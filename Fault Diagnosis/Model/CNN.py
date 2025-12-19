#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, MaxPooling1D, Flatten,
    Dropout, concatenate, GaussianNoise
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


FIG_SIZE = (8, 6)
DPI      = 300
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

def save_figure(path_png: str):

    plt.savefig(path_png, dpi=DPI, bbox_inches='tight')
    path_pdf = path_png.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"[Saved] {path_png}")
    print(f"[Saved] {path_pdf}")
    plt.close()

def preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ['time', 'device', 'signals', 'fault']:
        if col not in df.columns:
            raise ValueError(f"CSV 文件缺少必要列: {col}")


    def convert_signals(s):
        if pd.isna(s):
            return []
        parts = [p.strip() for p in s.strip('"').split(',') if p.strip()]
        seq = []
        for p in parts:
            try:
                seq.append(int(p, 16))
            except ValueError:
                pass
        return seq

    df['sig_seq'] = df['signals'].apply(convert_signals)
    df = df[df['sig_seq'].map(len) > 0].reset_index(drop=True)


    max_len = df['sig_seq'].map(len).max()
    seqs = np.array(
        [seq + [0] * (max_len - len(seq)) for seq in df['sig_seq']],
        dtype=np.float32
    ).reshape(-1, max_len, 1)


    sig_scaler = MinMaxScaler((0, 1))
    X_signals = sig_scaler.fit_transform(seqs.reshape(-1, 1)).reshape(seqs.shape)


    dev_le = LabelEncoder().fit(df['device'])
    dev_ints = dev_le.transform(df['device']).reshape(-1, 1)
    X_devices = OneHotEncoder(sparse_output=False).fit_transform(dev_ints)


    X_time = MinMaxScaler((0, 1)).fit_transform(df['time'].astype(float).values.reshape(-1, 1))


    def fault_to_int(f):
        try:
            return int(f, 16) if isinstance(f, str) and f.startswith('0x') else int(f)
        except:
            return 0

    df['fault_int'] = df['fault'].apply(fault_to_int)
    fault_le = LabelEncoder().fit(df['fault_int'])
    y_int = fault_le.transform(df['fault_int'])
    y = to_categorical(y_int, num_classes=len(fault_le.classes_))
    fault_labels = [hex(i) for i in fault_le.classes_]

    return (X_devices, X_signals, X_time), y, y_int, max_len, X_devices.shape[1], y.shape[1], fault_labels

def build_multi_input_model(num_devices, max_seq_length, num_classes):

    dev_in = Input(shape=(num_devices,), name='device_input')
    dev_branch = Dense(32, activation='relu')(dev_in)
    dev_branch = Dropout(0.40)(dev_branch)

    sig_in = Input(shape=(max_seq_length, 1), name='signal_input')
    z = GaussianNoise(0.05)(sig_in)
    x = Conv1D(8, 3, activation='relu', padding='same')(z)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.40)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.40)(x)
    x = Flatten()(x)
    sig_branch = Dense(56, activation='relu')(x)


    time_in = Input(shape=(1,), name='time_input')
    time_branch = Dense(12, activation='relu')(time_in)
    time_branch = Dropout(0.15)(time_branch)


    merged = concatenate([dev_branch, sig_branch, time_branch])
    y = Dense(128, activation='relu', name='feature_layer')(merged)
    y = Dropout(0.50)(y)
    out = Dense(num_classes, activation='softmax')(y)


    optimizer = Adam(learning_rate=9e-4)
    loss_fn = CategoricalCrossentropy(label_smoothing=0.02)

    model = Model(inputs=[dev_in, sig_in, time_in], outputs=out)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    return model

def train_model_extract_metrics(
    csv_path='dataset31.csv',
    random_state=42,
    epochs=100,
    batch_size=160
):

    (X_dev, X_sig, X_time), y, y_int, max_len, num_dev, num_cls, fault_labels = preprocess_data(csv_path)


    Xd_rem, Xd_test, Xs_rem, Xs_test, Xt_rem, Xt_test, y_rem, y_test = train_test_split(
        X_dev, X_sig, X_time, y,
        test_size=0.15, random_state=random_state, stratify=y_int
    )
    val_ratio = 0.15 / 0.85
    Xd_train, Xd_val, Xs_train, Xs_val, Xt_train, Xt_val, y_train, y_val = train_test_split(
        Xd_rem, Xs_rem, Xt_rem, y_rem,
        test_size=val_ratio, random_state=random_state, stratify=np.argmax(y_rem, axis=1)
    )


    model = build_multi_input_model(num_dev, max_len, num_cls)


    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        [Xd_train, Xs_train, Xt_train], y_train,
        validation_data=([Xd_val, Xs_val, Xt_val], y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop, reduce_lr], verbose=1
    )


    loss, acc = model.evaluate([Xd_test, Xs_test, Xt_test], y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}\n")


    y_pred = np.argmax(model.predict([Xd_test, Xs_test, Xt_test], verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=fault_labels, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (raw counts):")
    print(cm)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.imshow(cm, cmap='Blues')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Count', fontsize=18)

    n = len(fault_labels)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(fault_labels, rotation=45, ha='right')
    ax.set_yticklabels(fault_labels)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, cm[i, j],
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm[i, j] > thresh else 'black'
            )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix', pad=12)

    plt.tight_layout()
    save_figure('1confusion_matrix.png')

    feature_extractor = Model(inputs=model.input,
                              outputs=model.get_layer('feature_layer').output)
    features = feature_extractor.predict([Xd_test, Xs_test, Xt_test], verbose=0)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, verbose=0)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    colors = plt.cm.tab20(np.linspace(0, 1, len(fault_labels)))

    for idx_label, label in enumerate(np.unique(y_true)):
        idx = (y_true == label)
        ax.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            s=64, alpha=0.85,
            c=[colors[idx_label]],
            marker='o',
            edgecolors='none',
            label=fault_labels[label]
        )

    ax.set_title('t-SNE Visualization')
    ax.set_xlabel('t-SNE')
    ax.set_ylabel('t-SNE')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)

    ncol = 2 if len(fault_labels) <= 10 else 3
    leg = ax.legend(
        title='Classes',
        ncol=ncol,
        frameon=True,
        framealpha=0.9,
        borderpad=0.8,
        loc='upper right'
    )
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(16)

    plt.tight_layout()
    save_figure('1tsne_features.png')

    model.save('fault_diagnosis_with_time.keras')
    print("\nModel saved to fault_diagnosis_with_time.keras")

    return history.history

if __name__ == '__main__':
    train_model_extract_metrics()
