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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, MaxPooling1D, Dropout,
    concatenate, LayerNormalization, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # ← 新增
import tensorflow.keras.backend as K


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


class ChannelAttention(LayerNormalization):
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel = input_shape[-1]
        self.conv = Conv1D(
            self.channel, 1,
            activation='sigmoid',
            padding='same',
            kernel_initializer='he_normal'
        )
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = K.mean(inputs, axis=1, keepdims=True)
        max_pool = K.max(inputs, axis=1, keepdims=True)
        combined = K.concatenate([avg_pool, max_pool], axis=1)
        scale = self.conv(combined)
        scale = K.mean(scale, axis=1, keepdims=True)
        return inputs * scale

class TemporalAttention(LayerNormalization):
    def call(self, inputs):
        attention = K.sigmoid(K.mean(inputs, axis=-1, keepdims=True))
        return inputs * attention

def preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ['time', 'device', 'signals', 'fault']:
        if col not in df.columns:
            raise ValueError(f"Missing column {col}")

    def parse_signal(s):
        if not isinstance(s, str):
            return []
        parts = s.split(',')
        vals = []
        for p in parts:
            p = p.strip().replace('0x', '')
            if p:
                vals.append(int(p, 16))
        return vals

    df['signals_parsed'] = df['signals'].apply(parse_signal)
    df = df[df['signals_parsed'].map(len) > 0].reset_index(drop=True)

    max_len = df['signals_parsed'].map(len).max()
    pads = np.array(
        [seq + [0] * (max_len - len(seq)) for seq in df['signals_parsed']],
        dtype=np.float32
    )

    sig_scaler = MinMaxScaler((0, 1))
    X_sig = sig_scaler.fit_transform(pads.reshape(-1, 1)).reshape(pads.shape + (1,))

    dev_le = LabelEncoder().fit(df['device'])
    dev_idx = dev_le.transform(df['device']).reshape(-1, 1)

    try:
        X_dev = OneHotEncoder(sparse_output=False).fit_transform(dev_idx)
    except TypeError:
        X_dev = OneHotEncoder(sparse=False).fit_transform(dev_idx)

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    X_time = MinMaxScaler((0, 1)).fit_transform(df['time'].values.reshape(-1, 1))

    def parse_fault(f):
        return int(f, 16) if isinstance(f, str) and f.startswith('0x') else int(f)
    df['fault_int'] = df['fault'].apply(parse_fault)
    fenc = LabelEncoder().fit(df['fault_int'])
    y_int = fenc.transform(df['fault_int'])
    y = tf.keras.utils.to_categorical(y_int)

    labels = [f"0x{c}" for c in fenc.classes_]
    return (X_dev, X_sig, X_time), y, max_len, X_dev.shape[1], y.shape[1], labels

def build_model(num_dev, seq_len, num_cls):

    inp_dev = Input((num_dev,), name='device_input')
    bd = Dense(24, activation='relu')(inp_dev)
    bd = Dropout(0.3)(bd)


    inp_sig = Input((seq_len, 1), name='signal_input')
    x = Conv1D(16, 3, padding='same', activation='relu')(inp_sig)
    x = MaxPooling1D(2, padding='same')(x)
    x = ChannelAttention()(x)
    x = TemporalAttention()(x)
    x = Dropout(0.4)(x)
    x = Conv1D(32, 3, padding='same', activation='relu')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = ChannelAttention()(x)
    x = TemporalAttention()(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    sd = Dense(64, activation='relu')(x)


    inp_time = Input((1,), name='time_input')
    td = Dense(12, activation='relu')(inp_time)
    td = Dropout(0.2)(td)


    merged = concatenate([bd, sd, td])
    m = Dense(128, activation='relu', name='feature_layer')(merged)
    m = Dropout(0.5)(m)
    out = Dense(num_cls, activation='softmax')(m)


    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.02)
    opt  = tf.keras.optimizers.Adam(learning_rate=8e-4)

    model = Model([inp_dev, inp_sig, inp_time], out)
    model.compile(opt, loss, metrics=['accuracy'])
    return model

def train_and_evaluate(csv='dataset31.csv'):

    (X_dev, X_sig, X_time), y, seq_len, num_dev, num_cls, labels = preprocess_data(csv)
    y_int = np.argmax(y, axis=1)


    Xd_r, Xd_test, Xs_r, Xs_test, Xt_r, Xt_test, y_r, y_test, yi_r, yi_test = train_test_split(
        X_dev, X_sig, X_time, y, y_int, test_size=0.15, random_state=42, stratify=y_int
    )
    vr = 0.15 / 0.85
    Xd_tr, Xd_val, Xs_tr, Xs_val, Xt_tr, Xt_val, y_tr, y_val, yi_tr, yi_val = train_test_split(
        Xd_r, Xs_r, Xt_r, y_r, yi_r, test_size=vr, random_state=42, stratify=yi_r
    )

    print(f"Samples total: {len(y)} | train: {len(y_tr)} | val: {len(y_val)} | test: {len(y_test)}")

    model = build_model(num_dev, seq_len, num_cls)
    model.summary()


    es  = EarlyStopping('val_loss', patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        [Xd_tr, Xs_tr, Xt_tr], y_tr,
        validation_data=([Xd_val, Xs_val, Xt_val], y_val),
        epochs=100, batch_size=128, callbacks=[es, rlr], verbose=1
    )


    loss, acc = model.evaluate([Xd_test, Xs_test, Xt_test], y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f} | Test Acc: {acc:.4f}\n")

    y_pred = np.argmax(model.predict([Xd_test, Xs_test, Xt_test], verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.imshow(cm, cmap='Blues')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Count', fontsize=18)

    n = len(labels)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

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
    save_figure('2confusion_matrix.png')


    feature_extractor = Model(inputs=model.input,
                              outputs=model.get_layer('feature_layer').output)
    features = feature_extractor.predict([Xd_test, Xs_test, Xt_test], verbose=0)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        learning_rate=200,
        verbose=0
    )
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    unique_labels = np.unique(y_true)
    for idx_label, label_idx in enumerate(unique_labels):
        idx = (y_true == label_idx)
        ax.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            s=64, alpha=0.85,
            c=[colors[idx_label]],
            marker='o',
            edgecolors='none',
            label=labels[label_idx]
        )

    ax.set_title('t-SNE Visualization')
    ax.set_xlabel('t-SNE')
    ax.set_ylabel('t-SNE')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)

    ncol = 2 if len(labels) <= 10 else 3
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
    save_figure('2tsne_features.png')

    # 保存模型为 Keras 原生格式
    model.save('final_model_with_time.keras')
    print("\nModel saved as final_model_with_time.keras")

if __name__ == "__main__":
    train_and_evaluate()
