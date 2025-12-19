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
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, concatenate, Layer,
    GlobalAveragePooling1D, BatchNormalization, Activation,
    Bidirectional, LSTM, Conv1D, Reshape, Multiply
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

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


class Attention(Layer):
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.W = self.add_weight(shape=(d, d), initializer='glorot_uniform',
                                 trainable=True, name='attn_W')
        self.b = self.add_weight(shape=(d,), initializer='zeros',
                                 trainable=True, name='attn_b')
        self.u = self.add_weight(shape=(d,), initializer='glorot_uniform',
                                 trainable=True, name='attn_u')
        super().build(input_shape)

    def call(self, x):

        u_it = K.tanh(K.dot(x, self.W) + self.b)
        ait  = K.sum(u_it * self.u, axis=-1)
        alpha = K.softmax(ait)
        alpha = K.expand_dims(alpha, -1)
        return K.sum(x * alpha, axis=1)

    def get_config(self):
        return super().get_config()


def WKN_block(x, filters=64, kernel_sizes=(3, 5, 7), dilation_rates=(1, 2), name_prefix="wkn"):
    branches = []
    for k in kernel_sizes:
        for d in dilation_rates:
            y = Conv1D(
                filters,
                k,
                padding='same',
                dilation_rate=d,
                use_bias=False,
                kernel_regularizer=regularizers.l2(5e-4),   # ▶ L2 稍微大一点
                name=f"{name_prefix}_conv_k{k}_d{d}"
            )(x)
            y = BatchNormalization(name=f"{name_prefix}_bn_k{k}_d{d}")(y)
            y = Activation('relu', name=f"{name_prefix}_relu_k{k}_d{d}")(y)
            branches.append(y)

    merged = concatenate(branches, name=f"{name_prefix}_concat")  # [B,T, filters * num_branches]
    ch = filters * len(branches)

    se = GlobalAveragePooling1D(name=f"{name_prefix}_se_gap")(merged)  # [B, ch]
    se = Dense(max(ch // 8, 8), activation='relu', name=f"{name_prefix}_se_fc1")(se)
    se = Dense(ch, activation='sigmoid', name=f"{name_prefix}_se_fc2")(se)  # [B, ch]
    se = Reshape((1, ch), name=f"{name_prefix}_se_reshape")(se)             # [B,1,ch]
    merged = Multiply(name=f"{name_prefix}_se_scale")([merged, se])         # [B,T,ch]
    return merged


def preprocess_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    for col in ['time', 'device', 'signals', 'fault']:
        if col not in df.columns:
            raise ValueError(f"缺少列: {col}")

    def hex2seq(s):
        if pd.isna(s):
            return []
        parts = [p.strip() for p in s.strip('"').split(',') if p.strip()]
        seq = []
        for p in parts:
            try:
                seq.append(int(p, 16))
            except Exception:
                pass
        return seq

    df['sig_seq'] = df['signals'].apply(hex2seq)
    df = df[df['sig_seq'].map(len) > 0].reset_index(drop=True)

    max_len = df['sig_seq'].map(len).max()
    seqs = np.array(
        [seq + [0]*(max_len-len(seq)) for seq in df['sig_seq']],
        dtype=np.float32
    )
    seqs = seqs.reshape(-1, max_len, 1)

    X_sig = MinMaxScaler((0, 1)).fit_transform(seqs.reshape(-1, 1)).reshape(seqs.shape)

    dev_le = LabelEncoder().fit(df['device'])
    dev_idx = dev_le.transform(df['device']).reshape(-1, 1)
    try:
        X_dev = OneHotEncoder(sparse_output=False).fit_transform(dev_idx)
    except TypeError:
        X_dev = OneHotEncoder(sparse=False).fit_transform(dev_idx)

    X_time = MinMaxScaler((0, 1)).fit_transform(df['time'].astype(float).values.reshape(-1, 1))

    def f2i(f):
        try:
            return int(f, 16) if isinstance(f, str) and f.startswith('0x') else int(f)
        except Exception:
            return 0

    df['fault_int'] = df['fault'].apply(f2i)
    fault_le = LabelEncoder().fit(df['fault_int'])
    y = to_categorical(
        fault_le.transform(df['fault_int']),
        num_classes=len(fault_le.classes_)
    )
    labels = [hex(i) for i in fault_le.classes_]

    return (X_dev, X_sig, X_time), y, max_len, X_dev.shape[1], y.shape[1], labels


def inject_label_noise(y_onehot, rate, n_cls, seed=42):

    if rate <= 0:
        return y_onehot
    rng = np.random.default_rng(seed)
    y_idx = np.argmax(y_onehot, axis=1)
    n = y_onehot.shape[0]
    mask = rng.random(n) < rate

    rand_labels = rng.integers(0, n_cls, size=n)
    same = rand_labels == y_idx
    rand_labels[same] = (rand_labels[same] + 1) % n_cls

    noisy_idx = y_idx.copy()
    noisy_idx[mask] = rand_labels[mask]
    return to_categorical(noisy_idx, num_classes=n_cls)


def build_model(num_dev, seq_len, num_cls):
    inp_dev = Input((num_dev,), name='device_input')
    b_dev = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(5e-4)   # L2 更强
    )(inp_dev)

    inp_sig = Input((seq_len, 1), name='signal_input')


    x = WKN_block(
        inp_sig,
        filters=64,
        kernel_sizes=(3, 5, 7),
        dilation_rates=(1, 2),
        name_prefix="wkn1"
    )
    x = WKN_block(
        x,
        filters=64,
        kernel_sizes=(3, 5, 7),
        dilation_rates=(1, 2),
        name_prefix="wkn2"
    )

    x = Bidirectional(
        LSTM(
            96,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(5e-4),
            recurrent_regularizer=regularizers.l2(5e-4)
        ),
        name='bilstm1'
    )(x)
    x = Dropout(0.4, name='bilstm1_drop')(x)
    x = Bidirectional(
        LSTM(
            96,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(5e-4),
            recurrent_regularizer=regularizers.l2(5e-4)
        ),
        name='bilstm2'
    )(x)

    b_sig = Attention(name='temporal_attention')(x)   # [B, 192]
    b_sig = Dense(
        96,
        activation='relu',
        kernel_regularizer=regularizers.l2(5e-4),
        name='sig_fc'
    )(b_sig)

    inp_time = Input((1,), name='time_input')
    b_time = Dense(
        16,
        activation='relu',
        kernel_regularizer=regularizers.l2(5e-4)
    )(inp_time)

    merged = concatenate([b_dev, b_sig, b_time])
    x = Dense(
        192,
        activation='relu',
        kernel_regularizer=regularizers.l2(5e-4),
        name='feature_layer'
    )(merged)
    x = Dropout(0.6)(x)
    out = Dense(num_cls, activation='softmax')(x)

    model = Model([inp_dev, inp_sig, inp_time], out)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    CSV = 'dataset31.csv'
    RS = 42
    EPOCHS = 60
    BS = 64

    NOISE_RATE = 0.20

    (X_dev, X_sig, X_time), y, seq_len, n_dev, n_cls, labels = preprocess_data(CSV)

    Xd_r, Xd_t, Xs_r, Xs_t, Xt_r, Xt_t, y_r, y_t = train_test_split(
        X_dev, X_sig, X_time, y,
        test_size=0.15, random_state=RS, stratify=y
    )
    val_r = 0.15 / 0.85
    Xd_tr, Xd_v, Xs_tr, Xs_v, Xt_tr, Xt_v, y_tr, y_v = train_test_split(
        Xd_r, Xs_r, Xt_r, y_r,
        test_size=val_r, random_state=RS, stratify=y_r
    )

    print(f"样本总数：{len(y)}，训练：{len(y_tr)}，验证：{len(y_v)}，测试：{len(y_t)}")
    print(f"训练集标签噪声比例 = {NOISE_RATE}")

    y_tr_noisy = inject_label_noise(y_tr, rate=NOISE_RATE, n_cls=n_cls, seed=RS)

    model = build_model(n_dev, seq_len, n_cls)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]

    hist = model.fit(
        [Xd_tr, Xs_tr, Xt_tr], y_tr_noisy,
        validation_data=([Xd_v, Xs_v, Xt_v], y_v),
        epochs=EPOCHS,
        batch_size=BS,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate([Xd_t, Xs_t, Xt_t], y_t, verbose=0)
    print(f"\nTest Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    y_pred = np.argmax(model.predict([Xd_t, Xs_t, Xt_t]), axis=1)
    y_true = np.argmax(y_t, axis=1)

    print("\nClassification Report (WKN–BiLSTM–AM, reg+noise):")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    # ===== 混淆矩阵 =====
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
                fontsize=14,
                fontweight='bold',
                color='white' if cm[i, j] > thresh else 'black'
            )

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix (WKN–BiLSTM–AM)', pad=12)

    plt.tight_layout()
    save_figure('confusion_matrix_wkn_bilstm_am_reg_noise020.png')

    feature_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer('feature_layer').output
    )
    features = feature_extractor.predict([Xd_t, Xs_t, Xt_t])

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        learning_rate=200
    )
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    for idx_label, lab in enumerate(np.unique(y_true)):
        idx = (y_true == lab)
        ax.scatter(
            features_2d[idx, 0],
            features_2d[idx, 1],
            s=64,
            alpha=0.85,
            c=[colors[idx_label]],
            marker='o',
            edgecolors='none',
            label=labels[lab]
        )

    ax.set_title('t-SNE Visualization (WKN–BiLSTM–AM, reg+noise)')
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
    save_figure('tsne_wkn_bilstm_am_reg_noise020.png')

    model.save('fault_diagnosis_wkn_bilstm_am_reg_noise020.keras')
    print("\n模型已保存: fault_diagnosis_wkn_bilstm_am_reg_noise020.keras")


if __name__ == '__main__':
    main()
