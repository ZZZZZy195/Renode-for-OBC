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
    Conv1D, BatchNormalization, Activation,
    GlobalAveragePooling1D, GlobalMaxPooling1D,
    LayerNormalization, MultiHeadAttention,
    Reshape, Multiply
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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



class PositionalEncoding(Layer):
    def call(self, inputs):

        shape = tf.shape(inputs)
        T = shape[1]
        C_static = inputs.shape[-1]
        C = C_static if C_static is not None else shape[2]

        pos = tf.cast(tf.range(T)[:, tf.newaxis], tf.float32)
        i   = tf.cast(tf.range(C)[tf.newaxis, :], tf.float32)

        angle_rates = tf.pow(10000.0, -(2.0 * tf.floor(i / 2.0)) / tf.cast(C, tf.float32))
        angle_rads  = pos * angle_rates

        even_mask = tf.cast(tf.equal(tf.math.mod(tf.range(C), 2), 0), tf.float32)[tf.newaxis, :]
        odd_mask  = 1.0 - even_mask
        pe = tf.sin(angle_rads) * even_mask + tf.cos(angle_rads) * odd_mask

        pe = tf.cast(pe[tf.newaxis, :, :], inputs.dtype)
        return inputs + pe

    def compute_output_shape(self, input_shape):
        return input_shape



def transformer_encoder(x, num_heads, d_model, dff, dropout=0.1, name_prefix="tr"):

    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        name=f"{name_prefix}_mha"
    )(x, x)
    attn_out = Dropout(dropout, name=f"{name_prefix}_drop1")(attn_out)
    x = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x + attn_out)

    ffn = Dense(dff, activation='relu', name=f"{name_prefix}_ffn1")(x)
    ffn = Dense(d_model, name=f"{name_prefix}_ffn2")(ffn)
    ffn = Dropout(dropout, name=f"{name_prefix}_drop2")(ffn)
    x = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x + ffn)
    return x



def se_block(x, reduction=8, name_prefix="se"):
    ch = int(x.shape[-1])
    y = GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    y = Dense(max(ch // reduction, 8), activation='relu', name=f"{name_prefix}_fc1")(y)
    y = Dense(ch, activation='sigmoid', name=f"{name_prefix}_fc2")(y)
    y = Reshape((1, ch), name=f"{name_prefix}_reshape")(y)
    return Multiply(name=f"{name_prefix}_scale")([x, y])


def WKN_block(x, filters=32, kernels=(3,5,7), dilation_rates=(1,2), name_prefix="wkn"):
    branches = []
    for k in kernels:
        for d in dilation_rates:
            y = Conv1D(filters, k, padding='same', dilation_rate=d, use_bias=False,
                       name=f"{name_prefix}_conv_k{k}_d{d}")(x)
            y = BatchNormalization(name=f"{name_prefix}_bn_k{k}_d{d}")(y)
            y = Activation('relu', name=f"{name_prefix}_relu_k{k}_d{d}")(y)
            branches.append(y)
    x = concatenate(branches, name=f"{name_prefix}_concat")
    out_ch = filters * 2
    x = Conv1D(out_ch, 1, padding='same', use_bias=False, name=f"{name_prefix}_conv1x1")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn1x1")(x)
    x = Activation('relu', name=f"{name_prefix}_relu1x1")(x)
    x = se_block(x, reduction=8, name_prefix=f"{name_prefix}_se")
    x = Dropout(0.15, name=f"{name_prefix}_drop")(x)
    return x


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
    seqs = np.array([seq + [0]*(max_len-len(seq)) for seq in df['sig_seq']], dtype=np.float32)
    seqs = seqs.reshape(-1, max_len, 1)
    X_sig = MinMaxScaler((0,1)).fit_transform(seqs.reshape(-1,1)).reshape(seqs.shape)

    dev_le = LabelEncoder().fit(df['device'])
    dev_idx = dev_le.transform(df['device']).reshape(-1,1)
    try:
        X_dev = OneHotEncoder(sparse_output=False).fit_transform(dev_idx)
    except TypeError:
        X_dev = OneHotEncoder(sparse=False).fit_transform(dev_idx)

    X_time = MinMaxScaler((0,1)).fit_transform(df['time'].astype(float).values.reshape(-1,1))

    def f2i(f):
        try:
            return int(f,16) if isinstance(f, str) and f.startswith('0x') else int(f)
        except Exception:
            return 0
    df['fault_int'] = df['fault'].apply(f2i)
    fault_le = LabelEncoder().fit(df['fault_int'])
    y = to_categorical(fault_le.transform(df['fault_int']),
                       num_classes=len(fault_le.classes_))
    labels = [hex(i) for i in fault_le.classes_]

    return (X_dev, X_sig, X_time), y, max_len, X_dev.shape[1], y.shape[1], labels


def build_model(num_dev, seq_len, num_cls):

    inp_dev = Input((num_dev,), name='device_input')
    b_dev = Dense(64, activation='relu', name='dev_fc')(inp_dev)

    inp_sig = Input((seq_len, 1), name='signal_input')

    x = WKN_block(inp_sig, filters=32, kernels=(3,5,7), dilation_rates=(1,2), name_prefix="wkn1")
    x = WKN_block(x,      filters=32, kernels=(3,5,7), dilation_rates=(1,2), name_prefix="wkn2")

    d_model   = 128
    num_heads = 4
    dff       = 4 * d_model
    num_layers= 3
    dropout   = 0.1

    x = Dense(d_model, activation=None, name='proj_to_dmodel')(x)  # [B,T,d_model]
    x = PositionalEncoding(name='pos_encoding')(x)

    for i in range(num_layers):
        x = transformer_encoder(
            x, num_heads=num_heads, d_model=d_model, dff=dff,
            dropout=dropout, name_prefix=f"encoder{i+1}"
        )

    gap = GlobalAveragePooling1D(name='sig_gap')(x)
    gmp = GlobalMaxPooling1D(name='sig_gmp')(x)
    b_sig = concatenate([gap, gmp], name='sig_pool_concat')     # [B, 2*d_model]
    b_sig = Dense(128, activation='relu', name='sig_fc')(b_sig)
    b_sig = Dropout(0.3, name='sig_drop')(b_sig)

    inp_time = Input((1,), name='time_input')
    b_time = Dense(16, activation='relu', name='time_fc')(inp_time)

    merged = concatenate([b_dev, b_sig, b_time], name='merge_all')
    x = Dense(256, activation='relu', name='feature_layer')(merged)   # 保留特征层命名
    x = Dropout(0.5, name='head_drop')(x)
    out = Dense(num_cls, activation='softmax', name='clf')(x)

    model = Model([inp_dev, inp_sig, inp_time], out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    CSV = 'dataset31.csv'
    RS = 42; EPOCHS = 100; BS = 32

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

    print(f"样本：{len(y)}，训练：{len(y_tr)}，验证：{len(y_v)}，测试：{len(y_t)}")

    model = build_model(n_dev, seq_len, n_cls)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    hist = model.fit(
        [Xd_tr, Xs_tr, Xt_tr], y_tr,
        validation_data=([Xd_v, Xs_v, Xt_v], y_v),
        epochs=EPOCHS, batch_size=BS,
        callbacks=callbacks, verbose=1
    )

    loss, acc = model.evaluate([Xd_t, Xs_t, Xt_t], y_t, verbose=0)
    print(f"\nTest Loss: {loss:.4f} | Test Acc: {acc:.4f}")

    y_pred = np.argmax(model.predict([Xd_t, Xs_t, Xt_t]), axis=1)
    y_true = np.argmax(y_t, axis=1)

    print("\nClassification Report:")
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
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
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
    save_figure('confusion_matrix_wkn_transformer.png')

    feature_extractor = Model(inputs=model.input,
                              outputs=model.get_layer('feature_layer').output)
    features = feature_extractor.predict([Xd_t, Xs_t, Xt_t])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    for idx_label, lab in enumerate(np.unique(y_true)):
        idx = (y_true == lab)
        ax.scatter(
            features_2d[idx, 0], features_2d[idx, 1],
            s=64, alpha=0.85,
            c=[colors[idx_label]],
            marker='o', edgecolors='none',
            label=labels[lab]
        )

    ax.set_title('t-SNE Visualization')
    ax.set_xlabel('t-SNE'); ax.set_ylabel('t-SNE')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)

    ncol = 2 if len(labels) <= 10 else 3
    leg = ax.legend(title='Classes', ncol=ncol, frameon=True, framealpha=0.9,
                    borderpad=0.8, loc='upper right')
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(16)

    plt.tight_layout()
    save_figure('tsne_features_wkn_transformer.png')

    model.save('fault_diagnosis_wkn_transformer.keras')
    print("\n模型已保存: fault_diagnosis_wkn_transformer.keras")


if __name__ == '__main__':
    main()
