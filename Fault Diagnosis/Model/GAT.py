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
    Input, Dense, Dropout, concatenate, Layer,
    Conv1D, BatchNormalization, Activation,
    GlobalAveragePooling1D, Lambda
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


RAW_EDGES = [
    ('W25Q5121',  'PC1041'),
    ('W25Q5122',  'PC1041'),
    ('TJA1050',   'PC1041'),
    ('SN74AHC2731','ULN2803A1'),
    ('ULN2803A1', 'PC1041'),
    ('SN74AHC2732','ULN2803A2'),
    ('ULN2803A2', 'PC1041'),
    ('SN74HC175', 'ULN2803A3'),
    ('ULN2803A3', 'PC1042'),
    ('ULN2803A3', 'JB1910'),
    ('ADS83441',  'PC1042'),
    ('ADS83442',  'PC1042'),
    ('JB1910',    'PC1041'),
]

def build_adj_binary(dev_classes):
    idx = {name:i for i,name in enumerate(dev_classes)}
    N = len(dev_classes)
    A = np.zeros((N, N), dtype=np.float32)
    for u,v in RAW_EDGES:
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1.0
            A[idx[v], idx[u]] = 1.0
    np.fill_diagonal(A, 1.0)
    return A.astype(np.float32)



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
    dev_classes = list(dev_le.classes_)
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

    return (X_dev, X_sig, X_time), y, max_len, X_dev.shape[1], y.shape[1], labels, dev_classes


def build_signal_encoder(seq_len, out_dim=64):
    inp = Input((seq_len,1), name='signal_input_raw')
    x = Conv1D(32, 3, padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = Conv1D(32, 5, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = GlobalAveragePooling1D(name='sig_gap')(x)
    x = Dense(out_dim, activation='relu', name='sig_repr')(x)
    return Model(inp, x, name='signal_encoder')



class GraphAttention(Layer):
    def __init__(self, out_dim, num_heads=4, attn_dropout=0.1, feat_dropout=0.1,
                 concat_heads=True, activation='elu', adjacency=None, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.feat_dropout = feat_dropout
        self.concat_heads = concat_heads
        self.activation = activation
        assert adjacency is not None, "Adjacency matrix is required for GAT"
        self.A = tf.constant(adjacency, dtype=tf.float32)  # [N,N]

    def build(self, input_shape):
        self.proj = Dense(self.num_heads * self.out_dim, use_bias=False, name=f"{self.name}_W")
        self.a_src = self.add_weight(
            shape=(self.num_heads, self.out_dim, 1),
            initializer='glorot_uniform', trainable=True, name=f"{self.name}_a_src"
        )
        self.a_dst = self.add_weight(
            shape=(self.num_heads, self.out_dim, 1),
            initializer='glorot_uniform', trainable=True, name=f"{self.name}_a_dst"
        )
        self.leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.2)
        if self.activation == 'elu':
            self.act = tf.keras.layers.ELU()
        elif self.activation == 'relu':
            self.act = tf.keras.layers.ReLU()
        elif self.activation is None:
            self.act = tf.identity
        else:
            self.act = tf.keras.layers.Activation(self.activation)
        super().build(input_shape)

    def call(self, X, training=False):
        B = tf.shape(X)[0]
        N = tf.shape(X)[1]
        H = self.num_heads
        D = self.out_dim

        Wh = self.proj(X)
        Wh = tf.reshape(Wh, (B, -1, H, D))
        if self.feat_dropout and training:
            Wh = tf.nn.dropout(Wh, rate=self.feat_dropout)


        Wh_e = tf.expand_dims(Wh, axis=-1)
        a_src_e = tf.reshape(self.a_src, (1, 1, H, D, 1))
        a_dst_e = tf.reshape(self.a_dst, (1, 1, H, D, 1))

        el = tf.reduce_sum(Wh_e * a_src_e, axis=3)
        er = tf.reduce_sum(Wh_e * a_dst_e, axis=3)


        er_T = tf.transpose(er, [0, 3, 2, 1])
        e = el + er_T
        e = self.leaky_relu(e)

        A_exp = tf.expand_dims(tf.expand_dims(self.A, 0), 2)
        neg_inf = tf.constant(-1e9, dtype=e.dtype)
        e_masked = tf.where(tf.equal(A_exp, 1.0), e, neg_inf)

        alpha = tf.nn.softmax(e_masked, axis=3)
        if self.attn_dropout and training:
            alpha = tf.nn.dropout(alpha, rate=self.attn_dropout)


        h_prime = tf.einsum('bihj,bjhd->bihd', alpha, Wh)

        if self.concat_heads:
            out = tf.reshape(h_prime, (B, N, H * D))
        else:
            out = tf.reduce_mean(h_prime, axis=2)

        out = self.act(out)
        return out

    def compute_output_shape(self, input_shape):

        N = input_shape[1]
        if self.concat_heads:
            return (input_shape[0], N, self.num_heads * self.out_dim)
        else:
            return (input_shape[0], N, self.out_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "out_dim": self.out_dim,
            "num_heads": self.num_heads,
            "attn_dropout": self.attn_dropout,
            "feat_dropout": self.feat_dropout,
            "concat_heads": self.concat_heads,
            "activation": self.activation,
        })
        return cfg


def build_model(num_dev, seq_len, num_cls, A_bin_np):
    A_bin = A_bin_np


    inp_dev  = Input((num_dev,), name='device_input_onehot')
    inp_time = Input((1,), name='time_input')
    inp_sig  = Input((seq_len,1), name='signal_input')


    sig_encoder = build_signal_encoder(seq_len, out_dim=64)
    f_sig = sig_encoder(inp_sig)


    X_nodes = Lambda(lambda z: tf.einsum('bi,bf->bif', z[0], z[1]),
                     name='scatter_to_graph')([inp_dev, f_sig])


    X = GraphAttention(out_dim=32, num_heads=4, attn_dropout=0.1, feat_dropout=0.1,
                       concat_heads=True, activation='elu', adjacency=A_bin,
                       name='gat1')(X_nodes)
    X = Dropout(0.2, name='gat1_drop')(X)


    X = GraphAttention(out_dim=32, num_heads=4, attn_dropout=0.1, feat_dropout=0.1,
                       concat_heads=False, activation=None, adjacency=A_bin,
                       name='gat2')(X)
    X = Activation('elu', name='gat2_act')(X)


    h_target = Lambda(lambda z: tf.reduce_sum(z[0] * tf.expand_dims(z[1], -1), axis=1),
                      name='gather_target')([X, inp_dev])


    b_dev = Dense(32, activation='relu', name='dev_fc')(inp_dev)


    b_time = Dense(16, activation='relu', name='time_fc')(inp_time)


    merged = concatenate([h_target, b_dev, b_time], name='merge_all')
    x = Dense(256, activation='relu', name='feature_layer')(merged)  # 保留名称供可视化/导出
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

    (X_dev, X_sig, X_time), y, seq_len, n_dev, n_cls, labels, dev_classes = preprocess_data(CSV)


    A_bin = build_adj_binary(dev_classes)  # [n_dev, n_dev]


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
    print(f"节点数（设备）：{n_dev}")

    model = build_model(n_dev, seq_len, n_cls, A_bin)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    model.fit(
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
    save_figure('confusion_matrix_gat.png')


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
    save_figure('tsne_features_gat.png')


    model.save('fault_diagnosis_gat.keras')
    print("\n模型已保存: fault_diagnosis_gat.keras")


if __name__ == '__main__':
    main()
