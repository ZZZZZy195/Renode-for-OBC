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
    Input, Dense, Conv1D, MaxPooling1D,
    Dropout, concatenate, LSTM, Layer, Softmax
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau


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
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight', shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform', trainable=True
        )
        self.b = self.add_weight(
            name='att_bias', shape=(input_shape[-1],),
            initializer='zeros', trainable=True
        )
        self.u = self.add_weight(
            name='att_u', shape=(input_shape[-1],),
            initializer='glorot_uniform', trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        u_it = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # (B,T,F)
        ait = tf.tensordot(u_it, self.u, axes=1)                       # (B,T)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, axis=-1)                              # (B,T,1)
        weighted = inputs * ait
        return tf.reduce_sum(weighted, axis=1)                          # (B,F)


class ConfidencePenalty(Layer):
    def __init__(self, beta=0.04, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta)

    def call(self, probs, training=None):
        p = tf.clip_by_value(probs, 1e-8, 1.0)
        penalty = self.beta * tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p), axis=-1))
        training_flag = tf.constant(False) if training is None else tf.cast(training, tf.bool)

        def train_branch():
            self.add_loss(penalty)
            return probs

        def infer_branch():
            return probs

        return tf.cond(training_flag, train_branch, infer_branch)


class DelayedEarlyStopping(tf.keras.callbacks.Callback):

    def __init__(self, monitor='val_loss', patience=5, start_epoch=60,
                 min_delta=0.0, mode='min', restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best_weights = bool(restore_best_weights)

        if self.mode not in ['min', 'max', 'auto']:
            self.mode = 'auto'

        self._best = None
        self._wait = 0
        self._stopped_epoch = 0
        self._started = False
        self._best_weights = None

    def _is_improvement(self, current, best):
        if self.mode == 'min':
            return current < (best - self.min_delta)
        elif self.mode == 'max':
            return current > (best + self.min_delta)
        else:
            if 'acc' in self.monitor or 'auc' in self.monitor or 'f1' in self.monitor:
                return current > (best + self.min_delta)
            return current < (best - self.min_delta)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)


        if current is None:
            return


        if (epoch + 1) < self.start_epoch:
            return


        if not self._started:
            self._started = True
            self._best = current
            self._wait = 0
            if self.restore_best_weights:
                self._best_weights = self.model.get_weights()
            return


        if self._is_improvement(current, self._best):
            self._best = current
            self._wait = 0
            if self.restore_best_weights:
                self._best_weights = self.model.get_weights()
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._stopped_epoch = epoch + 1
                self.model.stop_training = True
                if self.restore_best_weights and self._best_weights is not None:
                    self.model.set_weights(self._best_weights)


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
    try:
        X_devices = OneHotEncoder(sparse_output=False).fit_transform(dev_ints)
    except TypeError:
        X_devices = OneHotEncoder(sparse=False).fit_transform(dev_ints)
    num_devices = X_devices.shape[1]

    X_time = MinMaxScaler((0, 1)).fit_transform(
        df['time'].astype(float).values.reshape(-1, 1)
    )

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
    num_classes = y.shape[1]

    return (X_devices, X_signals, X_time), y, max_len, num_devices, num_classes, fault_labels


def build_hybrid_model(num_devices, max_seq_length, num_classes,
                       label_smooth=0.02, cp_beta=0.04):
    dev_in = Input(shape=(num_devices,), name='device_input')
    dev_branch = Dense(32, activation='relu')(dev_in)

    sig_in = Input(shape=(max_seq_length, 1), name='signal_input')
    x = Conv1D(24, 3, activation='relu', padding='same')(sig_in)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.35)(x)
    x = Conv1D(24, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.35)(x)
    x = LSTM(24, return_sequences=True)(x)
    x = Attention()(x)
    sig_branch = Dense(96, activation='relu')(x)

    time_in = Input(shape=(1,), name='time_input')
    time_branch = Dense(16, activation='relu')(time_in)

    merged = concatenate([dev_branch, sig_branch, time_branch])
    y = Dense(96, activation='relu', name='feature_layer')(merged)
    y = Dropout(0.45)(y)

    logits = Dense(num_classes, activation=None, name='logits')(y)
    probs  = Softmax(name='softmax')(logits)
    out    = ConfidencePenalty(beta=cp_beta, name='cp')(probs)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)

    model = Model(inputs=[dev_in, sig_in, time_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss=loss_fn,
        metrics=['accuracy']
    )
    return model


def plot_confusion_matrix(cm, class_names, filename='confusion_matrix_v5.png'):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    im = ax.imshow(cm, cmap='Blues')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Count', fontsize=18)

    n = len(class_names)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j],
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix', pad=12)
    plt.tight_layout()
    save_figure(filename)


def main():
    CSV_PATH     = 'dataset31.csv'
    RANDOM_STATE = 42
    EPOCHS       = 100
    BATCH_SIZE   = 96

    (X_dev, X_sig, X_time), y, max_len, num_dev, num_cls, fault_labels = preprocess_data(CSV_PATH)


    Xd_rem, Xd_test, Xs_rem, Xs_test, Xt_rem, Xt_test, y_rem, y_test = train_test_split(
        X_dev, X_sig, X_time, y,
        test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = 0.15 / 0.85
    Xd_train, Xd_val, Xs_train, Xs_val, Xt_train, Xt_val, y_train, y_val = train_test_split(
        Xd_rem, Xs_rem, Xt_rem, y_rem,
        test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_rem
    )

    print(f"样本总数：{len(y)}，训练：{len(y_train)}，验证：{len(y_val)}，测试：{len(y_test)}")

    model = build_hybrid_model(num_dev, max_len, num_cls, label_smooth=0.02, cp_beta=0.04)
    model.summary()


    delayed_early_stop = DelayedEarlyStopping(
        monitor='val_loss', patience=5, start_epoch=60,
        min_delta=0.0, mode='min', restore_best_weights=True
    )
    reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)  #

    history = model.fit(
        [Xd_train, Xs_train, Xt_train], y_train,
        validation_data=([Xd_val, Xs_val, Xt_val], y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[delayed_early_stop, reduce_lr],
        verbose=1
    )


    try:
        tr_loss = history.history.get('loss', [])
        tr_acc  = history.history.get('accuracy', history.history.get('acc', []))
        pd.DataFrame({
            'epoch': np.arange(1, len(tr_loss)+1),
            'train_loss': tr_loss,
            'train_accuracy': tr_acc,
            'val_loss': history.history.get('val_loss', []),
            'val_accuracy': history.history.get('val_accuracy', history.history.get('val_acc', [])),
        }).to_csv('train_metrics.csv', index=False)
        print("[Saved] train_metrics.csv")
    except Exception as e:
        print(f"[Warn] 保存训练曲线数据失败：{e}")


    loss, acc = model.evaluate([Xd_test, Xs_test, Xt_test], y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f} | Test Accuracy: {acc:.4f}\n")


    y_pred = np.argmax(model.predict([Xd_test, Xs_test, Xt_test]), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=fault_labels, digits=4))


    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (raw counts):")
    print(cm)
    plot_confusion_matrix(cm, fault_labels, filename='confusion_matrix_v5.png')


    feature_extractor = Model(inputs=model.input,
                              outputs=model.get_layer('feature_layer').output)
    features = feature_extractor.predict([Xd_test, Xs_test, Xt_test])

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    colors = plt.cm.tab20(np.linspace(0, 1, len(fault_labels)))
    for idx_label, label in enumerate(np.unique(y_true)):
        idx = (y_true == label)
        ax.scatter(features_2d[idx, 0], features_2d[idx, 1],
                   s=64, alpha=0.85, c=[colors[idx_label]], marker='o',
                   edgecolors='none', label=fault_labels[label])
    ax.set_title('t-SNE Visualization')
    ax.set_xlabel('t-SNE'); ax.set_ylabel('t-SNE')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.6)
    ncol = 2 if len(fault_labels) <= 10 else 3
    leg = ax.legend(title='Classes', ncol=ncol, frameon=True, framealpha=0.9,
                    borderpad=0.8, loc='upper right')
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(16)
    plt.tight_layout()
    save_figure('tsne_features_v5.png')

    model.save('fault_diagnosis_hybrid_v5.keras')
    print("\nModel saved to fault_diagnosis_hybrid_v5.keras")

if __name__ == "__main__":
    main()
