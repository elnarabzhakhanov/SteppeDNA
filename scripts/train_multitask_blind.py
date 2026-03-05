# Train a TensorFlow/Keras Multi-Task Deep Neural Network (MAVE-Blind)
import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineering import (
    load_phylop_scores, load_mave_scores, load_alphamissense_scores,
    load_structural_features, load_gnomad_frequencies, load_spliceai_scores,
    engineer_features
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("\n============================================================")
print(" SteppeDNA: Multi-Task Keras Network (MAVE-Blind)")
print("============================================================\n")

print("[1] Loading Data & Constructing True Holdout...")
mutation_df = pd.read_csv("brca2_missense_dataset_2.csv")

mave_df = pd.read_csv("data/mave_scores.csv")
def parse_hgvs_pro(hgvs):
    try:
        p = hgvs.replace("p.", "")
        return p[:3], int(''.join(filter(str.isdigit, p))), p[-3:]
    except:
        return None, None, None

mave_set = set()
for _, row in mave_df.iterrows():
    r, p, a = parse_hgvs_pro(row['hgvs_pro'])
    if r:
        mave_set.add((r, p, a))

is_mave = mutation_df.apply(lambda r: (str(r['AA_ref']).strip(), int(r['AA_pos']), str(r['AA_alt']).strip()) in mave_set, axis=1)
mutation_train_df = mutation_df[~is_mave].copy()

phylop = load_phylop_scores(data_dir="data")
mave = load_mave_scores(data_dir="data")
am = load_alphamissense_scores(data_dir="data")
struct = load_structural_features(data_dir="data")
gnomad = load_gnomad_frequencies(data_dir="data")
spliceai = load_spliceai_scores(data_dir="data")

print("[2] Engineering Features (Strict Generalization Set)...")
X = engineer_features(mutation_train_df, phylop, mave, am, struct, gnomad, spliceai_data=spliceai)

y_regression = X['mave_score'].values
has_mave_mask = X['has_mave'].values.astype(np.float32)

mave_cols = ['mave_score', 'has_mave', 'mave_abnormal', 'mave_x_blosum']
X = X.drop(columns=mave_cols, errors='ignore')

noise_cols = [c for c in X.columns if c.startswith('AA_ref_') or c.startswith('AA_alt_') or c.startswith('Mutation_')]
X = X.drop(columns=noise_cols, errors='ignore')

y_classification = mutation_train_df["Label"].values.astype(np.float32)

print("    -> Splitting Data (80% Train, 20% Val)...")
indices = np.arange(len(X))
idx_train, idx_val = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y_classification)

X_train, X_val = X.values[idx_train], X.values[idx_val]
y_c_train, y_c_val = y_classification[idx_train], y_classification[idx_val]
y_r_train, y_r_val = y_regression[idx_train], y_regression[idx_val]
mask_train, mask_val = has_mave_mask[idx_train], has_mave_mask[idx_val]

print("    -> Scaling Data based strictly on Train Set...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)  

with open("data/scaler_multitask_blind.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("data/feature_names_multitask_blind.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

# Calculate class weight mapping for Keras
neg_count = np.sum(y_c_train == 0)
pos_count = np.sum(y_c_train == 1)
total = len(y_c_train)
weight_for_0 = (1 / neg_count) * (total / 2.0)
weight_for_1 = (1 / pos_count) * (total / 2.0)
class_weights = {0: weight_for_0, 1: weight_for_1}

print("\n[3] Building TensorFlow/Keras Dual-Head Model...")

def build_multitask_model(input_dim):
    inputs = Input(shape=(input_dim,))
    
    # Shared Representation (Extracting core biology)
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    shared_features = Dense(64, activation='relu')(x)
    shared_features = BatchNormalization()(shared_features)
    
    # Head 1: Clinical Classification (Pathogenic vs Benign)
    c = Dense(32, activation='relu')(shared_features)
    class_output = Dense(1, activation='sigmoid', name='class_out')(c)
    
    # Head 2: Lab-Assay Regression (Exact MAVE Score prediction)
    r = Dense(32, activation='relu')(shared_features)
    reg_output = Dense(1, activation='linear', name='reg_out')(r)
    
    model = Model(inputs=inputs, outputs=[class_output, reg_output])
    return model

model = build_multitask_model(X_train_s.shape[1])

# Custom Masked MSE Loss for Regression
def masked_mse(y_true, y_pred):
    # y_true will actually be combined (target_value, mask)
    target = y_true[:, 0]
    mask = y_true[:, 1]
    
    # Flatten y_pred to match target shape
    pred = tf.reshape(y_pred, [-1])
    
    # Calculate squared error
    squared_diff = tf.square(target - pred)
    
    # Apply mask
    masked_diff = squared_diff * mask
    
    # Calculate sum and divide by number of unmasked elements (add epsilon to avoid div zero)
    unmasked_count = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum(masked_diff) / unmasked_count

# Prepare the targets to include the mask for the regression head
y_r_train_masked = np.column_stack((y_r_train, mask_train))
y_r_val_masked = np.column_stack((y_r_val, mask_val))

print("\n[4] Compiling Multi-Task Objective Function...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        'class_out': 'binary_crossentropy',
        'reg_out': masked_mse
    },
    loss_weights={
        'class_out': 1.0,  # Alpha
        'reg_out': 0.5     # Beta (secondary regularization task)
    }
)

print("\n[5] Executing Dual-Objective Keras Training Loop...")

# Custom Callback to track ROC-AUC
class RocAucEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.best_auc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)[0] # Get class_out
        try:
            val_auc = roc_auc_score(self.y_val, y_pred)
            print(f" — val_roc_auc: {val_auc:.4f}")
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.model.save("data/brca2_multitask_blind.h5")
        except ValueError:
            pass

# Subclassing so we don't crash the Keras API
import keras.callbacks
class RocAucEvaluation(keras.callbacks.Callback):
    def __init__(self, validation_data=()):
        super(keras.callbacks.Callback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.best_auc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)[0] 
        try:
            val_auc = roc_auc_score(self.y_val, y_pred)
            print(f" — val_roc_auc: {val_auc:.4f}")
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.model.save("data/brca2_multitask_blind.h5")
        except ValueError:
            pass
            
roc_auc_tracker = RocAucEvaluation(validation_data=(X_val_s, y_c_val))

# Manually compute sample weights to bypass Keras multi-output class_weight limitations
sample_weights_train = np.ones(shape=(len(y_c_train),), dtype=np.float32)
sample_weights_train[y_c_train == 0] = weight_for_0
sample_weights_train[y_c_train == 1] = weight_for_1

history = model.fit(
    x=X_train_s,
    y={'class_out': y_c_train, 'reg_out': y_r_train_masked},
    sample_weight={'class_out': sample_weights_train},
    validation_data=(X_val_s, {'class_out': y_c_val, 'reg_out': y_r_val_masked}),
    epochs=50,
    batch_size=64,
    callbacks=[roc_auc_tracker]
)

print(f"\n[6] Keras Multi-Task Network Built. Best Internal Validation AUC: {roc_auc_tracker.best_auc:.4f}")
print("[SUCCESS] Model serialized to data/brca2_multitask_blind.h5")
