import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
import seaborn as sns
import joblib
import time
import glob
from scipy.stats import pearsonr
from scipy import stats

# Import TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class MusicEmotionPredictorOptimized:
    def __init__(self, dataset_path="./deam_dataset", models_dir="./models_optimized", plots_dir="./plots_results"):
        self.dataset_path = dataset_path
        self.features_path = os.path.join(dataset_path, "features", "features")
        self.annotations_path = os.path.join(dataset_path, "DEAM_Annotations", "annotations", 
                                           "annotations averaged per song", "song_level")
        self.scaler = StandardScaler()
        
        # Models
        self.combined_model = None
        self.optimized_model = None
        
        self.feature_names = []
        self.emotion_categories = ['Positive/Energetic', 'Negative/Depressed']
        
        # Optimization properties
        self.selected_features_indices = None
        self.feature_engineering_enabled = True
        
        # Directory management
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        self._create_directories()
        
    def _create_directories(self):
        """Create necessary directories for models and plots"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        print(f"Directories created: {self.models_dir}, {self.plots_dir}")
        
    def load_annotations(self):
        """Load annotations from dataset"""
        annotations_file = os.path.join(
            self.annotations_path,
            "static_annotations_averaged_songs_1_2000.csv"
        )
        
        if not os.path.exists(annotations_file):
            possible_files = [
                "static_annotations_averaged_per_song.csv",
                "dynamic_annotations_averaged_per_song.csv",
                "annotations_averaged_per_song.csv"
            ]
            
            for filename in possible_files:
                alt_path = os.path.join(self.annotations_path, filename)
                if os.path.exists(alt_path):
                    annotations_file = alt_path
                    break
            else:
                raise FileNotFoundError(f"No annotations file found in {self.annotations_path}")
        
        print(f"Loading annotations from: {annotations_file}")
        annotations = pd.read_csv(annotations_file)
        annotations.columns = annotations.columns.str.strip()
        return annotations
    
    def load_single_track_features(self, track_id):
        """Load features for a single track from pre-extracted CSV file"""
        try:
            csv_path = os.path.join(self.features_path, f"{track_id}.csv")
            if not os.path.exists(csv_path):
                return None
                
            try:
                df = pd.read_csv(csv_path, delimiter=';')
            except:
                df = pd.read_csv(csv_path, delimiter=',')
            
            if 'frameTime' in df.columns:
                df = df.drop('frameTime', axis=1)
            
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df) > 1:
                features = numeric_df.mean().values
            else:
                features = numeric_df.iloc[0].values
                
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features
            
        except Exception as e:
            print(f"Error loading track {track_id}: {e}")
            return None
    
    def get_feature_names(self):
        """Get feature names from the first available file"""
        try:
            csv_files = glob.glob(os.path.join(self.features_path, "*.csv"))
            if not csv_files:
                return None
                
            first_file = csv_files[0]
            try:
                df = pd.read_csv(first_file, delimiter=';')
            except:
                df = pd.read_csv(first_file, delimiter=',')
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'frameTime' in numeric_cols:
                numeric_cols.remove('frameTime')
                
            return numeric_cols
            
        except Exception as e:
            print(f"Error getting feature names: {e}")
            return None
    
    def prepare_data(self, max_samples=None):
        """Prepare training data using pre-extracted features"""
        print("Loading annotations...")
        annotations = self.load_annotations()
        
        available_csv_files = glob.glob(os.path.join(self.features_path, "*.csv"))
        available_track_ids = [os.path.basename(f).replace('.csv', '') for f in available_csv_files]
        
        self.feature_names = self.get_feature_names()
        if self.feature_names is None:
            raise ValueError("Cannot retrieve feature names")
        
        X = []
        y = []
        song_ids = []
        
        annotations_to_process = annotations
        if max_samples is not None:
            annotations_to_process = annotations.head(max_samples)
        
        for index, row in annotations_to_process.iterrows():
            song_id = str(int(row['song_id']))
            
            if song_id in available_track_ids:
                features = self.load_single_track_features(song_id)
                if features is not None and len(features) > 0:
                    X.append(features)
                    y.append([row['valence_mean'], row['arousal_mean']])
                    song_ids.append(song_id)
        
        if not X:
            raise ValueError("No features could be loaded.")
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.song_ids = song_ids
        
        print(f"Data prepared: {len(self.X)} samples, {self.X.shape[1]} features")
        return self.X, self.y
    
    def create_emotion_categories_extreme(self, y):
        """Create 2 extreme categories based on valence and arousal"""
        categories = []
        filtered_indices = []
        
        valence_q25, valence_q75 = np.percentile(y[:, 0], [25, 75])
        arousal_q25, arousal_q75 = np.percentile(y[:, 1], [25, 75])
        
        print(f"Thresholds for extreme categories:")
        print(f"Valence Q25: {valence_q25:.3f}, Q75: {valence_q75:.3f}")
        print(f"Arousal Q25: {arousal_q25:.3f}, Q75: {arousal_q75:.3f}")
        
        for i in range(len(y)):
            v, a = y[i]
            
            if (v >= valence_q75 and a >= arousal_q25) or (a >= arousal_q75 and v >= np.median(y[:, 0])):
                categories.append(0)  # Positive/Energetic
                filtered_indices.append(i)
            elif v <= valence_q25 and a <= arousal_q75:
                categories.append(1)  # Negative/Depressed
                filtered_indices.append(i)
        
        categories = np.array(categories)
        filtered_indices = np.array(filtered_indices)
        
        print(f"Samples retained: {len(categories)}/{len(y)} ({len(categories)/len(y)*100:.1f}%)")
        print(f"Category distribution:")
        print(f"- Positive/Energetic: {np.sum(categories == 0)} ({np.sum(categories == 0)/len(categories)*100:.1f}%)")
        print(f"- Negative/Depressed: {np.sum(categories == 1)} ({np.sum(categories == 1)/len(categories)*100:.1f}%)")
        
        return categories, filtered_indices
    
    def advanced_feature_selection(self, X_train, y_train, y_cat_train, top_k=50):
        """Advanced feature selection based on multiple criteria"""
        print(f"\nAdvanced feature selection (top {top_k})")
        print("="*60)
        
        n_features = X_train.shape[1]
        
        # 1. Random Forest importance
        print("1. Computing Random Forest importance...")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_clf.fit(X_train, y_cat_train)
        rf_importance = rf_clf.feature_importances_
        
        # 2. Mutual Information for classification
        print("2. Computing Mutual Information...")
        mi_scores = mutual_info_classif(X_train, y_cat_train, random_state=42)
        
        # 3. F-score for classification
        print("3. Computing F-scores...")
        f_scores = f_classif(X_train, y_cat_train)[0]
        
        # 4. Correlation with valence and arousal
        print("4. Computing correlations...")
        valence_corr = []
        arousal_corr = []
        
        for i in range(n_features):
            val_corr, _ = pearsonr(X_train[:, i], y_train[:, 0])
            ar_corr, _ = pearsonr(X_train[:, i], y_train[:, 1])
            valence_corr.append(abs(val_corr))
            arousal_corr.append(abs(ar_corr))
        
        # Combine scores (normalization then weighted average)
        def normalize_scores(scores):
            scores = np.array(scores)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        rf_norm = normalize_scores(rf_importance)
        mi_norm = normalize_scores(mi_scores)
        f_norm = normalize_scores(f_scores)
        val_norm = normalize_scores(valence_corr)
        ar_norm = normalize_scores(arousal_corr)
        
        # Weighted composite score
        composite_scores = (
            0.3 * rf_norm +      # Random Forest (main importance)
            0.25 * mi_norm +     # Mutual Information
            0.2 * f_norm +       # F-score
            0.15 * val_norm +    # Valence correlation
            0.1 * ar_norm        # Arousal correlation
        )
        
        # Select top features
        top_indices = np.argsort(composite_scores)[::-1][:top_k]
        self.selected_features_indices = sorted(top_indices)
        
        # Create results DataFrame
        selection_results = pd.DataFrame({
            'feature_name': [self.feature_names[i] for i in top_indices],
            'composite_score': composite_scores[top_indices],
            'rf_importance': rf_importance[top_indices],
            'mutual_info': mi_scores[top_indices],
            'f_score': f_scores[top_indices],
            'valence_corr': [valence_corr[i] for i in top_indices],
            'arousal_corr': [arousal_corr[i] for i in top_indices]
        })
        
        print(f"\nTop {min(15, top_k)} selected features:")
        print(selection_results.head(15).to_string(index=False, float_format='%.4f'))
        
        # Visualize selection
        self._plot_feature_selection(selection_results.head(20))
        
        return self.selected_features_indices, selection_results
    
    def _plot_feature_selection(self, selection_df):
        """Visualize feature selection results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Composite scores
        top_15 = selection_df.head(15)
        bars1 = ax1.barh(range(len(top_15)), top_15['composite_score'], alpha=0.7, color='purple')
        ax1.set_yticks(range(len(top_15)))
        ax1.set_yticklabels(top_15['feature_name'], fontsize=10)
        ax1.set_xlabel('Composite Score')
        ax1.set_title('Top Features - Composite Score')
        ax1.invert_yaxis()
        
        # 2. Score comparison heatmap
        scores_matrix = top_15[['rf_importance', 'mutual_info', 'f_score', 'valence_corr', 'arousal_corr']].values
        
        im = ax2.imshow(scores_matrix.T, cmap='viridis', aspect='auto')
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(['RF Importance', 'Mutual Info', 'F-Score', 'Val Corr', 'Ar Corr'])
        ax2.set_xticks(range(len(top_15)))
        ax2.set_xticklabels([f"F{i+1}" for i in range(len(top_15))], rotation=45)
        ax2.set_title('Score Heatmap by Feature')
        plt.colorbar(im, ax=ax2)
        
        # 3. RF Importance vs Mutual Information
        ax3.scatter(selection_df['rf_importance'], selection_df['mutual_info'], alpha=0.7, s=80)
        ax3.set_xlabel('Random Forest Importance')
        ax3.set_ylabel('Mutual Information')
        ax3.set_title('RF Importance vs Mutual Information')
        ax3.grid(True, alpha=0.3)
        
        # 4. Valence vs Arousal correlations
        scatter = ax4.scatter(selection_df['valence_corr'], selection_df['arousal_corr'], 
                            alpha=0.7, s=80, c=selection_df['composite_score'], cmap='viridis')
        ax4.set_xlabel('Valence Correlation')
        ax4.set_ylabel('Arousal Correlation')
        ax4.set_title('Valence vs Arousal Correlations (color = composite score)')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4)
        
        plt.tight_layout()
        
        selection_path = os.path.join(self.plots_dir, 'advanced_feature_selection.png')
        plt.savefig(selection_path, dpi=300, bbox_inches='tight')
        print(f"Feature selection plot saved: {selection_path}")
        plt.show()
    
    def create_engineered_features(self, X):
        """Create engineered features based on analysis"""
        print("\nCreating engineered features")
        
        if not self.feature_engineering_enabled:
            return X
        
        X_engineered = X.copy()
        new_features = []
        
        try:
            # Identify indices of important features
            mfcc_indices = [i for i, name in enumerate(self.feature_names) if 'mfcc' in name.lower()]
            spectral_indices = [i for i, name in enumerate(self.feature_names) if 'spectral' in name.lower()]
            
            if len(mfcc_indices) >= 2:
                # MFCC ratio
                mfcc_ratio = X[:, mfcc_indices[0]] / (X[:, mfcc_indices[1]] + 1e-8)
                X_engineered = np.column_stack([X_engineered, mfcc_ratio])
                new_features.append("mfcc_ratio_0_1")
            
            if len(spectral_indices) >= 2:
                # Spectral ratio
                spectral_ratio = X[:, spectral_indices[0]] / (X[:, spectral_indices[1]] + 1e-8)
                X_engineered = np.column_stack([X_engineered, spectral_ratio])
                new_features.append("spectral_ratio_0_1")
            
            # 2. Quadratic interaction features for top features
            if self.selected_features_indices is not None and len(self.selected_features_indices) >= 5:
                top_5_indices = self.selected_features_indices[:5]
                
                for i, idx1 in enumerate(top_5_indices[:3]):
                    for idx2 in top_5_indices[i+1:4]:
                        interaction = X[:, idx1] * X[:, idx2]
                        X_engineered = np.column_stack([X_engineered, interaction])
                        new_features.append(f"interaction_{idx1}_{idx2}")
            
            # 3. Power features (square of important features)
            if self.selected_features_indices is not None:
                for idx in self.selected_features_indices[:3]:
                    power_feature = np.power(X[:, idx], 2)
                    X_engineered = np.column_stack([X_engineered, power_feature])
                    new_features.append(f"power_{idx}")
            
            # 4. Log features (for always positive features)
            positive_features = []
            for i in range(X.shape[1]):
                if np.all(X[:, i] > 0):
                    positive_features.append(i)
            
            for idx in positive_features[:5]:  # Limit to 5
                log_feature = np.log(X[:, idx] + 1e-8)
                X_engineered = np.column_stack([X_engineered, log_feature])
                new_features.append(f"log_{idx}")
            
        except Exception as e:
            print(f"Error creating engineered features: {e}")
            return X
        
        print(f"Created {len(new_features)} new features")
        print(f"Dimensions: {X.shape} -> {X_engineered.shape}")
        
        # Update feature names
        self.engineered_feature_names = self.feature_names + new_features
        
        return X_engineered
    
    def create_optimized_model(self, input_dim, use_attention=True, use_regularization=True):
        """Create optimized architecture based on feature analysis"""
        print(f"\nCreating optimized model")
        print(f"Input dimension: {input_dim}")
        
        inputs = layers.Input(shape=(input_dim,), name='main_input')
        
        # 1. Attention mechanism on features (optional)
        if use_attention and input_dim > 50:
            # Simple attention on features
            attention_weights = layers.Dense(input_dim, activation='softmax', name='attention_weights')(inputs)
            attended_features = layers.Multiply(name='attended_features')([inputs, attention_weights])
            x = attended_features
        else:
            x = inputs
        
        # 2. Main layers with optimized architecture
        x = layers.Dense(256, activation='relu', name='dense1')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        if use_regularization:
            x = layers.Dropout(0.3, name='dropout1')(x)
        
        x = layers.Dense(128, activation='relu', name='dense2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        if use_regularization:
            x = layers.Dropout(0.4, name='dropout2')(x)
        
        x = layers.Dense(64, activation='relu', name='dense3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        if use_regularization:
            x = layers.Dropout(0.3, name='dropout3')(x)
        
        # 3. Specialized branches
        shared_features = layers.Dense(32, activation='relu', name='shared_final')(x)
        
        # Valence branch with specialization
        valence_branch = layers.Dense(16, activation='relu', name='valence_spec')(shared_features)
        if use_regularization:
            valence_branch = layers.Dropout(0.2, name='valence_dropout')(valence_branch)
        valence_output = layers.Dense(1, activation='linear', name='valence')(valence_branch)
        
        # Arousal branch with specialization
        arousal_branch = layers.Dense(16, activation='relu', name='arousal_spec')(shared_features)
        if use_regularization:
            arousal_branch = layers.Dropout(0.2, name='arousal_dropout')(arousal_branch)
        arousal_output = layers.Dense(1, activation='linear', name='arousal')(arousal_branch)
        
        # Emotion branch with more complexity
        emotion_branch = layers.Dense(32, activation='relu', name='emotion_spec1')(shared_features)
        emotion_branch = layers.Dense(16, activation='relu', name='emotion_spec2')(emotion_branch)
        if use_regularization:
            emotion_branch = layers.Dropout(0.3, name='emotion_dropout')(emotion_branch)
        emotion_output = layers.Dense(2, activation='softmax', name='emotion')(emotion_branch)
        
        # 4. Create model
        model = keras.Model(
            inputs=inputs,
            outputs=[valence_output, arousal_output, emotion_output]
        )
        
        # 5. Optimizer with fixed learning rate
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        
        # 6. Compilation with adjusted weights
        model.compile(
            optimizer=optimizer,
            loss={
                'valence': 'mse',
                'arousal': 'mse',
                'emotion': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'valence': 1.0,   # Balanced for better correlation
                'arousal': 1.0,   # Balanced for better correlation
                'emotion': 1.0    # Balanced to optimize all tasks
            },
            metrics={
                'valence': ['mae', 'mse'],
                'arousal': ['mae', 'mse'],
                'emotion': ['accuracy']
            }
        )
        
        return model
    
    def train_optimized_model(self, test_size=0.2, epochs=100, use_feature_selection=True, 
                            use_feature_engineering=True, k_best=50):
        """Train model with all optimizations"""
        print("\n" + "="*80)
        print("Training optimized model")
        print("="*80)
        
        # 1. Prepare data with extreme categories
        y_categories_all, filtered_indices = self.create_emotion_categories_extreme(self.y)
        X_filtered = self.X[filtered_indices]
        y_filtered = self.y[filtered_indices]
        y_categories_filtered = y_categories_all
        
        print(f"Filtered data: {len(X_filtered)} samples")
        
        # 2. Train/test split
        X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = train_test_split(
            X_filtered, y_filtered, y_categories_filtered, 
            test_size=test_size, random_state=42, stratify=y_categories_filtered
        )
        
        # 3. Advanced feature selection
        if use_feature_selection:
            selected_indices, selection_results = self.advanced_feature_selection(
                X_train, y_train, y_cat_train, top_k=k_best
            )
            X_train = X_train[:, selected_indices]
            X_test = X_test[:, selected_indices]
            print(f"Selected features: {len(selected_indices)}")
        
        # 4. Feature engineering
        if use_feature_engineering:
            X_train = self.create_engineered_features(X_train)
            X_test = self.create_engineered_features(X_test)
        
        # 5. Normalization
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        input_dim = X_train_scaled.shape[1]
        print(f"Final input dimension: {input_dim}")
        
        # 6. Create optimized model
        self.optimized_model = self.create_optimized_model(
            input_dim, use_attention=True, use_regularization=True
        )
        
        print("\nOptimized model architecture:")
        self.optimized_model.summary()
        
        # 7. Improved callbacks
        callbacks_list = [
            EarlyStopping(
                monitor='val_emotion_accuracy', 
                patience=20, 
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=10, 
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(self.models_dir, 'best_optimized_model.keras'),
                monitor='val_emotion_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # 8. Training
        print(f"\nStarting training (epochs={epochs})...")
        
        history = self.optimized_model.fit(
            X_train_scaled,
            {
                'valence': y_train[:, 0],
                'arousal': y_train[:, 1],
                'emotion': y_cat_train
            },
            validation_data=(
                X_test_scaled,
                {
                    'valence': y_test[:, 0],
                    'arousal': y_test[:, 1],
                    'emotion': y_cat_test
                }
            ),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 9. Evaluation
        print("\nModel evaluation")
        print("="*50)
        
        predictions = self.optimized_model.predict(X_test_scaled)
        valence_pred = predictions[0].flatten()
        arousal_pred = predictions[1].flatten()
        emotion_proba = predictions[2]
        emotion_pred = np.argmax(emotion_proba, axis=1)
        
        # Metrics
        valence_rmse = np.sqrt(mean_squared_error(y_test[:, 0], valence_pred))
        arousal_rmse = np.sqrt(mean_squared_error(y_test[:, 1], arousal_pred))
        valence_mae = mean_absolute_error(y_test[:, 0], valence_pred)
        arousal_mae = mean_absolute_error(y_test[:, 1], arousal_pred)
        emotion_accuracy = accuracy_score(y_cat_test, emotion_pred)
        emotion_f1 = f1_score(y_cat_test, emotion_pred, average='weighted')
        
        valence_corr, _ = pearsonr(y_test[:, 0], valence_pred)
        arousal_corr, _ = pearsonr(y_test[:, 1], arousal_pred)
        
        print(f"Optimized results:")
        print(f"   Valence  - RMSE: {valence_rmse:.4f}, MAE: {valence_mae:.4f}, Corr: {valence_corr:.4f}")
        print(f"   Arousal  - RMSE: {arousal_rmse:.4f}, MAE: {arousal_mae:.4f}, Corr: {arousal_corr:.4f}")
        print(f"   Emotion  - Accuracy: {emotion_accuracy:.4f} ({emotion_accuracy*100:.1f}%), F1: {emotion_f1:.4f}")
        
        # 10. Store results before visualizations
        self.optimization_results = {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_cat_test': y_cat_test,
            'valence_pred': valence_pred,
            'arousal_pred': arousal_pred,
            'emotion_pred': emotion_pred,
            'emotion_proba': emotion_proba,
            'history': history,
            'metrics': {
                'valence_rmse': valence_rmse,
                'valence_mae': valence_mae,
                'valence_corr': valence_corr,
                'arousal_rmse': arousal_rmse,
                'arousal_mae': arousal_mae,
                'arousal_corr': arousal_corr,
                'emotion_accuracy': emotion_accuracy,
                'emotion_f1': emotion_f1
            }
        }
        
        # 11. Create visualizations
        self._create_optimization_plots(
            y_test, y_cat_test, valence_pred, arousal_pred, emotion_pred, 
            history, emotion_accuracy
        )
        
        # 12. Create valence-arousal space visualization
        self._create_valence_arousal_space_plot(
            y_test, valence_pred, arousal_pred, emotion_pred, y_cat_test
        )
        
        # 13. Cross-validation for robustness
        cv_scores = self._perform_cross_validation(X_train_scaled, y_cat_train)
        
        # Add CV results
        self.optimization_results['cv_scores'] = cv_scores
        
        return self.optimization_results
    
    def _create_valence_arousal_space_plot(self, y_test, valence_pred, arousal_pred, emotion_pred, y_cat_test):
        """Create comprehensive valence-arousal space visualization"""
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main plot: Valence-Arousal space with predictions vs reality
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        
        # Real values (true values)
        colors_real = ['blue' if cat == 0 else 'red' for cat in y_cat_test]
        scatter_real = ax_main.scatter(y_test[:, 0], y_test[:, 1], 
                                      c=colors_real, alpha=0.6, s=80, 
                                      marker='o', edgecolors='black', linewidth=1,
                                      label='True Values')
        
        # Arrows from true values to predictions
        for i in range(len(y_test)):
            ax_main.annotate('', xy=(valence_pred[i], arousal_pred[i]), 
                            xytext=(y_test[i, 0], y_test[i, 1]),
                            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4, lw=0.8))
        
        # Predicted values
        colors_pred = ['lightblue' if cat == 0 else 'lightcoral' for cat in emotion_pred]
        scatter_pred = ax_main.scatter(valence_pred, arousal_pred, 
                                      c=colors_pred, alpha=0.8, s=60, 
                                      marker='x', linewidth=2,
                                      label='Predictions')
        
        # Customize main plot
        ax_main.set_xlabel('Valence', fontsize=14, fontweight='bold')
        ax_main.set_ylabel('Arousal', fontsize=14, fontweight='bold')
        ax_main.set_title('Emotional Space Valence-Arousal\nReality vs Predictions', 
                         fontsize=16, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(fontsize=12, loc='upper right')
        
        # Reference lines
        ax_main.axhline(y=np.mean(y_test[:, 1]), color='gray', linestyle='--', alpha=0.5)
        ax_main.axvline(x=np.mean(y_test[:, 0]), color='gray', linestyle='--', alpha=0.5)
        
        # Add emotional quadrants
        mean_val, mean_ar = np.mean(y_test[:, 0]), np.mean(y_test[:, 1])
        ax_main.text(mean_val + 1, mean_ar + 1, 'Happy\n(High Val, High Ar)', 
                    ha='center', va='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        ax_main.text(mean_val - 1, mean_ar + 1, 'Excited\n(Low Val, High Ar)', 
                    ha='center', va='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
        ax_main.text(mean_val - 1, mean_ar - 1, 'Sad\n(Low Val, Low Ar)', 
                    ha='center', va='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax_main.text(mean_val + 1, mean_ar - 1, 'Relaxed\n(High Val, Low Ar)', 
                    ha='center', va='center', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # 2. Valence prediction errors
        ax_val_error = fig.add_subplot(gs[0, 2])
        valence_errors = valence_pred - y_test[:, 0]
        ax_val_error.hist(valence_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax_val_error.set_title('Valence Prediction\nError Distribution', fontsize=12, fontweight='bold')
        ax_val_error.set_xlabel('Error (Predicted - True)')
        ax_val_error.set_ylabel('Frequency')
        ax_val_error.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax_val_error.grid(True, alpha=0.3)
        
        # Valence error statistics
        val_mae = np.mean(np.abs(valence_errors))
        val_rmse = np.sqrt(np.mean(valence_errors**2))
        ax_val_error.text(0.02, 0.98, f'MAE: {val_mae:.3f}\nRMSE: {val_rmse:.3f}', 
                         transform=ax_val_error.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. Arousal prediction errors
        ax_ar_error = fig.add_subplot(gs[1, 2])
        arousal_errors = arousal_pred - y_test[:, 1]
        ax_ar_error.hist(arousal_errors, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax_ar_error.set_title('Arousal Prediction\nError Distribution', fontsize=12, fontweight='bold')
        ax_ar_error.set_xlabel('Error (Predicted - True)')
        ax_ar_error.set_ylabel('Frequency')
        ax_ar_error.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax_ar_error.grid(True, alpha=0.3)
        
        # Arousal error statistics
        ar_mae = np.mean(np.abs(arousal_errors))
        ar_rmse = np.sqrt(np.mean(arousal_errors**2))
        ax_ar_error.text(0.02, 0.98, f'MAE: {ar_mae:.3f}\nRMSE: {ar_rmse:.3f}', 
                        transform=ax_ar_error.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 4. Valence correlation
        ax_val_corr = fig.add_subplot(gs[2, 0])
        ax_val_corr.scatter(y_test[:, 0], valence_pred, alpha=0.6, color='blue', s=50)
        
        # Regression line
        z = np.polyfit(y_test[:, 0], valence_pred, 1)
        p = np.poly1d(z)
        ax_val_corr.plot(y_test[:, 0], p(y_test[:, 0]), "r--", alpha=0.8)
        
        # Perfect line
        min_val, max_val = y_test[:, 0].min(), y_test[:, 0].max()
        ax_val_corr.plot([min_val, max_val], [min_val, max_val], 'g--', alpha=0.8, label='Perfect')
        
        corr_val, _ = pearsonr(y_test[:, 0], valence_pred)
        ax_val_corr.set_xlabel('True Valence')
        ax_val_corr.set_ylabel('Predicted Valence')
        ax_val_corr.set_title(f'Valence Correlation\nr = {corr_val:.3f}')
        ax_val_corr.grid(True, alpha=0.3)
        ax_val_corr.legend()
        
        # 5. Arousal correlation
        ax_ar_corr = fig.add_subplot(gs[2, 1])
        ax_ar_corr.scatter(y_test[:, 1], arousal_pred, alpha=0.6, color='green', s=50)
        
        # Regression line
        z = np.polyfit(y_test[:, 1], arousal_pred, 1)
        p = np.poly1d(z)
        ax_ar_corr.plot(y_test[:, 1], p(y_test[:, 1]), "r--", alpha=0.8)
        
        # Perfect line
        min_ar, max_ar = y_test[:, 1].min(), y_test[:, 1].max()
        ax_ar_corr.plot([min_ar, max_ar], [min_ar, max_ar], 'g--', alpha=0.8, label='Perfect')
        
        corr_ar, _ = pearsonr(y_test[:, 1], arousal_pred)
        ax_ar_corr.set_xlabel('True Arousal')
        ax_ar_corr.set_ylabel('Predicted Arousal')
        ax_ar_corr.set_title(f'Arousal Correlation\nr = {corr_ar:.3f}')
        ax_ar_corr.grid(True, alpha=0.3)
        ax_ar_corr.legend()
        
        # 6. Classification accuracy by confidence
        ax_conf = fig.add_subplot(gs[2, 2])
        emotion_confidence = np.max(self.optimization_results['emotion_proba'], axis=1)
        correct_mask = y_cat_test == emotion_pred
        
        # Binned accuracy by confidence
        conf_bins = np.linspace(0.5, 1.0, 6)
        bin_centers = []
        bin_accuracies = []
        
        for i in range(len(conf_bins)-1):
            mask = (emotion_confidence >= conf_bins[i]) & (emotion_confidence < conf_bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(correct_mask[mask])
                bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                bin_accuracies.append(bin_accuracy)
        
        if bin_centers:
            ax_conf.plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=8)
            ax_conf.set_xlabel('Prediction Confidence')
            ax_conf.set_ylabel('Accuracy')
            ax_conf.set_title('Accuracy vs Confidence')
            ax_conf.grid(True, alpha=0.3)
            ax_conf.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        valence_arousal_path = os.path.join(self.plots_dir, 'valence_arousal_space_detailed.png')
        plt.savefig(valence_arousal_path, dpi=300, bbox_inches='tight')
        print(f"Valence-Arousal space plot saved: {valence_arousal_path}")
        plt.show()
    
    def _perform_cross_validation(self, X, y_cat, cv_folds=5):
        """Cross-validation to evaluate robustness"""
        print(f"\nCross-validation ({cv_folds} folds)")
        
        # Create simpler model for CV (faster)
        def create_simple_model():
            inputs = layers.Input(shape=(X.shape[1],))
            x = layers.Dense(128, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(2, activation='softmax')(x)
            
            model = keras.Model(inputs, outputs)
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            return model
        
        # Manual cross-validation with Keras
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_cat)):
            print(f"Fold {fold + 1}/{cv_folds}...")
            
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y_cat[train_idx], y_cat[val_idx]
            
            model_cv = create_simple_model()
            
            # Quick training
            model_cv.fit(
                X_train_cv, y_train_cv,
                validation_data=(X_val_cv, y_val_cv),
                epochs=30,
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )
            
            # Evaluation
            _, accuracy = model_cv.evaluate(X_val_cv, y_val_cv, verbose=0)
            cv_scores.append(accuracy)
            
            # Free memory
            del model_cv
            keras.backend.clear_session()
        
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        
        print(f"CV Results: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   Scores per fold: {[f'{score:.4f}' for score in cv_scores]}")
        
        return {'scores': cv_scores, 'mean': cv_mean, 'std': cv_std}
    
    def _create_optimization_plots(self, y_test, y_cat_test, valence_pred, arousal_pred, 
                                 emotion_pred, history, final_accuracy):
        """Create all optimization visualizations"""
        
        # 1. Training history
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Loss curves
        epochs_range = range(1, len(history.history['loss']) + 1)
        ax1.plot(epochs_range, history.history['loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs_range, history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Loss Evolution', fontsize=14)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs_range, history.history['emotion_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(epochs_range, history.history['val_emotion_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        ax2.axhline(y=final_accuracy, color='green', linestyle='--', 
                   label=f'Final: {final_accuracy:.3f}', linewidth=2)
        ax2.set_title('Accuracy Evolution', fontsize=14)
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Valence validation
        ax3.scatter(y_test[:, 0], valence_pred, alpha=0.6, s=50)
        ax3.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
        corr_val, _ = pearsonr(y_test[:, 0], valence_pred)
        ax3.set_title(f'Valence Validation (r={corr_val:.3f})', fontsize=14)
        ax3.set_xlabel('True Valence')
        ax3.set_ylabel('Predicted Valence')
        ax3.grid(True, alpha=0.3)
        
        # Confusion matrix
        cm = confusion_matrix(y_cat_test, emotion_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=self.emotion_categories,
                   yticklabels=self.emotion_categories)
        ax4.set_title(f'Optimized Confusion Matrix\nAccuracy: {final_accuracy:.3f}', fontsize=14)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        
        plt.tight_layout()
        
        training_path = os.path.join(self.plots_dir, 'optimization_training_history.png')
        plt.savefig(training_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved: {training_path}")
        plt.show()
        
        # 2. Comparison plot
        self._create_comparison_plot(y_test, y_cat_test, valence_pred, arousal_pred, emotion_pred)
    
    def _create_comparison_plot(self, y_test, y_cat_test, valence_pred, arousal_pred, emotion_pred):
        """Create before/after optimization comparison plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Prediction error distribution
        valence_errors = np.abs(y_test[:, 0] - valence_pred)
        arousal_errors = np.abs(y_test[:, 1] - arousal_pred)
        
        ax1.hist(valence_errors, bins=20, alpha=0.7, color='blue', label='Valence')
        ax1.hist(arousal_errors, bins=20, alpha=0.7, color='green', label='Arousal')
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Emotion prediction confidence
        emotion_confidence = np.max(self.optimization_results['emotion_proba'], axis=1)
        correct_mask = y_cat_test == emotion_pred
        
        ax2.hist(emotion_confidence[correct_mask], bins=15, alpha=0.7, color='green', 
                label=f'Correct ({np.sum(correct_mask)})')
        ax2.hist(emotion_confidence[~correct_mask], bins=15, alpha=0.7, color='red', 
                label=f'Incorrect ({np.sum(~correct_mask)})')
        ax2.set_xlabel('Prediction Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlations by emotional category
        pos_mask = y_cat_test == 0
        neg_mask = y_cat_test == 1
        
        if np.sum(pos_mask) > 5:
            corr_pos_val, _ = pearsonr(y_test[pos_mask, 0], valence_pred[pos_mask])
            corr_pos_ar, _ = pearsonr(y_test[pos_mask, 1], arousal_pred[pos_mask])
        else:
            corr_pos_val = corr_pos_ar = 0
            
        if np.sum(neg_mask) > 5:
            corr_neg_val, _ = pearsonr(y_test[neg_mask, 0], valence_pred[neg_mask])
            corr_neg_ar, _ = pearsonr(y_test[neg_mask, 1], arousal_pred[neg_mask])
        else:
            corr_neg_val = corr_neg_ar = 0
        
        categories = ['Positive\nValence', 'Positive\nArousal', 'Negative\nValence', 'Negative\nArousal']
        correlations = [corr_pos_val, corr_pos_ar, corr_neg_val, corr_neg_ar]
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        
        bars = ax3.bar(categories, correlations, color=colors, alpha=0.8)
        ax3.set_ylabel('Correlation')
        ax3.set_title('Correlations by Emotional Category')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Baseline vs Optimized metrics comparison
        metrics_names = ['Accuracy', 'F1-Score', 'Val Corr', 'Ar Corr']
        
        # Use previous results as baseline
        baseline_values = [0.897, 0.89, 0.816, 0.785]  # Previous results
        current_values = [
            accuracy_score(y_cat_test, emotion_pred),
            f1_score(y_cat_test, emotion_pred, average='weighted'),
            pearsonr(y_test[:, 0], valence_pred)[0],
            pearsonr(y_test[:, 1], arousal_pred)[0]
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='lightgray')
        bars2 = ax4.bar(x + width/2, current_values, width, label='Optimized', alpha=0.8, color='skyblue')
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_title('Baseline vs Optimized Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add values and improvement
        for i, (baseline, current) in enumerate(zip(baseline_values, current_values)):
            improvement = ((current - baseline) / baseline) * 100
            ax4.text(i, max(baseline, current) + 0.02, f'{improvement:+.1f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='green' if improvement > 0 else 'red')
        
        plt.tight_layout()
        
        comparison_path = os.path.join(self.plots_dir, 'optimization_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Optimization comparison saved: {comparison_path}")
        plt.show()
    
    def save_models(self):
        """Save all trained models and related files"""
        print(f"\nSaving models to {self.models_dir}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved: {scaler_path}")
        
        # Save optimized model (using new Keras format)
        if self.optimized_model is not None:
            model_path = os.path.join(self.models_dir, "optimized_model.keras")
            self.optimized_model.save(model_path)
            print(f"Optimized model saved: {model_path}")
        
        # Save metadata (convert numpy types to native Python types)
        metadata = {
            'emotion_categories': self.emotion_categories,
            'input_dimension': int(self.X.shape[1]) if hasattr(self, 'X') else None,
            'n_samples': int(len(self.X)) if hasattr(self, 'X') else None,
            'feature_names': self.feature_names,
            'selected_features_indices': [int(x) for x in self.selected_features_indices] if self.selected_features_indices is not None else None
        }
        
        import json
        metadata_path = os.path.join(self.models_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {metadata_path}")
        
        print(f"All models saved successfully in {self.models_dir}")
    
    def load_models(self):
        """Load all saved models and related files"""
        try:
            # Load scaler
            scaler_path = os.path.join(self.models_dir, "scaler.joblib")
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
            
            # Load optimized model (try both formats for compatibility)
            keras_path = os.path.join(self.models_dir, "optimized_model.keras")
            h5_path = os.path.join(self.models_dir, "optimized_model.h5")
            
            if os.path.exists(keras_path):
                self.optimized_model = keras.models.load_model(keras_path)
                print(f"Optimized model loaded from {keras_path}")
            elif os.path.exists(h5_path):
                self.optimized_model = keras.models.load_model(h5_path)
                print(f"Optimized model loaded from {h5_path}")
            else:
                print("No model file found")
            
            # Load metadata
            metadata_path = os.path.join(self.models_dir, "metadata.json")
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.emotion_categories = metadata.get('emotion_categories', self.emotion_categories)
                self.feature_names = metadata.get('feature_names', [])
                self.selected_features_indices = metadata.get('selected_features_indices', None)
            print(f"Metadata loaded from {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def generate_optimization_report(self):
        """Generate detailed optimization report"""
        if not hasattr(self, 'optimization_results'):
            print("No optimization results available. Run train_optimized_model() first")
            return
        
        results = self.optimization_results
        metrics = results['metrics']
        
        print("\n" + "="*80)
        print("COMPLETE OPTIMIZATION REPORT")
        print("="*80)
        
        print(f"\nFINAL PERFORMANCE:")
        print(f"   • Classification Accuracy: {metrics['emotion_accuracy']:.4f} ({metrics['emotion_accuracy']*100:.1f}%)")
        print(f"   • F1-Score: {metrics['emotion_f1']:.4f}")
        print(f"   • Valence Correlation: {metrics['valence_corr']:.4f}")
        print(f"   • Arousal Correlation: {metrics['arousal_corr']:.4f}")
        print(f"   • Valence RMSE: {metrics['valence_rmse']:.4f}")
        print(f"   • Arousal RMSE: {metrics['arousal_rmse']:.4f}")
        
        if 'cv_scores' in results:
            cv_results = results['cv_scores']
            print(f"\nCROSS-VALIDATION:")
            print(f"   • Mean score: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
            robustness = 'Excellent' if cv_results['std'] < 0.02 else 'Good' if cv_results['std'] < 0.05 else 'Moderate'
            print(f"   • Robustness: {robustness}")
        
        print(f"\nIDENTIFIED IMPROVEMENTS:")
        
        # Comparison with baseline
        baseline_accuracy = 0.897
        improvement = ((metrics['emotion_accuracy'] - baseline_accuracy) / baseline_accuracy) * 100
        
        if improvement > 0:
            print(f"   ✓ Accuracy improved by +{improvement:.1f}%")
        else:
            print(f"   • Accuracy: {improvement:.1f}% (optimization needed)")
            
        baseline_val_corr = 0.816
        val_improvement = ((metrics['valence_corr'] - baseline_val_corr) / baseline_val_corr) * 100
        
        if val_improvement > 0:
            print(f"   ✓ Valence correlation improved by +{val_improvement:.1f}%")
        
        print(f"\nOPTIMIZATION TECHNIQUES APPLIED:")
        print(f"   • Advanced feature selection (composite score)")
        print(f"   • Feature engineering (ratios, interactions, transformations)")
        print(f"   • Optimized architecture with attention")
        print(f"   • Adaptive regularization")
        print(f"   • Learning rate scheduling")
        print(f"   • Early stopping with accuracy monitoring")
        
        print(f"\nFILES GENERATED:")
        print(f"   • Optimized model: {self.models_dir}/optimized_model.h5")
        print(f"   • Visualizations: {self.plots_dir}/*.png")
        print(f"   • Feature selection: {self.plots_dir}/advanced_feature_selection.png")
        print(f"   • Valence-Arousal space: {self.plots_dir}/valence_arousal_space_detailed.png")


# Main optimized function
def main_optimized():
    """
    Main function for complete system optimization
    """
    print("=" * 80)
    print("ADVANCED MUSIC EMOTION OPTIMIZATION SYSTEM")
    print("=" * 80)
    print("Version with feature selection, engineering and optimized architecture")
    
    # Initialize optimized predictor
    predictor = MusicEmotionPredictorOptimized(
        dataset_path="./deam_dataset",
        models_dir="./models_optimized",
        plots_dir="./plots_results"
    )
    
    try:
        # Step 1: Prepare data
        print("\n1. Loading data...")
        X, y = predictor.prepare_data(max_samples=None)
        
        # Step 2: Complete optimized training
        print("\n2. Training optimized model...")
        results = predictor.train_optimized_model(
            epochs=80,
            use_feature_selection=True,
            use_feature_engineering=True,
            k_best=60
        )
        
        # Step 3: Generate detailed report
        print("\n3. Generating report...")
        predictor.generate_optimization_report()
        
        # Step 4: Save models
        print("\n4. Saving models...")
        predictor.save_models()
        
        print(f"\nOPTIMIZATION COMPLETED!")
        print(f"Final accuracy: {results['metrics']['emotion_accuracy']:.4f}")
        print(f"Results in: {predictor.plots_dir}")
        print(f"Models in: {predictor.models_dir}")
        
        return predictor, results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def predict_emotion_from_audio(predictor, track_id):
    """
    Predict emotion for a single audio track
    """
    if predictor.optimized_model is None:
        print("No trained model available. Train the model first.")
        return None
    
    print(f"\nAnalyzing track ID: {track_id}")
    
    # Load track features
    features = predictor.load_single_track_features(track_id)
    if features is None:
        print(f"Cannot load features for track {track_id}")
        return None
    
    # Apply feature selection if used during training
    if predictor.selected_features_indices is not None:
        features = features[predictor.selected_features_indices]
    
    # Apply feature engineering if used during training
    if predictor.feature_engineering_enabled:
        features = predictor.create_engineered_features(features.reshape(1, -1))
        features = features.flatten()
    
    # Normalize features
    features_scaled = predictor.scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    predictions = predictor.optimized_model.predict(features_scaled, verbose=0)
    valence_pred = float(predictions[0][0])
    arousal_pred = float(predictions[1][0])
    emotion_proba = predictions[2][0]
    emotion_pred = np.argmax(emotion_proba)
    emotion_confidence = float(emotion_proba[emotion_pred])
    
    # Interpret results
    emotion_label = predictor.emotion_categories[emotion_pred]
    
    result = {
        'track_id': track_id,
        'valence': valence_pred,
        'arousal': arousal_pred,
        'emotion_category': emotion_label,
        'emotion_confidence': emotion_confidence,
        'interpretation': _interpret_emotion_result(valence_pred, arousal_pred, emotion_label)
    }
    
    # Display results
    print(f"\nEMOTIONAL ANALYSIS RESULTS")
    print(f"Valence: {valence_pred:.3f} (1=very positive, -1=very negative)")
    print(f"Arousal: {arousal_pred:.3f} (1=very energetic, -1=very calm)")
    print(f"Emotion category: {emotion_label}")
    print(f"Confidence: {emotion_confidence:.3f}")
    print(f"\nInterpretation: {result['interpretation']}")
    
    return result


def _interpret_emotion_result(valence, arousal, category):
    """
    Interpret numerical values into textual description
    """
    # Valence descriptions
    if valence > 0.3:
        valence_desc = "very positive"
    elif valence > 0.1:
        valence_desc = "rather positive"
    elif valence > -0.1:
        valence_desc = "neutral"
    elif valence > -0.3:
        valence_desc = "rather negative"
    else:
        valence_desc = "very negative"
    
    # Arousal descriptions
    if arousal > 0.3:
        arousal_desc = "very energetic"
    elif arousal > 0.1:
        arousal_desc = "rather energetic"
    elif arousal > -0.1:
        arousal_desc = "moderately energetic"
    elif arousal > -0.3:
        arousal_desc = "rather calm"
    else:
        arousal_desc = "very calm"
    
    interpretation = f"This music has a {valence_desc} emotional tone with {arousal_desc} energy level. "
    interpretation += f"It belongs to the '{category}' category."
    
    return interpretation


def analyze_multiple_tracks(predictor, track_ids):
    """
    Analyze multiple tracks and create comparison with detailed progress
    """
    results = []
    
    print(f"\nANALYZING {len(track_ids)} TRACKS")
    print("=" * 60)
    
    for i, track_id in enumerate(track_ids):
        print(f"\n[{i+1}/{len(track_ids)}] Processing Track {track_id}...")
        
        result = predict_emotion_from_audio(predictor, track_id)
        if result is not None:
            results.append(result)
            # Quick summary for each track
            print(f"    ✓ {result['emotion_category']} (conf: {result['emotion_confidence']:.3f})")
            print(f"      Val: {result['valence']:+.3f}, Ar: {result['arousal']:+.3f}")
        else:
            print(f"    ✗ Analysis failed for track {track_id}")
    
    # Create comparative summary
    if results:
        print(f"\n" + "="*60)
        print("CREATING ANALYSIS SUMMARY AND VISUALIZATIONS")
        print("="*60)
        _create_comparison_summary(results, predictor.plots_dir)
    
    return results


def _create_comparison_summary(results, plots_dir):
    """
    Create comparative summary of analyses
    """
    print(f"\nCOMPARATIVE SUMMARY")
    
    # General statistics
    valences = [r['valence'] for r in results]
    arousals = [r['arousal'] for r in results]
    
    print(f"Number of tracks analyzed: {len(results)}")
    print(f"Average valence: {np.mean(valences):.3f} (std: {np.std(valences):.3f})")
    print(f"Average arousal: {np.mean(arousals):.3f} (std: {np.std(arousals):.3f})")
    
    # Category distribution
    categories = [r['emotion_category'] for r in results]
    from collections import Counter
    category_counts = Counter(categories)
    
    print(f"\nEmotion category distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(results)) * 100
        print(f"- {category}: {count} tracks ({percentage:.1f}%)")
    
    # Create visualization
    _plot_tracks_analysis(results, plots_dir)


def _plot_tracks_analysis(results, plots_dir):
    """
    Create visualizations of analysis results
    """
    if not results:
        return
    
    # Extract data
    valences = [r['valence'] for r in results]
    arousals = [r['arousal'] for r in results]
    categories = [r['emotion_category'] for r in results]
    confidences = [r['emotion_confidence'] for r in results]
    track_ids = [r.get('track_id', f'Track_{i}') for i, r in enumerate(results)]
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Valence-Arousal scatter plot
    colors = ['red' if cat == 'Negative/Depressed' else 'blue' for cat in categories]
    sizes = [conf * 200 for conf in confidences]  # Size proportional to confidence
    
    scatter = ax1.scatter(valences, arousals, c=colors, alpha=0.7, s=sizes, 
                         edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Valence', fontsize=12)
    ax1.set_ylabel('Arousal', fontsize=12)
    ax1.set_title('Valence-Arousal Distribution\n(Size = confidence)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Legend
    ax1.scatter([], [], c='blue', s=100, label='Positive/Energetic', alpha=0.7)
    ax1.scatter([], [], c='red', s=100, label='Negative/Depressed', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # 2. Valence histogram
    n_bins = min(15, len(valences) // 2) if len(valences) > 2 else 5
    n, bins, patches = ax2.hist(valences, bins=n_bins, alpha=0.7, color='blue', 
                               edgecolor='black')
    ax2.set_xlabel('Valence', fontsize=12)
    ax2.set_ylabel('Number of tracks', fontsize=12)
    ax2.set_title(f'Valence Distribution\nMean: {np.mean(valences):.3f}, Std: {np.std(valences):.3f}', fontsize=14)
    ax2.axvline(x=np.mean(valences), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(valences):.3f}')
    ax2.axvline(x=np.median(valences), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(valences):.3f}')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Arousal histogram
    n, bins, patches = ax3.hist(arousals, bins=n_bins, alpha=0.7, color='green', 
                               edgecolor='black')
    ax3.set_xlabel('Arousal', fontsize=12)
    ax3.set_ylabel('Number of tracks', fontsize=12)
    ax3.set_title(f'Arousal Distribution\nMean: {np.mean(arousals):.3f}, Std: {np.std(arousals):.3f}', fontsize=14)
    ax3.axvline(x=np.mean(arousals), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(arousals):.3f}')
    ax3.axvline(x=np.median(arousals), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(arousals):.3f}')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Category bar plot
    from collections import Counter
    category_counts = Counter(categories)
    category_names = list(category_counts.keys())
    counts = list(category_counts.values())
    
    if category_names:
        bars = ax4.bar(category_names, counts, alpha=0.7, 
                      color=['blue' if 'Positive' in cat else 'red' for cat in category_names])
        ax4.set_ylabel('Number of tracks', fontsize=12)
        ax4.set_title('Emotion Categories Distribution', fontsize=14)
        
        # Add values and percentages on bars
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    analysis_path = os.path.join(plots_dir, 'tracks_analysis_overview.png')
    plt.savefig(analysis_path, dpi=300, bbox_inches='tight')
    print(f"Tracks analysis saved: {analysis_path}")
    plt.show()


if __name__ == "__main__":
    # Run main optimization
    predictor, results = main_optimized()
    
    # Example of prediction on more tracks with detailed reporting
    if predictor is not None and len(predictor.song_ids) > 10:
        print("\n" + "="*80)
        print("COMPREHENSIVE TESTING ON MULTIPLE TRACKS")
        print("="*80)
        
        # Test on more tracks from dataset (let's try 15 tracks)
        num_test_tracks = min(15, len(predictor.song_ids))
        test_track_ids = predictor.song_ids[:num_test_tracks]
        
        print(f"\nSelected {num_test_tracks} tracks for comprehensive testing:")
        print("Track IDs being tested:")
        for i, track_id in enumerate(test_track_ids, 1):
            print(f"  {i:2d}. Track ID: {track_id}")
        
        print(f"\n" + "-"*60)
        print("DETAILED ANALYSIS RESULTS")
        print("-"*60)
        
        # Analyze all selected tracks
        analysis_results = analyze_multiple_tracks(predictor, test_track_ids)
        
        # Create enhanced summary with track details
        if analysis_results:
            print(f"\n" + "="*80)
            print("COMPREHENSIVE RESULTS SUMMARY")
            print("="*80)
            
            # Display results for each track
            print(f"\nINDIVIDUAL TRACK RESULTS:")
            print("-" * 40)
            for i, result in enumerate(analysis_results, 1):
                print(f"{i:2d}. Track {result['track_id']}:")
                print(f"    Valence: {result['valence']:6.3f} | Arousal: {result['arousal']:6.3f}")
                print(f"    Category: {result['emotion_category']} (confidence: {result['emotion_confidence']:.3f})")
                print()
            
            # Overall statistics
            valences = [r['valence'] for r in analysis_results]
            arousals = [r['arousal'] for r in analysis_results]
            confidences = [r['emotion_confidence'] for r in analysis_results]
            categories = [r['emotion_category'] for r in analysis_results]
            
            from collections import Counter
            category_counts = Counter(categories)
            
            print(f"AGGREGATE STATISTICS:")
            print(f"  Total tracks analyzed: {len(analysis_results)}")
            print(f"  Valence  - Mean: {np.mean(valences):.3f}, Std: {np.std(valences):.3f}, Range: [{min(valences):.3f}, {max(valences):.3f}]")
            print(f"  Arousal  - Mean: {np.mean(arousals):.3f}, Std: {np.std(arousals):.3f}, Range: [{min(arousals):.3f}, {max(arousals):.3f}]")
            print(f"  Confidence - Mean: {np.mean(confidences):.3f}, Std: {np.std(confidences):.3f}")
            
            print(f"\nCATEGORY DISTRIBUTION:")
            for category, count in category_counts.items():
                percentage = (count / len(analysis_results)) * 100
                print(f"  {category}: {count} tracks ({percentage:.1f}%)")
            
            # Confidence analysis by category
            pos_confidences = [r['emotion_confidence'] for r in analysis_results if r['emotion_category'] == 'Positive/Energetic']
            neg_confidences = [r['emotion_confidence'] for r in analysis_results if r['emotion_category'] == 'Negative/Depressed']
            
            if pos_confidences:
                print(f"\nCONFIDENCE BY CATEGORY:")
                print(f"  Positive/Energetic: Mean = {np.mean(pos_confidences):.3f}, Std = {np.std(pos_confidences):.3f}")
            if neg_confidences:
                print(f"  Negative/Depressed: Mean = {np.mean(neg_confidences):.3f}, Std = {np.std(neg_confidences):.3f}")
            
            # Find extreme cases
            print(f"\nEXTREME CASES:")
            
            # Most positive
            most_positive = max(analysis_results, key=lambda x: x['valence'])
            print(f"  Most Positive: Track {most_positive['track_id']} (Valence: {most_positive['valence']:.3f})")
            
            # Most negative  
            most_negative = min(analysis_results, key=lambda x: x['valence'])
            print(f"  Most Negative: Track {most_negative['track_id']} (Valence: {most_negative['valence']:.3f})")
            
            # Most energetic
            most_energetic = max(analysis_results, key=lambda x: x['arousal'])
            print(f"  Most Energetic: Track {most_energetic['track_id']} (Arousal: {most_energetic['arousal']:.3f})")
            
            # Most calm
            most_calm = min(analysis_results, key=lambda x: x['arousal'])
            print(f"  Most Calm: Track {most_calm['track_id']} (Arousal: {most_calm['arousal']:.3f})")
            
            # Highest confidence
            highest_conf = max(analysis_results, key=lambda x: x['emotion_confidence'])
            print(f"  Highest Confidence: Track {highest_conf['track_id']} ({highest_conf['emotion_category']}, {highest_conf['emotion_confidence']:.3f})")
            
            # Lowest confidence
            lowest_conf = min(analysis_results, key=lambda x: x['emotion_confidence'])
            print(f"  Lowest Confidence: Track {lowest_conf['track_id']} ({lowest_conf['emotion_category']}, {lowest_conf['emotion_confidence']:.3f})")
            
        print(f"\n" + "="*80)
        print("TESTING COMPLETED")
        print("="*80)
        print("Files generated:")
        print("  • Detailed visualizations: plots_results/")
        print("  • Track analysis overview: plots_results/tracks_analysis_overview.png")
        print("  • Saved models: models_optimized/")
        print("  • All individual track analyses shown above")
        
    else:
        print(f"\nInsufficient tracks available for comprehensive testing.")
        print(f"Available tracks: {len(predictor.song_ids) if predictor else 0}")


def test_specific_tracks(predictor, track_ids_list):
    """
    Test specific tracks by ID with detailed reporting
    
    Args:
        predictor: Trained MusicEmotionPredictorOptimized instance
        track_ids_list: List of specific track IDs to test
    """
    print(f"\n" + "="*80)
    print("TESTING SPECIFIC TRACKS")
    print("="*80)
    
    print(f"Testing {len(track_ids_list)} specific tracks:")
    for i, track_id in enumerate(track_ids_list, 1):
        print(f"  {i}. Track ID: {track_id}")
    
    results = []
    successful_analyses = 0
    
    for i, track_id in enumerate(track_ids_list, 1):
        print(f"\n[{i}/{len(track_ids_list)}] Analyzing Track {track_id}")
        print("-" * 50)
        
        result = predict_emotion_from_audio(predictor, track_id)
        if result is not None:
            results.append(result)
            successful_analyses += 1
            
            # Display formatted result
            print(f"✓ RESULTS FOR TRACK {track_id}:")
            print(f"    Emotional Category: {result['emotion_category']}")
            print(f"    Confidence Level: {result['emotion_confidence']:.1%}")
            print(f"    Valence Score: {result['valence']:+.3f} ({'Positive' if result['valence'] > 0 else 'Negative'})")
            print(f"    Arousal Score: {result['arousal']:+.3f} ({'High Energy' if result['arousal'] > 0 else 'Low Energy'})")
            print(f"    Interpretation: {result['interpretation']}")
            
        else:
            print(f"✗ FAILED to analyze Track {track_id}")
    
    print(f"\n" + "="*60)
    print(f"SPECIFIC TRACKS TESTING SUMMARY")
    print("="*60)
    print(f"Total tracks requested: {len(track_ids_list)}")
    print(f"Successfully analyzed: {successful_analyses}")
    print(f"Failed analyses: {len(track_ids_list) - successful_analyses}")
    
    if results:
        # Create comparison visualization
        _create_comparison_summary(results, predictor.plots_dir)
        
        # Save detailed results to file
        import json
        results_file = os.path.join(predictor.plots_dir, 'specific_tracks_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {results_file}")
    
    return results