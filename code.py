
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WSNAnomalyDetector:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.class_names = ['normal', 'Blackhole', 'Forwarding', 'Flooding']
        self.results = {}
        self.best_model = None

    def load_and_preprocess_data(self, file_path, test_size=0.2):
        """Load and preprocess the entire WSN dataset and split into train/test"""
        print("Loading and preprocessing entire dataset...")

        # Load the data
        df = pd.read_csv(file_path)
        
        print(f"Total samples in dataset: {len(df):,}")

        # Display initial class distribution
        print("\nClass Distribution:")
        class_dist = df['Class'].value_counts()
        print(class_dist)

        # Handle NaN values in Class column
        nan_count = df['Class'].isna().sum()
        if nan_count > 0:
            print(f"\nFound {nan_count} NaN values in Class column. Removing them...")
            df = df.dropna(subset=['Class'])
            print(f"Remaining samples after NaN removal: {len(df):,}")

        # Select key features for efficient processing
        feature_columns = ['Time', 'S_Node', 'Node_id', 'Rest_Energy', 'Packet_Size', 'TTL', 'Hop_Count']

        # Use only available features
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Using features: {available_features}")

        X = df[available_features]
        y = df['Class']

        # Handle missing values in features
        X = X.fillna(X.mean())

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split the data
        print(f"\nSplitting data: {(1-test_size)*100}% training, {test_size*100}% testing")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set shape: {X_train_scaled.shape} ({X_train_scaled.shape[0]:,} samples)")
        print(f"Test set shape: {X_test_scaled.shape} ({X_test_scaled.shape[0]:,} samples)")

        return X_train_scaled, X_test_scaled, y_train, y_test, X, y_encoded

    def train_models(self, X_train, y_train):
        """Train multiple models including Stacking Ensemble"""
        print("\nTraining models...")

        # Define base models
        base_models = [
            ('xgb', XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                n_jobs=-1
            )),
            ('rf', RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                n_jobs=-1
            )),
            ('lr', LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            ))
        ]

        # Stacking classifier
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )

        # Train all models including stacking ensemble
        model_names = ['XGBoost', 'Random Forest', 'Logistic Regression', 'Stacking Ensemble']
        models_list = [
            base_models[0][1], base_models[1][1], base_models[2][1], stacking_model
        ]

        training_times = {}

        for name, model in zip(model_names, models_list):
            print(f"Training {name}...")
            start_time = time.time()

            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                training_times[name] = training_time
                self.models[name] = model
                print(f"  ✓ {name} trained in {training_time:.2f}s")
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")

        return training_times

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models with AUC-ROC"""
        print("\nEvaluating models...")

        results = {}

        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            start_time = time.time()

            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calculate AUC-ROC (multi-class)
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                n_classes = y_test_bin.shape[1]
                
                # Calculate ROC AUC for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Calculate macro-average ROC AUC
                roc_auc_macro = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

                # Store results
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'accuracy': accuracy,
                    'roc_auc_macro': roc_auc_macro,
                    'roc_auc_per_class': roc_auc,
                    'fpr': fpr,
                    'tpr': tpr,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(y_test, y_pred)
                }

                eval_time = time.time() - start_time
                print(f"  ✓ Accuracy: {accuracy:.4f}, AUC-ROC: {roc_auc_macro:.4f} (eval time: {eval_time:.2f}s)")

            except Exception as e:
                print(f"  ✗ Error evaluating {name}: {e}")

        self.results = results
        
        # Find best model
        if results:
            self.best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            best_auc_model = max(results.items(), key=lambda x: x[1]['roc_auc_macro'])
            print(f"\n Best Model (Accuracy): {self.best_model[0]} (Accuracy: {self.best_model[1]['accuracy']:.4f})")
            print(f" Best Model (AUC-ROC): {best_auc_model[0]} (AUC-ROC: {best_auc_model[1]['roc_auc_macro']:.4f})")
        
        return results

    def plot_roc_curves(self):
        """Plot ROC curves for all models and all classes"""
        if not self.results:
            print("No results to plot ROC curves")
            return
            
        print("\nPlotting ROC curves...")
        
        n_models = len(self.models)
        n_classes = len(self.label_encoder.classes_)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        # Plot ROC for each class
        for class_idx in range(n_classes):
            ax = axes[class_idx]
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            
            for idx, (model_name, result) in enumerate(self.results.items()):
                fpr = result['fpr'][class_idx]
                tpr = result['tpr'][class_idx]
                roc_auc = result['roc_auc_per_class'][class_idx]
                
                ax.plot(fpr, tpr, color=colors[idx % len(colors)], 
                       lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {class_name}')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots if any
        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('roc_curves_per_class.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot macro-average ROC for all models
        plt.figure(figsize=(10, 8))
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            # Compute macro-average ROC curve
            all_fpr = np.unique(np.concatenate([result['fpr'][i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, result['fpr'][i], result['tpr'][i])
            
            mean_tpr /= n_classes
            
            roc_auc_macro = result['roc_auc_macro']
            
            plt.plot(all_fpr, mean_tpr, color=colors[idx % len(colors)], 
                    lw=3, label=f'{model_name} (Macro AUC = {roc_auc_macro:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Macro-Average ROC Curves - All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('macro_roc_curves.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_auc_comparison(self):
        """Plot AUC comparison across models"""
        if not self.results:
            print("No results to plot AUC comparison")
            return
            
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['roc_auc_macro'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, auc_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.title('Model AUC-ROC Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Macro AUC-ROC Score', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, auc_score in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{auc_score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('auc_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        if not self.results:
            print("No results to plot")
            return
            
        print("\nPlotting confusion matrices...")

        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        available_classes = self.label_encoder.classes_

        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=available_classes,
                       yticklabels=available_classes,
                       ax=axes[idx])

            axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}\nAUC: {result["roc_auc_macro"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        # Hide empty subplots if any
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig('confusion_matrices_all.png', dpi=150, bbox_inches='tight')
        plt.show()

    def plot_stacking_ensemble_confusion(self):
        """Plot detailed confusion matrix specifically for Stacking Ensemble"""
        if 'Stacking Ensemble' not in self.results:
            print("Stacking Ensemble results not available")
            return
            
        print("\nCreating Stacking Ensemble confusion matrix...")
        
        ensemble_result = self.results['Stacking Ensemble']
        cm = ensemble_result['confusion_matrix']
        
        # Get available class names
        available_classes = self.label_encoder.classes_
        
        plt.figure(figsize=(12, 10))
        
        # Create the heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=available_classes, 
                    yticklabels=available_classes,
                    cbar_kws={'label': 'Number of Predictions'},
                    annot_kws={'size': 12, 'weight': 'bold'})
        
        # Customize the plot
        plt.title('Stacking Ensemble - Confusion Matrix\n', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Add performance metrics to the plot
        accuracy = ensemble_result['accuracy']
        auc_score = ensemble_result['roc_auc_macro']
        
        metrics_text = f'Accuracy: {accuracy:.4f}\nAUC-ROC: {auc_score:.4f}'
        
        plt.text(0.5, -0.15, metrics_text, 
                 transform=plt.gca().transAxes, fontsize=12, 
                 ha='center', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('stacking_ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed analysis
        print(f"\n Stacking Ensemble Detailed Analysis:")
        print("-" * 40)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        
        print(f"\nPer-class Performance:")
        print("-" * 25)
        
        for i, class_name in enumerate(available_classes):
            true_positives = cm[i, i]
            total_actual = np.sum(cm[i, :])
            accuracy_per_class = true_positives / total_actual if total_actual > 0 else 0
            
            print(f"{class_name}: {accuracy_per_class:.4f} ({true_positives}/{total_actual})")

    def plot_class_distribution(self, y_original):
        """Plot class distribution"""
        print("\nPlotting class distribution...")

        unique, counts = np.unique(y_original, return_counts=True)
        labels = [self.label_encoder.inverse_transform([cls])[0] for cls in unique]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, counts, color='lightblue', alpha=0.7)
        plt.title(f'Class Distribution\nTotal Samples: {len(y_original):,}', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()

    def print_detailed_report(self):
        """Print detailed performance report"""
        if not self.best_model:
            print("No results to analyze")
            return
            
        best_name, best_result = self.best_model
        
        print("\n" + "="*70)
        print("DETAILED PERFORMANCE REPORT WITH AUC-ROC")
        print("="*70)
        
        print(f"\n  Best Model: {best_name}")
        print(f" Accuracy: {best_result['accuracy']:.4f}")
        print(f" Macro AUC-ROC: {best_result['roc_auc_macro']:.4f}")
        print(f" Test Samples: {len(best_result['predictions']):,}")
        
        print(f"\n Per-class AUC-ROC:")
        print("-" * 40)
        for class_idx, class_name in enumerate(self.label_encoder.classes_):
            auc_score = best_result['roc_auc_per_class'][class_idx]
            print(f"{class_name}: {auc_score:.4f}")
        
        print(f"\n Classification Report:")
        print("-" * 50)
        print(classification_report(
            self.y_test_global, 
            best_result['predictions'], 
            target_names=self.label_encoder.classes_
        ))

    def save_models(self):
        """Save trained models and preprocessing objects"""
        if not self.best_model:
            print("No models to save")
            return
            
        print("\n Saving models...")

        try:
            # Save best model
            best_name, best_result = self.best_model
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(best_result['model'], f)

            # Save all models
            with open('all_models.pkl', 'wb') as f:
                pickle.dump(self.models, f)

            # Save preprocessing objects
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)

            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)

            print(" All models and preprocessing objects saved successfully!")
            print(f" Files saved: best_model.pkl, all_models.pkl, scaler.pkl, label_encoder.pkl")

        except Exception as e:
            print(f" Error saving models: {e}")

    def test_new_data(self, test_file_path):
        """Test the trained model on new data with AUC-ROC"""
        if not self.best_model:
            print("No trained model available. Please train first.")
            return
            
        print(f"\n  Testing on new data: {test_file_path}")
        
        try:
            # Load test data
            df_test = pd.read_csv(test_file_path)
            print(f" Loaded {len(df_test):,} test samples")
            
            # Preprocess test data (same features as training)
            feature_columns = ['Time', 'S_Node', 'Node_id', 'Rest_Energy', 'Packet_Size', 'TTL', 'Hop_Count']
            available_features = [col for col in feature_columns if col in df_test.columns]
            
            X_test_new = df_test[available_features]
            X_test_new = X_test_new.fillna(X_test_new.mean())
            X_test_new_scaled = self.scaler.transform(X_test_new)
            
            # Get true labels if available
            if 'Class' in df_test.columns:
                y_test_new = self.label_encoder.transform(df_test['Class'])
                has_true_labels = True
            else:
                has_true_labels = False
                print("  No 'Class' column found - performing predictions only")
            
            # Make predictions
            best_name, best_result = self.best_model
            model = best_result['model']
            
            print(f" Making predictions using {best_name}...")
            y_pred_new = model.predict(X_test_new_scaled)
            y_pred_proba_new = model.predict_proba(X_test_new_scaled)
            
            # Convert back to original labels
            y_pred_labels = self.label_encoder.inverse_transform(y_pred_new)
            
            # Calculate metrics if true labels available
            if has_true_labels:
                accuracy = accuracy_score(y_test_new, y_pred_new)
                auc_roc = roc_auc_score(y_test_new, y_pred_proba_new, multi_class='ovr', average='macro')
                
                print(f" Test Accuracy: {accuracy:.4f}")
                print(f" Test AUC-ROC: {auc_roc:.4f}")
                
                print(f"\n Classification Report:")
                print("-" * 50)
                print(classification_report(y_test_new, y_pred_new, 
                                          target_names=self.label_encoder.classes_))
                
                # Plot confusion matrix
                cm = confusion_matrix(y_test_new, y_pred_new)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=self.label_encoder.classes_,
                           yticklabels=self.label_encoder.classes_)
                plt.title(f'Test Results - {best_name}\nAccuracy: {accuracy:.4f}, AUC: {auc_roc:.4f}', 
                         fontsize=14, fontweight='bold')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
                plt.show()
                
                # Plot ROC curves for test data
                y_test_bin = label_binarize(y_test_new, classes=np.unique(y_test_new))
                n_classes = y_test_bin.shape[1]
                
                plt.figure(figsize=(10, 8))
                colors = ['blue', 'red', 'green', 'orange']
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba_new[:, i])
                    roc_auc = auc(fpr, tpr)
                    class_name = self.label_encoder.inverse_transform([i])[0]
                    plt.plot(fpr, tpr, color=colors[i % len(colors)], 
                            lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - Test Data ({best_name})\nMacro AUC: {auc_roc:.4f}')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('test_roc_curves.png', dpi=150, bbox_inches='tight')
                plt.show()
            
            # Create results DataFrame
            results_df = df_test.copy()
            results_df['Predicted_Class'] = y_pred_labels
            
            # Add probabilities for each class
            for i, class_name in enumerate(self.label_encoder.classes_):
                results_df[f'Probability_{class_name}'] = y_pred_proba_new[:, i]
            
            # Save results
            results_df.to_csv('test_predictions.csv', index=False)
            print(f" Predictions saved to 'test_predictions.csv'")
            
            # Print prediction distribution
            print(f"\n  Prediction Distribution:")
            pred_dist = results_df['Predicted_Class'].value_counts()
            for class_name, count in pred_dist.items():
                print(f"  {class_name}: {count:,} ({count/len(results_df)*100:.1f}%)")
            
            return results_df
            
        except Exception as e:
            print(f" Error testing new data: {e}")
            return None

    def run_complete_workflow(self, train_file_path, test_file_path=None, test_size=0.2):
        """Run complete training and testing workflow with AUC-ROC and Stacking Ensemble"""
        start_time = time.time()
        
        print(" WSN ANOMALY DETECTION - COMPLETE WORKFLOW")
        print("=" * 65)
        print(" Features: Stacking Ensemble + AUC-ROC Analysis")
        print("=" * 65)
        
        # Store test data for reporting
        self.X_test_global = None
        self.y_test_global = None
        
        try:
            # 1. TRAINING PHASE
            print("\n PHASE 1: TRAINING")
            print("-" * 30)
            
            X_train, X_test, y_train, y_test, X, y = self.load_and_preprocess_data(
                train_file_path, test_size=test_size
            )
            self.X_test_global = X_test
            self.y_test_global = y_test
            
            # Plot class distribution
            self.plot_class_distribution(y)
            
            # Train models (including Stacking Ensemble)
            training_times = self.train_models(X_train, y_train)
            
            if not self.models:
                raise ValueError(" No models were successfully trained")
            
            # Evaluate models
            results = self.evaluate_models(X_test, y_test)
            
            if not self.results:
                raise ValueError(" No models were successfully evaluated")
            
            # Create visualizations
            self.plot_confusion_matrices()
            self.plot_roc_curves()
            self.plot_auc_comparison()
            
            # Special Stacking Ensemble visualization
            if 'Stacking Ensemble' in self.results:
                self.plot_stacking_ensemble_confusion()
            
            # Print detailed report
            self.print_detailed_report()
            
            # Save models
            self.save_models()
            
            # 2. TESTING PHASE
            if test_file_path:
                print("\n  PHASE 2: TESTING")
                print("-" * 30)
                
                test_results = self.test_new_data(test_file_path)
            
            total_time = time.time() - start_time

            print(f"\n WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f" Total execution time: {total_time:.2f}s ({total_time/60:.2f}min)")
            
            print(f"\n  GENERATED OUTPUT FILES:")
            print("-" * 30)
            print("1. confusion_matrices_all.png - All models confusion matrices")
            print("2. stacking_ensemble_confusion_matrix.png - Stacking Ensemble detailed")
            print("3. roc_curves_per_class.png - ROC curves per class")
            print("4. macro_roc_curves.png - Macro-average ROC curves")
            print("5. auc_comparison.png - AUC comparison across models")
            print("6. class_distribution.png - Class distribution")
            print("7. best_model.pkl - Best trained model")
            print("8. all_models.pkl - All trained models")
            print("9. scaler.pkl, label_encoder.pkl - Preprocessing objects")
            
            if test_file_path:
                print("10. test_results.png - Test results")
                print("11. test_roc_curves.png - Test ROC curves")
                print("12. test_predictions.csv - Prediction results")
            
            return results
            
        except Exception as e:
            print(f" Workflow failed: {e}")
            return None

# Main execution
if __name__ == "__main__":
    # Initialize the detector
    detector = WSNAnomalyDetector()
    
    # Run complete workflow with Stacking Ensemble and AUC-ROC
    results = detector.run_complete_workflow(
        train_file_path='WSN_BFSF.csv',
        test_size=0.2  # 80% train, 20% test
    )

