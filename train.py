import os
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.optim as optim
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import transformers
transformers.logging.set_verbosity_error()
import numpy as np
from tqdm import tqdm
import json
import time
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from bs4 import BeautifulSoup
import re
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Original DeepFW utility functions
from utils.data_utils import load_pretrained_glove_model, preprocess_text_for_html, get_web_files, filter_empty_files
from utils.data_utils import TextWithLabelDataset, collate_fn
from models.FFAN import FFANModel
from models.classifier import Classifier
from losses.HCTCL import HCTCLoss

# ======================== DOM-LMSS Core Components ========================

def extract_dom_sequence(html_content, max_length=500):
    """
    Extract DOM tag sequence (pre-order traversal) with max length limit
    :param html_content: HTML content
    :param max_length: Maximum sequence length (truncate if needed)
    :return: DOM tag sequence
    """
    try:
        # Fix common HTML issues
        html_content = re.sub(r'<(!DOCTYPE|html|head|body|meta|link|script|style|title)[^>]*?>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<\?xml[^>]*?>', '', html_content)
        html_content = re.sub(r'<!--.*?-->', '', html_content, flags=re.DOTALL)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        sequence = []
        
        def traverse(node):
            if node.name and len(sequence) < max_length:
                # Keep only tag name, remove attributes and content
                sequence.append(node.name.lower())  # Convert to lowercase
                for child in node.children:
                    if child.name:
                        traverse(child)
        
        if soup.html:
            traverse(soup.html)
        else:
            # Handle case without html tag
            traverse(soup)
        
        return sequence[:max_length]  # Ensure doesn't exceed max length
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return []  # Return empty sequence on failure

def lmss_similarity(seqA, seqB):
    """
    Calculate Longest Matching Subsequence Similarity (LMSS)
    :param seqA: Sequence A
    :param seqB: Sequence B
    :return: Normalized similarity score
    """
    if not seqA or not seqB:
        return 0.0
    
    m, n = len(seqA), len(seqB)
    # Optimization: Use smaller data type to save memory
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seqA[i - 1] == seqB[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    lcs_length = dp[m][n]
    return lcs_length / max(m, n)  # Normalized similarity

class DOMFingerprintDatabase:
    def __init__(self):
        self.fingerprints = {}  # {version: [dom_sequence1, ...]}
        self.version_files = {}  # {version: [file_name1, ...]}
    
    def build(self, train_texts, train_labels, train_fnames, label_encoder):
        """Build version fingerprint database"""
        self.label_encoder = label_encoder
        for text, label, fname in zip(train_texts, train_labels, train_fnames):
            # Decode label
            version_name = label_encoder.inverse_transform([label])[0]
            dom_seq = extract_dom_sequence(text)
            
            if version_name not in self.fingerprints:
                self.fingerprints[version_name] = []
                self.version_files[version_name] = []
            
            self.fingerprints[version_name].append(dom_seq)
            self.version_files[version_name].append(fname)
    
    def predict_version(self, test_seq):
        """Predict version for test sample"""
        max_similarity = -1
        predicted_version = None
        
        for version, seq_list in self.fingerprints.items():
            for train_seq in seq_list:
                sim = lmss_similarity(test_seq, train_seq)
                if sim > max_similarity:
                    max_similarity = sim
                    predicted_version = version
        
        return predicted_version
    
    def parallel_predict(self, test_sequences):
        """Parallel prediction for multiple test sequences"""
        with Pool(processes=max(1, cpu_count() // 2)) as pool:
            results = pool.map(self.predict_version, test_sequences)
        return results

# ======================== DeepFW Model Components ========================

def train_model_with_early_stopping(train_texts, train_labels, train_fnames, val_texts, val_labels, val_fnames, 
                                   model, device, tokenizer, max_epochs=150, 
                                   learning_rate=0.0001, margin=1.0, lambda_reg=0.01, 
                                   patience=5, min_delta=0.001):
    """
    Training function with early stopping (optimized for GPU memory usage)
    """
    a = 1.0
    b = 0.1
    
    train_batch_size = 32
    val_batch_size = 32
    
    print(f"Using batch sizes: Train={train_batch_size}, Val={val_batch_size}")
    
    # Create train and validation datasets
    train_dataset = TextWithLabelDataset(train_texts, train_labels, tokenizer, model, train_fnames)
    val_dataset = TextWithLabelDataset(val_texts, val_labels, tokenizer, model, val_fnames)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=collate_fn, shuffle=False)
    
    ffan_model = FFANModel(train_dataset.embedding_matrix).to(device)
    classifier = Classifier(input_dim=ffan_model.hidden_dim * 3, num_classes=len(set(train_labels))).to(device)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_hctc = HCTCLoss(margin=margin, lambda_reg=lambda_reg, 
                             num_classes=len(set(train_labels)), 
                             encoding_dim=ffan_model.hidden_dim * 3, device=device)
    optimizer = optim.Adam(list(ffan_model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = {
        'ffan_model': None,
        'classifier': None,
        'epoch': 0
    }
    
    # Training history
    train_history = []
    val_history = []
    
    for epoch in range(max_epochs):
        start_time = time.time()
        ffan_model.train()
        classifier.train()
        total_train_loss = 0.0
        total_samples = 0
        
        # Training phase
        for sequences, labels_batch, file_names_batch in train_dataloader:
            sequences, labels_batch = sequences.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            
            # Batch processing for feature calculation
            features = ffan_model(sequences)
            logits = classifier(features)
            
            # Optimized negative sample search
            negatives = []
            for i, file_name in enumerate(file_names_batch):
                # First try to find same filename with different label
                negative_found = False
                for j, other_file_name in enumerate(file_names_batch):
                    if i != j and os.path.basename(file_name) == os.path.basename(other_file_name) and labels_batch[i] != labels_batch[j]:
                        negatives.append(features[j].detach().clone())
                        negative_found = True
                        break
                
                # If not found, randomly select a different class sample
                if not negative_found:
                    diff_class_indices = [idx for idx, lbl in enumerate(labels_batch) if lbl != labels_batch[i]]
                    if diff_class_indices:
                        random_idx = random.choice(diff_class_indices)
                        negatives.append(features[random_idx].detach().clone())
                    else:
                        negatives.append(torch.zeros_like(features[0]).to(device))
            
            negatives = torch.stack(negatives)
            
            # Calculate losses
            loss_cls = criterion_cls(logits, labels_batch)
            loss_hctc = criterion_hctc(features, negatives, labels_batch)
            loss = a * loss_cls + b * loss_hctc
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * sequences.size(0)
            total_samples += sequences.size(0)
            
            # Clean intermediate variables to free memory
            del sequences, labels_batch, features, logits, negatives
            torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / total_samples
        train_history.append(avg_train_loss)
        
        # Validation phase
        ffan_model.eval()
        classifier.eval()
        total_val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for sequences, labels_batch, _ in val_dataloader:
                sequences, labels_batch = sequences.to(device), labels_batch.to(device)
                features = ffan_model(sequences)
                logits = classifier(features)
                
                # Only classification loss needed for validation
                loss_cls = criterion_cls(logits, labels_batch)
                total_val_loss += loss_cls.item() * sequences.size(0)
                val_samples += sequences.size(0)
                
                del sequences, labels_batch, features, logits
                torch.cuda.empty_cache()
        
        avg_val_loss = total_val_loss / val_samples
        val_history.append(avg_val_loss)
        
        epoch_time = time.time() - start_time
        
        # Print training progress
        print(f'Epoch {epoch + 1}/{max_epochs}, Time: {epoch_time:.2f}s, '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save best model state
            best_model_state = {
                'ffan_model': ffan_model.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch
            }
            print(f'Validation loss improved to {best_val_loss:.4f}, saving model...')
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve. Patience: {epochs_no_improve}/{patience}')
            
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs!')
                break
    
    # Load best model state
    if best_model_state['ffan_model'] is not None:
        ffan_model.load_state_dict(best_model_state['ffan_model'])
        classifier.load_state_dict(best_model_state['classifier'])
        print(f'Training complete. Best model at epoch {best_model_state["epoch"] + 1}')
    else:
        print('Training complete without saving any model (no improvement)')
    
    return ffan_model, classifier, train_history, val_history

def evaluate_deepfw_model(ffan_model, classifier, texts, labels, file_names, device, glove_model, tokenizer, label_encoder):
    """Evaluate DeepFW model"""
    batch_size = 32
    
    dataset = TextWithLabelDataset(texts, labels, tokenizer, glove_model, file_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    ffan_model.eval()
    classifier.eval()
    all_preds = []
    all_labels = []
    total = 0
    correct = 0
    
    with torch.no_grad():
        for sequences, labels_batch, _ in dataloader:
            sequences, labels_batch = sequences.to(device), labels_batch.to(device)
            features = ffan_model(sequences)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            # Collect predictions and true labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
            del sequences, labels_batch, features, outputs
            torch.cuda.empty_cache()
    
    # Calculate evaluation metrics
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_dom_lmss_model(train_texts, train_labels, train_fnames, 
                           test_texts, test_labels, test_fnames, 
                           label_encoder):
    """Evaluate DOM-LMSS model"""
    # Build fingerprint database
    print("Building DOM fingerprint database...")
    fingerprint_db = DOMFingerprintDatabase()
    fingerprint_db.build(train_texts, train_labels, train_fnames, label_encoder)
    
    # Extract test DOM sequences (parallel processing)
    print("Extracting test DOM sequences...")
    test_sequences = []
    with Pool(processes=max(1, cpu_count() // 2)) as pool:
        test_sequences = list(tqdm(pool.imap(extract_dom_sequence, test_texts), total=len(test_texts)))
    
    # Parallel prediction
    print("Predicting versions with LMSS...")
    pred_versions = fingerprint_db.parallel_predict(test_sequences)
    
    # Decode true labels
    true_versions = label_encoder.inverse_transform(test_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(true_versions, pred_versions)
    precision = precision_score(true_versions, pred_versions, average='weighted', zero_division=0)
    recall = recall_score(true_versions, pred_versions, average='weighted', zero_division=0)
    f1 = f1_score(true_versions, pred_versions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ======================== Utility Functions ========================

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("Using CPU, no GPU memory to monitor")

def clear_memory():
    """Actively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared")

def load_and_preprocess_data(root_dir, selected_categories):
    """Load and preprocess data"""
    all_texts = []
    all_labels = []
    all_file_names = []
    
    # Collect data for selected categories
    for category in selected_categories:
        category_path = os.path.join(root_dir, category)
        files = get_web_files(category_path)
        files = filter_empty_files(files)
        
        for file in files:
            try:
                with open(file, 'r', encoding='iso-8859-1') as f:
                    text = preprocess_text_for_html(f.read())
                    all_texts.append(text)
                    all_file_names.append(file)
                    all_labels.append(category)
            except Exception as e:
                print(f"Error reading file {file}: {e}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    
    return all_texts, encoded_labels, all_file_names, label_encoder

def split_data(texts, labels, file_names):
    """Split dataset"""
    # Split into train, validation, and test sets (60% train, 20% val, 20% test)
    train_val_texts, test_texts, train_val_labels, test_labels, train_val_fnames, test_fnames = train_test_split(
        texts, labels, file_names, 
        test_size=0.2, random_state=42, stratify=labels
    )
    
    train_texts, val_texts, train_labels, val_labels, train_fnames, val_fnames = train_test_split(
        train_val_texts, train_val_labels, train_val_fnames, 
        test_size=0.25, random_state=42, stratify=train_val_labels
    )
    
    print(f"Training samples: {len(train_texts)}, "
          f"Validation samples: {len(val_texts)}, "
          f"Test samples: {len(test_texts)}")
    
    return {
        'train': (train_texts, train_labels, train_fnames),
        'val': (val_texts, val_labels, val_fnames),
        'test': (test_texts, test_labels, test_fnames)
    }

# ======================== Main Experiment Function ========================

def run_experiment(root_dir, pool_size, exp_idx, method, glove_model, bert_tokenizer, device):
    """Run a single experiment"""
    print(f"\n{'='*50}")
    print(f"Experiment {exp_idx+1} for pool size {pool_size} using {method}")
    print(f"{'='*50}")
    
    # Randomly select pool_size categories
    all_categories = [d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]
    selected_categories = random.sample(all_categories, pool_size)
    print(f"Selected categories: {selected_categories}")
    
    # Load and preprocess data
    texts, labels, file_names, label_encoder = load_and_preprocess_data(root_dir, selected_categories)
    
    # Split data
    data_splits = split_data(texts, labels, file_names)
    train_texts, train_labels, train_fnames = data_splits['train']
    val_texts, val_labels, val_fnames = data_splits['val']
    test_texts, test_labels, test_fnames = data_splits['test']
    
    # Execute experiment based on method
    if method == 'deepfw':
        # Train DeepFW model
        ffan_model, classifier, train_hist, val_hist = train_model_with_early_stopping(
            train_texts, train_labels, train_fnames,
            val_texts, val_labels, val_fnames,
            glove_model, device, bert_tokenizer,
            max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta
        )
        
        # Evaluate model
        test_metrics = evaluate_deepfw_model(
            ffan_model, classifier, 
            test_texts, test_labels, test_fnames, 
            device, glove_model, bert_tokenizer,
            label_encoder
        )
        
        # Free memory
        del ffan_model, classifier
        clear_memory()
        
        return {
            'method': 'deepfw',
            'pool_size': pool_size,
            'experiment': exp_idx,
            'categories': selected_categories,
            'test_metrics': test_metrics,
            'epochs_used': len(train_hist),
            'num_samples': {
                'total': len(texts),
                'train': len(train_texts),
                'val': len(val_texts),
                'test': len(test_texts)
            }
        }
    
    elif method == 'dom_lmss':
        # Evaluate DOM-LMSS model
        test_metrics = evaluate_dom_lmss_model(
            train_texts, train_labels, train_fnames,
            test_texts, test_labels, test_fnames,
            label_encoder
        )
        
        return {
            'method': 'dom_lmss',
            'pool_size': pool_size,
            'experiment': exp_idx,
            'categories': selected_categories,
            'test_metrics': test_metrics,
            'epochs_used': 0,  # No training process
            'num_samples': {
                'total': len(texts),
                'train': len(train_texts),
                'test': len(test_texts)
            }
        }
    
    else:
        raise ValueError(f"Unknown method: {method}")

# ======================== Main Function ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Firmware Version Identification Experiment')
    parser.add_argument('--root_dir', type=str, default='/mnt/data/leizhen/FirmID-Web/Firm-ID-dataset',
                        help='Root directory of dataset')
    parser.add_argument('--glove_path', type=str, default='Static_word_embedding_pre-trained_models/glove.6B.300d.txt',
                        help='Path to GloVe pretrained model')
    parser.add_argument('--bert_path', type=str, default='/mnt/data/leizhen/FirmID-Web/Firm-ID/scripts/bert-base-uncase',
                        help='Path to BERT model')
    parser.add_argument('--method', type=str, choices=['deepfw', 'dom_lmss', 'both'], default='both',
                        help='Method to run: deepfw, dom_lmss, or both')
    parser.add_argument('--pool_sizes', type=int, nargs='+', default=[2],
                        help='List of pool sizes to experiment with')
    parser.add_argument('--num_experiments', type=int, default=3,
                        help='Number of experiments per pool size')
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0001,
                        help='Minimum delta for early stopping')
    parser.add_argument('--output_dir', type=str, default='experiment_results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pretrained models
    glove_model = load_pretrained_glove_model(args.glove_path)
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine methods to run
    methods = []
    if args.method == 'both':
        methods = ['deepfw', 'dom_lmss']
    else:
        methods = [args.method]
    
    # Store experiment results
    all_results = {method: [] for method in methods}
    summary_results = []
    
    # Record start time
    start_time = time.time()
    
    # Run experiments
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Starting experiments for method: {method.upper()}")
        print(f"{'='*50}")
        
        for pool_size in args.pool_sizes:
            print(f"\nPool size: {pool_size}")
            pool_results = []
            
            for exp_idx in range(args.num_experiments):
                result = run_experiment(
                    root_dir=args.root_dir,
                    pool_size=pool_size,
                    exp_idx=exp_idx,
                    method=method,
                    glove_model=glove_model,
                    bert_tokenizer=bert_tokenizer,
                    device=device
                )
                
                # Save experiment result
                pool_results.append(result)
                all_results[method].append(result)
                
                # Save to file
                result_file = os.path.join(args.output_dir, 
                                         f"{method}_pool{pool_size}_exp{exp_idx}.json")
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=4)
                
                print(f"Completed experiment {exp_idx+1}/{args.num_experiments} for {method} (pool size {pool_size})")
                print(f"Test Accuracy: {result['test_metrics']['accuracy']:.4f}")
            
            # Calculate statistics for current pool size
            accuracies = [res['test_metrics']['accuracy'] for res in pool_results]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            f1_scores = [res['test_metrics']['f1'] for res in pool_results]
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            
            summary_results.append({
                'method': method,
                'pool_size': pool_size,
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'mean_f1': mean_f1,
                'std_f1': std_f1,
                'num_experiments': args.num_experiments
            })
            
            print(f"\nSummary for {method} (pool size {pool_size}):")
            print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Save final results
    final_result_file = os.path.join(args.output_dir, 'final_results.json')
    with open(final_result_file, 'w') as f:
        json.dump({
            'summary': summary_results,
            'all_results': all_results,
            'total_time': f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        }, f, indent=4)
    
    # Print final summary
    print("\n\nFinal Results Summary:")
    print("Method\t\tPool Size\tMean Accuracy\tStd Accuracy\tMean F1\t\tStd F1")
    for res in summary_results:
        print(f"{res['method']}\t{res['pool_size']}\t\t{res['mean_accuracy']:.4f}\t\t{res['std_accuracy']:.4f}\t\t"
              f"{res['mean_f1']:.4f}\t\t{res['std_f1']:.4f}")
    
    print(f"\nTotal execution time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    print(f"All results saved to: {args.output_dir}")