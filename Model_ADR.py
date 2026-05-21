import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import os
from tqdm import tqdm
import argparse
from rdkit import Chem
import math
from collections import defaultdict
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

SMILES_CHARS = [' ', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']', 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(SMILES_CHARS)}

MAX_SMILES_LEN = 128

FINE_CLASSIFICATION_THRESHOLD = 0.5

FINE_GRAINED_CLASSES = 15


def undersample_indices(label, random_state=42):
    np.random.seed(random_state)
    pos_indices = np.where(label == 1)[0]
    neg_indices = np.where(label == 0)[0]
    min_samples = min(len(pos_indices), len(neg_indices))
    
    if len(neg_indices) > min_samples:
        neg_indices = np.random.choice(neg_indices, min_samples, replace=False)
    
    selected_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(selected_indices)
    return selected_indices


def oversample_indices(label, random_state=42):
    np.random.seed(random_state)
    pos_indices = np.where(label == 1)[0]
    neg_indices = np.where(label == 0)[0]
    max_samples = max(len(pos_indices), len(neg_indices))
    
    if len(pos_indices) < max_samples:
        num_samples_needed = max_samples - len(pos_indices)
        additional_indices = np.random.choice(pos_indices, num_samples_needed, replace=True)
        pos_indices = np.concatenate([pos_indices, additional_indices])
    
    selected_indices = np.concatenate([pos_indices, neg_indices])
    np.random.shuffle(selected_indices)
    return selected_indices


def balance_data_indices(label_coarse, label_fine_list, random_state=42):
    np.random.seed(random_state)
    indices_coarse = oversample_indices(label_coarse, random_state)
    
    if label_fine_list is not None and len(label_fine_list) > 0:
        all_fine_indices = set()
        
        for class_idx in range(label_fine_list.shape[1]):
            class_labels = label_fine_list[:, class_idx]
            class_indices = oversample_indices(class_labels, random_state)
            all_fine_indices.update(class_indices)
        
        combined_indices = np.array(list(all_fine_indices))
        return np.intersect1d(indices_coarse, combined_indices)

    return indices_coarse


class DrugDrugInteractionDataset(Dataset):
    
    def __init__(self, drug1_smiles_list, drug2_smiles_list, label_coarse, label1_str_list=None, balance_data=True,
                 random_state=42):
        self.drug1_smiles_list = drug1_smiles_list
        self.drug2_smiles_list = drug2_smiles_list
        self.label_coarse = label_coarse
        
        if label1_str_list is not None:
            self.label_fine = np.zeros((len(label1_str_list), FINE_GRAINED_CLASSES), dtype=np.float32)
            for i, label_str in enumerate(label1_str_list):
                if isinstance(label_str, str):
                    labels = [int(x) for x in label_str.split(',')]
                    if len(labels) == FINE_GRAINED_CLASSES:
                        self.label_fine[i] = np.array(labels)
        else:
            self.label_fine = None
        
        if balance_data and self.label_fine is not None:
            indices = balance_data_indices(self.label_coarse, self.label_fine, random_state)
            self.drug1_smiles_list = [self.drug1_smiles_list[i] for i in indices]
            self.drug2_smiles_list = [self.drug2_smiles_list[i] for i in indices]
            self.label_coarse = self.label_coarse[indices]
            self.label_fine = self.label_fine[indices] if self.label_fine is not None else None
        
        self.drug1_cache = {}
        self.drug2_cache = {}
    
    def __len__(self):
        return len(self.drug1_smiles_list)
    
    def __getitem__(self, idx):
        if idx in self.drug1_cache:
            drug1_input_ids, drug1_attention_mask = self.drug1_cache[idx]
        else:
            drug1_smiles = self.drug1_smiles_list[idx]
            drug1_idx_seq = [CHAR_TO_IDX.get(c, 0) for c in drug1_smiles]
            drug1_idx_seq = [CHAR_TO_IDX[' ']] + drug1_idx_seq[:MAX_SMILES_LEN - 2] + [CHAR_TO_IDX[' ']]
            drug1_input_ids = drug1_idx_seq + [0] * (MAX_SMILES_LEN - len(drug1_idx_seq))
            drug1_attention_mask = [1] * len(drug1_idx_seq) + [0] * (MAX_SMILES_LEN - len(drug1_idx_seq))
            self.drug1_cache[idx] = (drug1_input_ids, drug1_attention_mask)
        
        if idx in self.drug2_cache:
            drug2_input_ids, drug2_attention_mask = self.drug2_cache[idx]
        else:
            drug2_smiles = self.drug2_smiles_list[idx]
            drug2_idx_seq = [CHAR_TO_IDX.get(c, 0) for c in drug2_smiles]
            drug2_idx_seq = [CHAR_TO_IDX[' ']] + drug2_idx_seq[:MAX_SMILES_LEN - 2] + [CHAR_TO_IDX[' ']]
            drug2_input_ids = drug2_idx_seq + [0] * (MAX_SMILES_LEN - len(drug2_idx_seq))
            drug2_attention_mask = [1] * len(drug2_idx_seq) + [0] * (MAX_SMILES_LEN - len(drug2_idx_seq))
            self.drug2_cache[idx] = (drug2_input_ids, drug2_attention_mask)
        
        item = {
            'drug1_input_ids': torch.tensor(drug1_input_ids, dtype=torch.long),
            'drug1_attention_mask': torch.tensor(drug1_attention_mask, dtype=torch.long),
            'drug2_input_ids': torch.tensor(drug2_input_ids, dtype=torch.long),
            'drug2_attention_mask': torch.tensor(drug2_attention_mask, dtype=torch.long),
            'label_coarse': torch.tensor(self.label_coarse[idx], dtype=torch.long),
        }
        
        if self.label_fine is not None:
            item['label_fine'] = torch.tensor(self.label_fine[idx], dtype=torch.float32)
        
        return item


class DrugTransformer(nn.Module):
    
    def __init__(self, vocab_size=len(SMILES_CHARS), embed_dim=256, num_heads=8, num_layers=6, hidden_dim=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=MAX_SMILES_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True,
            dropout=0.1, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
        x = self.layer_norm(x)
        return x[:, 0, :]


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :]


def load_pretrained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if
                           k.startswith('positional_encoding.') or k.startswith('token_embedding.') or k.startswith(
                               'transformer_encoder.') or k.startswith('layer_norm.')}
    
    if 'positional_encoding.pe' in filtered_checkpoint:
        pretrained_pe = filtered_checkpoint['positional_encoding.pe']
        model_pe = model_dict['positional_encoding.pe']
        
        if pretrained_pe.shape[0] < model_pe.shape[0]:
            model_pe[:pretrained_pe.shape[0], :] = pretrained_pe
            filtered_checkpoint['positional_encoding.pe'] = model_pe
    
    model.load_state_dict(filtered_checkpoint, strict=False)
    return model


class RLGateModel(nn.Module):
    
    def __init__(self, embed_dim=256, gate_hidden=128, fine_grained_classes=FINE_GRAINED_CLASSES):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2)
        )
        
        self.shared_fine_layer = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU()
        )
        
        self.fine_output_layer = nn.Linear(gate_hidden, fine_grained_classes)
        
        self.value_network = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1)
        )
    
    def forward(self, h_drug1, h_drug2, train_mode=True, threshold=FINE_CLASSIFICATION_THRESHOLD):
        combined = torch.cat([h_drug1, h_drug2], dim=-1)
        fused = self.fusion_layer(combined)
        gate_logits = self.gate_network(fused)
        gate_probs = nn.functional.softmax(gate_logits, dim=-1)
        
        if train_mode:
            gate_action = torch.multinomial(gate_probs, 1).squeeze(-1)
        else:
            gate_action = torch.argmax(gate_probs, dim=-1)
        
        fine_logits = None
        fine_probs = None
        
        if (gate_action == 1).any() or train_mode:
            shared_features = self.shared_fine_layer(fused)
            
            fine_logits = self.fine_output_layer(shared_features)
            fine_probs = torch.sigmoid(fine_logits)
            
            if not train_mode:
                return {
                    'gate_probs': gate_probs,
                    'gate_action': gate_action,
                    'fine_logits': fine_logits,
                    'fine_probs': fine_probs,
                    'value': self.value_network(fused).squeeze(-1)
                }
        
        value = self.value_network(fused).squeeze(-1)
        return {
            'gate_probs': gate_probs,
            'gate_action': gate_action,
            'fine_logits': fine_logits,
            'fine_probs': fine_probs,
            'value': value
        }


def compute_rewards(gate_action, label_coarse, fine_probs, label_fine, pos_weight=1.0,
                    lambda_coarse=1.0, lambda_fine=0.5):
    R_coarse = torch.where(
        label_coarse == 1,
        pos_weight * (gate_action == label_coarse).float(),
        (gate_action == label_coarse).float()
    )
    
    R_fine = torch.zeros_like(R_coarse)
    valid_mask = (gate_action == 1) & (label_coarse == 1)
    
    if fine_probs is not None and valid_mask.any():
        batch_indices = torch.arange(fine_probs.size(0), device=fine_probs.device)[valid_mask]
        true_labels = label_fine[valid_mask]
        
        weighted_probs = (fine_probs[batch_indices] * true_labels).sum(dim=1) / (true_labels.sum(dim=1) + 1e-8)
        R_fine[valid_mask] = weighted_probs
    
    return R_coarse, R_fine, lambda_coarse * R_coarse + lambda_fine * R_fine


def evaluate_model(model, drug_model, data_loader, device):
    drug_model.eval()
    model.eval()

    coarse_probs = []
    coarse_true = []
    fine_preds = []
    fine_true = []
    fine_probs_all = []
    fine_class_metrics = defaultdict(lambda: {'y_true': [], 'y_pred': []})

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            h_drug1 = drug_model(batch['drug1_input_ids'].to(device), batch['drug1_attention_mask'].to(device))
            h_drug2 = drug_model(batch['drug2_input_ids'].to(device), batch['drug2_attention_mask'].to(device))
            outputs = model(h_drug1, h_drug2, train_mode=False)

            coarse_probs.extend(outputs['gate_probs'][:, 1].cpu().numpy())
            coarse_true.extend(batch['label_coarse'].cpu().numpy())

            valid_mask = (batch['label_coarse'] == 1).cpu()
            if valid_mask.any() and outputs['fine_probs'] is not None:
                batch_fine_probs = outputs['fine_probs'].cpu().numpy()[valid_mask]
                batch_fine_true = batch['label_fine'].cpu().numpy()[valid_mask]
                
                fine_probs_all.append(batch_fine_probs)
                fine_true.append(batch_fine_true)
                
                for class_idx in range(FINE_GRAINED_CLASSES):
                    class_true = batch_fine_true[:, class_idx]
                    class_pred = batch_fine_probs[:, class_idx]
                    
                    fine_class_metrics[class_idx]['y_true'].extend(class_true)
                    fine_class_metrics[class_idx]['y_pred'].extend(class_pred)

    coarse_auc = roc_auc_score(coarse_true, coarse_probs)
    coarse_aupr = average_precision_score(coarse_true, coarse_probs)
    coarse_acc = accuracy_score(coarse_true, np.array(coarse_probs) > 0.5)

    fine_auc = 0.0
    fine_aupr = 0.0
    fine_acc = 0.0
    valid_classes = 0
    class_metrics = {}
    
    for class_idx in range(FINE_GRAINED_CLASSES):
        y_true = np.array(fine_class_metrics[class_idx]['y_true'])
        y_pred = np.array(fine_class_metrics[class_idx]['y_pred'])
        
        if len(y_true) > 0 and np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
            try:
                class_auc = roc_auc_score(y_true, y_pred)
                class_aupr = average_precision_score(y_true, y_pred)
                class_acc = accuracy_score(y_true, y_pred > FINE_CLASSIFICATION_THRESHOLD)
                
                class_metrics[class_idx] = {
                    'auc': class_auc,
                    'aupr': class_aupr,
                    'accuracy': class_acc,
                    'samples': len(y_true),
                    'positives': int(np.sum(y_true))
                }
                
                fine_auc += class_auc
                fine_aupr += class_aupr
                fine_acc += class_acc
                valid_classes += 1
            except:
                pass
    
    if valid_classes > 0:
        fine_auc /= valid_classes
        fine_aupr /= valid_classes
        fine_acc /= valid_classes

    return {
        'coarse_auc': coarse_auc,
        'coarse_aupr': coarse_aupr,
        'coarse_acc': coarse_acc,
        'fine_auc': fine_auc if valid_classes > 0 else 0,
        'fine_aupr': fine_aupr if valid_classes > 0 else 0,
        'fine_acc': fine_acc if valid_classes > 0 else 0,
        'class_metrics': class_metrics
    }


def print_metrics(metrics, prefix=""):
    print(f"\n{prefix}Coarse-grained Metrics:")
    print(f'AUC: {metrics["coarse_auc"]:.4f}, AUPR: {metrics["coarse_aupr"]:.4f}, Accuracy: {metrics["coarse_acc"]:.4f}')
    
    print(f'\n{prefix}Fine-grained Metrics (Multi-label):')
    print(f'Average AUC: {metrics["fine_auc"]:.4f}, Average AUPR: {metrics["fine_aupr"]:.4f}, Average Accuracy: {metrics["fine_acc"]:.4f}')
    
    print(f'\n{prefix}Detailed Metrics per Class:')
    for class_idx, metrics in metrics["class_metrics"].items():
        print(f'Class {class_idx+1}: Samples={metrics["samples"]}, Positives={metrics["positives"]}')
        print(f'  AUC: {metrics["auc"]:.4f}, AUPR: {metrics["aupr"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}')


def train_model(model, drug_model, data_loader, optimizer, total_epochs, device, lambda_coarse_schedule,
                lambda_fine_schedule, pos_weight, fine_pos_weight, test_loader, test_idx, drug1_smiles_list,
                drug2_smiles_list, output_path):
    drug_model.eval()
    model.train()
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    for epoch in range(total_epochs):
        lambda_coarse = lambda_coarse_schedule[epoch] if epoch < len(lambda_coarse_schedule) else lambda_coarse_schedule[-1]
        lambda_fine = lambda_fine_schedule[epoch] if epoch < len(lambda_fine_schedule) else lambda_fine_schedule[-1]

        print(f"\nEpoch {epoch + 1}/{total_epochs} | Lambda Coarse: {lambda_coarse:.1f} | Lambda Fine: {lambda_fine:.1f} | Pos Weight: {pos_weight:.1f} | Fine Pos Weight: {fine_pos_weight:.1f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        total_loss = 0.0
        coarse_acc = []
        fine_acc = []

        for batch in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{total_epochs}'):
            h_drug1 = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            h_drug2 = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )

            outputs = model(h_drug1, h_drug2)

            R_coarse, R_fine, R_total = compute_rewards(
                outputs['gate_action'],
                batch['label_coarse'].to(device),
                outputs['fine_probs'],
                batch['label_fine'].to(device) if 'label_fine' in batch else None,
                pos_weight=pos_weight,
                lambda_coarse=lambda_coarse,
                lambda_fine=lambda_fine
            )

            log_probs = torch.log(outputs['gate_probs'].gather(1, outputs['gate_action'].unsqueeze(1)).squeeze(1))
            advantage = R_total - outputs['value'].detach()
            policy_loss = -(log_probs * advantage).mean()

            value_loss = nn.functional.mse_loss(outputs['value'], R_total)

            fine_loss = 0.0
            if outputs['fine_logits'] is not None and 'label_fine' in batch:
                valid_indices = (batch['label_coarse'].to(device) == 1)
                
                if valid_indices.any():
                    fine_loss = nn.functional.binary_cross_entropy_with_logits(
                        outputs['fine_logits'][valid_indices],
                        batch['label_fine'].to(device)[valid_indices],
                        weight=batch['label_fine'].to(device)[valid_indices] * (fine_pos_weight - 1) + 1
                    )

            loss = policy_loss + 0.5 * value_loss + lambda_fine * fine_loss
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            coarse_acc.append(R_coarse.mean().item())
            if R_fine.numel() > 0:
                fine_acc.append(R_fine[R_fine > 0].mean().item())

        print(f'Epoch {epoch + 1} | Loss: {total_loss / len(data_loader):.4f} | Coarse Acc: {np.mean(coarse_acc):.4f} | Fine Acc: {np.mean(fine_acc) if fine_acc else 0:.4f}')
        
        scheduler.step()

        print('\nCurrent epoch evaluation results:')
        metrics = evaluate_model(model, drug_model, test_loader, device)
        print_metrics(metrics, prefix="Current Epoch ")

    print('\nFinal evaluation results:')
    metrics = evaluate_model(model, drug_model, test_loader, device)
    print_metrics(metrics)

    output_path_fold = f"{os.path.splitext(output_path)[0]}.csv"
    save_predictions(model, drug_model, test_loader, test_idx, drug1_smiles_list, drug2_smiles_list, output_path_fold,
                     device)
    print(f"Final prediction results saved to: {output_path_fold}")

    return metrics


def save_predictions(model, drug_model, data_loader, test_idx, drug1_smiles_list, drug2_smiles_list, output_path,
                     device):
    drug_model.eval()
    model.eval()

    results = []
    with torch.no_grad():
        for batch in data_loader:
            h_drug1 = drug_model(batch['drug1_input_ids'].to(device), batch['drug1_attention_mask'].to(device))
            h_drug2 = drug_model(batch['drug2_input_ids'].to(device), batch['drug2_attention_mask'].to(device))
            outputs = model(h_drug1, h_drug2, train_mode=False)

            for i in range(len(batch['label_coarse'])):
                record = {
                    'drug1_SMILES': drug1_smiles_list[test_idx[i]],
                    'drug2_SMILES': drug2_smiles_list[test_idx[i]],
                    'true_coarse': int(batch['label_coarse'][i]),
                    'pred_coarse_prob': float(outputs['gate_probs'][i, 1]),
                    'pred_coarse': int(outputs['gate_action'][i]),
                }
                
                if outputs['fine_probs'] is not None:
                    for j in range(FINE_GRAINED_CLASSES):
                        record[f'true_fine_{j+1}'] = float(batch['label_fine'][i, j]) if 'label_fine' in batch else -1
                        record[f'pred_fine_{j+1}_prob'] = float(outputs['fine_probs'][i, j])
                        record[f'pred_fine_{j+1}'] = int(outputs['fine_probs'][i, j] >= FINE_CLASSIFICATION_THRESHOLD)
                
                results.append(record)

    pd.DataFrame(results).to_csv(output_path, index=False)


def main():
    global FINE_CLASSIFICATION_THRESHOLD
    
    parser = argparse.ArgumentParser(description='Drug-Drug Interaction Prediction (RL Gated Version)')
    parser.add_argument('--threshold', type=float, default=FINE_CLASSIFICATION_THRESHOLD, help='Fine-grained classification threshold')
    parser.add_argument('--data', default='D_data_with_SMILES.csv', help='Dataset path')
    parser.add_argument('--drug_model', default='best_drug_transformer_BN.pth', help='Drug pre-trained model path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Coarse-grained positive sample weight for class imbalance')
    parser.add_argument('--fine_pos_weight', type=float, default=2.0, help='Fine-grained positive sample weight for class imbalance')
    parser.add_argument('--output', default='predictions.csv', help='Prediction output path')
    parser.add_argument('--no_balance', action='store_true', help='Do not use data balancing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    FINE_CLASSIFICATION_THRESHOLD = args.threshold

    df = pd.read_csv(args.data)
    drug1_smiles_list = df['drug1_SMILES'].tolist()
    drug2_smiles_list = df['drug2_SMILES'].tolist()
    label_coarse = df['label_coarse'].values
    
    label1_str_list = None
    if 'label1' in df.columns:
        label1_str_list = df['label1'].tolist()
    elif 'label_fine' in df.columns:
        label1_str_list = df['label_fine'].tolist()

    print(f"Stratified sampling {args.test_size:.0%} as test set from all data, using all data for training")
    _, test_idx = train_test_split(
        np.arange(len(label_coarse)),
        test_size=args.test_size,
        random_state=42,
        stratify=label_coarse
    )

    train_dataset = DrugDrugInteractionDataset(
        drug1_smiles_list,
        drug2_smiles_list,
        label_coarse,
        label1_str_list=label1_str_list,
        balance_data=not args.no_balance,
        random_state=42
    )
    
    test_dataset = DrugDrugInteractionDataset(
        [drug1_smiles_list[i] for i in test_idx],
        [drug2_smiles_list[i] for i in test_idx],
        label_coarse[test_idx],
        label1_str_list=[label1_str_list[i] for i in test_idx] if label1_str_list is not None else None,
        balance_data=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    drug_model = DrugTransformer()
    drug_model = load_pretrained_model(drug_model, args.drug_model, device)
    drug_model = drug_model.to(device)

    rl_model = RLGateModel().to(device)
    optimizer = torch.optim.AdamW(rl_model.parameters(), lr=1e-4, weight_decay=1e-5)

    lambda_coarse_schedule = [1.0] * args.epochs
    lambda_fine_schedule = [0.5] * 10 + [1.0] * (args.epochs - 10)

    print('\nStarting training...')
    metrics = train_model(rl_model, drug_model, train_loader, optimizer, args.epochs, device,
                          lambda_coarse_schedule, lambda_fine_schedule, args.pos_weight, args.fine_pos_weight,
                          test_loader, test_idx, drug1_smiles_list, drug2_smiles_list, args.output)

    print("\n===== Final Evaluation Results =====")
    print("\nCoarse-grained Metrics:")
    print(f"AUC: {metrics['coarse_auc']:.4f}")
    print(f"AUPR: {metrics['coarse_aupr']:.4f}")
    print(f"Accuracy: {metrics['coarse_acc']:.4f}")

    print("\nFine-grained Metrics (Multi-label):")
    print(f"Average AUC: {metrics['fine_auc']:.4f}")
    print(f"Average AUPR: {metrics['fine_aupr']:.4f}")
    print(f"Average Accuracy: {metrics['fine_acc']:.4f}")


if __name__ == "__main__":
    main()