import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import os
from tqdm import tqdm
import argparse
from rdkit import Chem
import math
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)

SMILES_CHARS = [' ', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']', 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(SMILES_CHARS)}

AA_PROPERTIES = {
    'A': {'hydrophobicity': 0.62, 'polarity': 8.1, 'volume': 31.0, 'mw': 89.09},
    'R': {'hydrophobicity': -2.53, 'polarity': 10.5, 'volume': 105.0, 'mw': 174.20},
    'N': {'hydrophobicity': -0.78, 'polarity': 11.6, 'volume': 56.0, 'mw': 132.12},
    'D': {'hydrophobicity': -0.90, 'polarity': 13.0, 'volume': 54.0, 'mw': 133.10},
    'C': {'hydrophobicity': 0.29, 'polarity': 5.5, 'volume': 55.0, 'mw': 121.15},
    'E': {'hydrophobicity': -0.74, 'polarity': 12.3, 'volume': 83.0, 'mw': 147.13},
    'Q': {'hydrophobicity': -0.85, 'polarity': 10.5, 'volume': 85.0, 'mw': 146.15},
    'G': {'hydrophobicity': 0.48, 'polarity': 9.0, 'volume': 3.0, 'mw': 75.07},
    'H': {'hydrophobicity': -0.40, 'polarity': 10.4, 'volume': 79.0, 'mw': 155.16},
    'I': {'hydrophobicity': 1.38, 'polarity': 5.2, 'volume': 111.0, 'mw': 131.17},
    'L': {'hydrophobicity': 1.06, 'polarity': 4.9, 'volume': 111.0, 'mw': 131.17},
    'K': {'hydrophobicity': -1.50, 'polarity': 11.3, 'volume': 100.0, 'mw': 146.19},
    'M': {'hydrophobicity': 0.64, 'polarity': 5.7, 'volume': 105.0, 'mw': 149.21},
    'F': {'hydrophobicity': 1.19, 'polarity': 5.2, 'volume': 132.0, 'mw': 165.19},
    'P': {'hydrophobicity': 0.12, 'polarity': 8.0, 'volume': 32.0, 'mw': 115.13},
    'S': {'hydrophobicity': -0.18, 'polarity': 9.2, 'volume': 32.0, 'mw': 105.09},
    'T': {'hydrophobicity': -0.05, 'polarity': 8.6, 'volume': 61.0, 'mw': 119.12},
    'W': {'hydrophobicity': 0.81, 'polarity': 5.4, 'volume': 170.0, 'mw': 204.23},
    'Y': {'hydrophobicity': 0.26, 'polarity': 6.2, 'volume': 136.0, 'mw': 181.19},
    'V': {'hydrophobicity': 1.08, 'polarity': 5.9, 'volume': 84.0, 'mw': 117.15}
}

standard_aas = sorted(AA_PROPERTIES.keys())
special_tokens = ['[PAD]', '[CLS]', '[MASK]']
PROTEIN_CHARS = special_tokens + standard_aas
PROTEIN_CHAR_TO_IDX = {char: idx for idx, char in enumerate(PROTEIN_CHARS)}

MAX_SMILES_LEN = 128
MAX_PROTEIN_LEN = 1000


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


def balance_data_indices(label_coarse, label_fine, fine_valid_mask=None, random_state=42):
    np.random.seed(random_state)
    
    indices_coarse = oversample_indices(label_coarse, random_state)
    
    if fine_valid_mask is not None:
        valid_indices = np.where(fine_valid_mask)[0]
        
        if len(valid_indices) > 0:
            pos_indices_fine = np.intersect1d(np.where(label_fine == 1)[0], valid_indices)
            neg_indices_fine = np.intersect1d(np.where(label_fine == 0)[0], valid_indices)
            
            if len(pos_indices_fine) > 0 and len(neg_indices_fine) > 0:
                label_fine_valid = np.zeros_like(label_fine)
                label_fine_valid[valid_indices] = label_fine[valid_indices]
                
                indices_fine = oversample_indices(label_fine_valid, random_state)
                
                combined_indices = np.unique(np.concatenate([indices_coarse, indices_fine]))
                return combined_indices
    
    return indices_coarse


class DrugProteinDataset(Dataset):

    def __init__(self, smiles_list, fasta_list, label_coarse, label_fine, balance_data=True, random_state=42):
        self.smiles_list = smiles_list
        self.fasta_list = fasta_list
        self.label_coarse = label_coarse
        self.label_fine = label_fine

        self.fine_valid_mask = ~np.isnan(self.label_fine)
        self.label_fine = np.nan_to_num(self.label_fine, nan=0).astype(int)

        if balance_data:
            indices = balance_data_indices(self.label_coarse, self.label_fine, self.fine_valid_mask, random_state)
            self.smiles_list = [self.smiles_list[i] for i in indices]
            self.fasta_list = [self.fasta_list[i] for i in indices]
            self.label_coarse = self.label_coarse[indices]
            self.label_fine = self.label_fine[indices]
            self.fine_valid_mask = self.fine_valid_mask[indices]

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        idx_seq = [CHAR_TO_IDX.get(c, 0) for c in smiles]
        idx_seq = [CHAR_TO_IDX[' ']] + idx_seq[:MAX_SMILES_LEN - 2] + [CHAR_TO_IDX[' ']]
        input_ids_drug = idx_seq + [0] * (MAX_SMILES_LEN - len(idx_seq))
        attention_mask_drug = [1] * len(idx_seq) + [0] * (MAX_SMILES_LEN - len(idx_seq))

        fasta = self.fasta_list[idx]
        idx_seq_prot = [PROTEIN_CHAR_TO_IDX['[CLS]']] + [PROTEIN_CHAR_TO_IDX.get(c, PROTEIN_CHAR_TO_IDX['[PAD]']) for c
                                                         in fasta]
        idx_seq_prot = idx_seq_prot[:MAX_PROTEIN_LEN]
        idx_seq_prot = idx_seq_prot + [PROTEIN_CHAR_TO_IDX['[PAD]']] * (MAX_PROTEIN_LEN - len(idx_seq_prot))
        input_ids_prot = idx_seq_prot
        attention_mask_prot = [1 if token != PROTEIN_CHAR_TO_IDX['[PAD]'] else 0 for token in idx_seq_prot]

        return {
            'drug_input_ids': torch.tensor(input_ids_drug, dtype=torch.long),
            'drug_attention_mask': torch.tensor(attention_mask_drug, dtype=torch.long),
            'prot_input_ids': torch.tensor(input_ids_prot, dtype=torch.long),
            'prot_attention_mask': torch.tensor(attention_mask_prot, dtype=torch.long),
            'label_coarse': torch.tensor(self.label_coarse[idx], dtype=torch.long),
            'label_fine': torch.tensor(self.label_fine[idx], dtype=torch.long),
            'label_fine_valid': torch.tensor(self.fine_valid_mask[idx], dtype=torch.bool)
        }


class DrugTransformer(nn.Module):

    def __init__(self, vocab_size=len(SMILES_CHARS), embed_dim=256, num_heads=8, num_layers=6, hidden_dim=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=MAX_SMILES_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
        return x[:, 0, :]


class ProteinTransformer(nn.Module):

    def __init__(self, vocab_size=len(PROTEIN_CHARS), embed_dim=256, num_heads=8, num_layers=6, hidden_dim=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=MAX_PROTEIN_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
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
                               'transformer_encoder.')}

    if 'positional_encoding.pe' in filtered_checkpoint:
        pretrained_pe = filtered_checkpoint['positional_encoding.pe']
        model_pe = model_dict['positional_encoding.pe']

        if pretrained_pe.shape[0] < model_pe.shape[0]:
            model_pe[:pretrained_pe.shape[0], :] = pretrained_pe
            filtered_checkpoint['positional_encoding.pe'] = model_pe

    model.load_state_dict(filtered_checkpoint, strict=False)
    return model


class RLGateModel(nn.Module):

    def __init__(self, embed_dim=256, gate_hidden=128, fine_grained_classes=2):
        super().__init__()
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 2)
        )
        self.fine_grained_classifier = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, fine_grained_classes)
        )
        self.value_network = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )

    def forward(self, h_drug, h_prot, train_mode=True):
        combined = torch.cat([h_drug, h_prot], dim=-1)
        fused = self.fusion_layer(combined)
        gate_logits = self.gate_network(fused)
        gate_probs = nn.functional.softmax(gate_logits, dim=-1)

        if train_mode:
            gate_action = torch.multinomial(gate_probs, 1).squeeze(-1)
        else:
            gate_action = torch.argmax(gate_probs, dim=-1)

        fine_logits = None
        if (gate_action == 1).any() or train_mode:
            fine_logits = self.fine_grained_classifier(fused)

        value = self.value_network(fused).squeeze(-1)
        return {
            'gate_probs': gate_probs,
            'gate_action': gate_action,
            'fine_logits': fine_logits,
            'value': value
        }


def compute_rewards(gate_action, label_coarse, fine_logits, label_fine, label_fine_valid, pos_weight=1.0,
                    lambda_coarse=1.0,
                    lambda_fine=0.5):
    R_coarse = torch.zeros_like(gate_action, dtype=torch.float)
    pos_mask = (label_coarse == 1)
    neg_mask = (label_coarse == 0)

    R_coarse[pos_mask] = pos_weight * (gate_action[pos_mask] == label_coarse[pos_mask]).float()
    R_coarse[neg_mask] = (gate_action[neg_mask] == label_coarse[neg_mask]).float()

    valid_mask = (gate_action == 1) & (label_coarse == 1) & label_fine_valid
    R_fine = torch.zeros_like(R_coarse)

    if fine_logits is not None and valid_mask.any():
        fine_pred = torch.argmax(fine_logits, dim=-1)
        R_fine[valid_mask] = (fine_pred[valid_mask] == label_fine[valid_mask]).float()

    return R_coarse, R_fine, lambda_coarse * R_coarse + lambda_fine * R_fine


def train_model(model, drug_model, protein_model, data_loader, optimizer, total_epochs, device, lambda_coarse_schedule,
                lambda_fine_schedule, pos_weight, fine_pos_weight, test_loader, test_idx, smiles_list, fasta_list,
                output_path, fold_idx=None):
    drug_model.eval()
    protein_model.eval()
    model.train()

    for epoch in range(total_epochs):
        lambda_coarse = lambda_coarse_schedule[epoch] if epoch < len(lambda_coarse_schedule) else \
            lambda_coarse_schedule[-1]
        lambda_fine = lambda_fine_schedule[epoch] if epoch < len(lambda_fine_schedule) else lambda_fine_schedule[-1]

        fold_prefix = f"Fold {fold_idx + 1}/5 | " if fold_idx is not None else ""
        print(
            f"\n{fold_prefix}Epoch {epoch + 1}/{total_epochs} | Lambda Coarse: {lambda_coarse:.1f} | Lambda Fine: {lambda_fine:.1f} | Pos Weight: {pos_weight:.1f} | Fine Pos Weight: {fine_pos_weight:.1f}")

        total_loss = 0.0
        coarse_acc = []
        fine_acc = []

        for batch in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{total_epochs}'):
            h_drug = drug_model(
                batch['drug_input_ids'].to(device),
                batch['drug_attention_mask'].to(device)
            )
            h_prot = protein_model(
                batch['prot_input_ids'].to(device),
                batch['prot_attention_mask'].to(device)
            )

            outputs = model(h_drug, h_prot)

            R_coarse, R_fine, R_total = compute_rewards(
                outputs['gate_action'],
                batch['label_coarse'].to(device),
                outputs['fine_logits'],
                batch['label_fine'].to(device),
                batch['label_fine_valid'].to(device),
                pos_weight=pos_weight,
                lambda_coarse=lambda_coarse,
                lambda_fine=lambda_fine
            )

            log_probs = torch.log(outputs['gate_probs'].gather(1, outputs['gate_action'].unsqueeze(1)).squeeze(1))
            advantage = R_total - outputs['value'].detach()
            policy_loss = -(log_probs * advantage).mean()

            value_loss = nn.functional.mse_loss(outputs['value'], R_total)

            if outputs['fine_logits'] is not None:
                valid_indices = batch['label_fine_valid'].to(device) & (batch['label_coarse'].to(device) == 1)
                if valid_indices.any():
                    valid_fine_logits = outputs['fine_logits'][valid_indices]
                    valid_labels = batch['label_fine'].to(device)[valid_indices]

                    class_weights = torch.tensor([1.0, fine_pos_weight], device=device)
                    fine_loss = nn.functional.cross_entropy(valid_fine_logits, valid_labels, weight=class_weights)
                else:
                    fine_loss = 0.0
            else:
                fine_loss = 0.0

            loss = policy_loss + 0.5 * value_loss + lambda_fine * fine_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            coarse_acc.append(R_coarse.mean().item())
            if R_fine.numel() > 0:
                fine_acc.append(R_fine[R_fine > 0].mean().item())

        print(
            f'Epoch {epoch + 1} | Loss: {total_loss / len(data_loader):.4f} | Coarse Acc: {np.mean(coarse_acc):.4f} | Fine Acc: {np.mean(fine_acc) if fine_acc else 0:.4f}')

        if (epoch + 1) == 30:
            print('\nEvaluation results at epoch 30:')
            metrics = evaluate_model(model, drug_model, protein_model, test_loader, device)

            fold_suffix = f"_fold{fold_idx + 1}" if fold_idx is not None else ""
            output_path_epoch30 = f"{os.path.splitext(output_path)[0]}{fold_suffix}_epoch30.csv"
            save_predictions(model, drug_model, protein_model, test_loader, test_idx, smiles_list, fasta_list,
                             output_path_epoch30, device)
            print(f"Prediction results at epoch 30 saved to: {output_path_epoch30}")

    print('\nFinal evaluation results:')
    metrics = evaluate_model(model, drug_model, protein_model, test_loader, device)

    fold_suffix = f"_fold{fold_idx + 1}" if fold_idx is not None else ""
    output_path_fold = f"{os.path.splitext(output_path)[0]}{fold_suffix}.csv"
    save_predictions(model, drug_model, protein_model, test_loader, test_idx, smiles_list, fasta_list, output_path_fold,
                     device)
    print(f"Final prediction results saved to: {output_path_fold}")

    return metrics


def evaluate_model(model, drug_model, protein_model, data_loader, device):
    drug_model.eval()
    protein_model.eval()
    model.eval()

    coarse_probs = []
    coarse_true = []
    fine_pred_probs = []
    fine_true = []
    fine_valid_mask = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            h_drug = drug_model(batch['drug_input_ids'].to(device), batch['drug_attention_mask'].to(device))
            h_prot = protein_model(batch['prot_input_ids'].to(device), batch['prot_attention_mask'].to(device))
            outputs = model(h_drug, h_prot, train_mode=False)

            coarse_probs.extend(outputs['gate_probs'][:, 1].cpu().numpy())
            coarse_true.extend(batch['label_coarse'].cpu().numpy())

            valid_mask = (batch['label_coarse'] == 1) & batch['label_fine_valid'].cpu()
            if outputs['fine_logits'] is not None and valid_mask.any():
                fine_logits = outputs['fine_logits'].cpu()[valid_mask]
                fine_prob = nn.functional.softmax(fine_logits, dim=-1)[:, 1].numpy()
                fine_pred_probs.extend(fine_prob)
                fine_true.extend(batch['label_fine'].cpu().numpy()[valid_mask])
                fine_valid_mask.extend(valid_mask.numpy())

    coarse_auc = roc_auc_score(coarse_true, coarse_probs)
    coarse_aupr = average_precision_score(coarse_true, coarse_probs)
    coarse_acc = accuracy_score(coarse_true, np.array(coarse_probs) > 0.5)

    fine_acc = 0.0
    fine_auc = 0.0
    fine_aupr = 0.0
    if fine_true:
        fine_acc = accuracy_score(fine_true, np.array(fine_pred_probs) > 0.5)
        fine_auc = roc_auc_score(fine_true, fine_pred_probs)
        fine_aupr = average_precision_score(fine_true, fine_pred_probs)

    print('\nCoarse-grained metrics:')
    print(f'AUC: {coarse_auc:.4f}, AUPR: {coarse_aupr:.4f}, Accuracy: {coarse_acc:.4f}')
    print('\nFine-grained metrics (Active and valid samples only):')
    print(f'AUC: {fine_auc:.4f}, AUPR: {fine_aupr:.4f}, Accuracy: {fine_acc:.4f}')

    return {
        'coarse_auc': coarse_auc,
        'coarse_aupr': coarse_aupr,
        'coarse_acc': coarse_acc,
        'fine_auc': fine_auc,
        'fine_aupr': fine_aupr,
        'fine_acc': fine_acc
    }


def save_predictions(model, drug_model, protein_model, data_loader, test_idx, smiles_list, fasta_list, output_path,
                     device):
    drug_model.eval()
    protein_model.eval()
    model.eval()

    results = []
    with torch.no_grad():
        for batch in data_loader:
            h_drug = drug_model(batch['drug_input_ids'].to(device), batch['drug_attention_mask'].to(device))
            h_prot = protein_model(batch['prot_input_ids'].to(device), batch['prot_attention_mask'].to(device))
            outputs = model(h_drug, h_prot, train_mode=False)

            for i in range(len(batch['label_coarse'])):
                fine_prob = -1
                if outputs['fine_logits'] is not None and outputs['gate_action'][i] == 1:
                    fine_prob = float(nn.functional.softmax(outputs['fine_logits'][i:i + 1], dim=-1)[:, 1])

                record = {
                    'SMILES': smiles_list[test_idx[i]],
                    'fasta': fasta_list[test_idx[i]],
                    'true_coarse': int(batch['label_coarse'][i]),
                    'true_fine': int(batch['label_fine'][i]) if batch['label_fine_valid'][i] else -1,
                    'pred_coarse_prob': float(outputs['gate_probs'][i, 1]),
                    'pred_coarse': int(outputs['gate_action'][i]),
                    'pred_fine': int(torch.argmax(outputs['fine_logits'][i])) if (
                            outputs['gate_action'][i] == 1 and outputs['fine_logits'] is not None) else -1,
                    'pred_fine_prob': fine_prob
                }
                results.append(record)

    pd.DataFrame(results).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Drug-Protein Interaction Prediction (RL Gated Version)')
    parser.add_argument('--data', default='DTI+moa.csv', help='Dataset path')
    parser.add_argument('--drug_model', default='best_drug_transformer_BN.pth', help='Drug pre-trained model path')
    parser.add_argument('--protein_model', default='best_protein_transformer_BN.pth', help='Protein pre-trained model path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Coarse-grained positive sample weight for class imbalance')
    parser.add_argument('--fine_pos_weight', type=float, default=2.0, help='Fine-grained positive sample weight for class imbalance')
    parser.add_argument('--output', default='predictions.csv', help='Prediction output path')
    parser.add_argument('--no_balance', action='store_true', help='Do not use data balancing')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    df = pd.read_csv(args.data)
    smiles_list = df['SMILES'].tolist()
    fasta_list = df['fasta'].tolist()
    label_coarse = df['label_1'].values
    label_fine = df['label_2'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(label_coarse)):
        print(f"\n===== Starting Fold {fold_idx + 1}/5 Cross-Validation =====")

        train_dataset = DrugProteinDataset(
            [smiles_list[i] for i in train_idx],
            [fasta_list[i] for i in train_idx],
            label_coarse[train_idx],
            label_fine[train_idx],
            balance_data=not args.no_balance,
            random_state=42 + fold_idx
        )
        test_dataset = DrugProteinDataset(
            [smiles_list[i] for i in test_idx],
            [fasta_list[i] for i in test_idx],
            label_coarse[test_idx],
            label_fine[test_idx],
            balance_data=False
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        drug_model = DrugTransformer()
        drug_model = load_pretrained_model(drug_model, args.drug_model, device)
        drug_model = drug_model.to(device)

        protein_model = ProteinTransformer()
        protein_model = load_pretrained_model(protein_model, args.protein_model, device)
        protein_model = protein_model.to(device)

        rl_model = RLGateModel().to(device)
        optimizer = torch.optim.Adam(rl_model.parameters(), lr=1e-4)

        lambda_coarse_schedule = [1.0] * args.epochs
        lambda_fine_schedule = [1.0] * min(30, args.epochs) + [1.0] * max(0,
                                                                          args.epochs - 30)

        print(f'\nStarting training for Fold {fold_idx + 1}/5...')
        metrics = train_model(rl_model, drug_model, protein_model, train_loader, optimizer, args.epochs, device,
                              lambda_coarse_schedule, lambda_fine_schedule, args.pos_weight, args.fine_pos_weight,
                              test_loader, test_idx, smiles_list, fasta_list, args.output, fold_idx)

        all_metrics.append(metrics)

    print("\n===== 5-Fold Cross-Validation Summary =====")
    print("\nAverage Coarse-grained Metrics:")
    print(
        f"AUC: {np.mean([m['coarse_auc'] for m in all_metrics]):.4f} ± {np.std([m['coarse_auc'] for m in all_metrics]):.4f}")
    print(
        f"AUPR: {np.mean([m['coarse_aupr'] for m in all_metrics]):.4f} ± {np.std([m['coarse_aupr'] for m in all_metrics]):.4f}")
    print(
        f"Accuracy: {np.mean([m['coarse_acc'] for m in all_metrics]):.4f} ± {np.std([m['coarse_acc'] for m in all_metrics]):.4f}")

    print("\nAverage Fine-grained Metrics (Active and valid samples only):")
    print(
        f"AUC: {np.mean([m['fine_auc'] for m in all_metrics]):.4f} ± {np.std([m['fine_auc'] for m in all_metrics]):.4f}")
    print(
        f"AUPR: {np.mean([m['fine_aupr'] for m in all_metrics]):.4f} ± {np.std([m['fine_aupr'] for m in all_metrics]):.4f}")
    print(
        f"Accuracy: {np.mean([m['fine_acc'] for m in all_metrics]):.4f} ± {np.std([m['fine_acc'] for m in all_metrics]):.4f}")


if __name__ == "__main__":
    main()