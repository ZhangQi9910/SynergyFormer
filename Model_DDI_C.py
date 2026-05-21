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

MAX_SMILES_LEN = 128

FINE_CLASSIFICATION_THRESHOLD = 0.5

FINE_GRAINED_CLASSES = 86


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
            pos_indices_fine = np.intersect1d(np.where(label_fine > 0)[0], valid_indices)
            neg_indices_fine = np.intersect1d(np.where(label_fine == 0)[0], valid_indices)
            
            if len(pos_indices_fine) > 0 and len(neg_indices_fine) > 0:
                label_fine_valid = np.zeros_like(label_fine)
                label_fine_valid[valid_indices] = label_fine[valid_indices]
                indices_fine = oversample_indices(label_fine_valid, random_state)
                combined_indices = np.unique(np.concatenate([indices_coarse, indices_fine]))
                return combined_indices

    return indices_coarse


class DrugDrugInteractionDataset(Dataset):
    
    def __init__(self, drug1_smiles_list, drug2_smiles_list, label_coarse, label_fine, balance_data=True,
                 random_state=42):
        self.drug1_smiles_list = drug1_smiles_list
        self.drug2_smiles_list = drug2_smiles_list
        self.label_coarse = label_coarse
        self.label_fine = label_fine
        
        self.fine_valid_mask = (self.label_fine > 0)
        self.label_fine = np.where(self.fine_valid_mask, self.label_fine, 0).astype(int)
        
        if balance_data:
            indices = balance_data_indices(self.label_coarse, self.label_fine, self.fine_valid_mask, random_state)
            self.drug1_smiles_list = [self.drug1_smiles_list[i] for i in indices]
            self.drug2_smiles_list = [self.drug2_smiles_list[i] for i in indices]
            self.label_coarse = self.label_coarse[indices]
            self.label_fine = self.label_fine[indices]
            self.fine_valid_mask = self.fine_valid_mask[indices]
        
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
        
        return {
            'drug1_input_ids': torch.tensor(drug1_input_ids, dtype=torch.long),
            'drug1_attention_mask': torch.tensor(drug1_attention_mask, dtype=torch.long),
            'drug2_input_ids': torch.tensor(drug2_input_ids, dtype=torch.long),
            'drug2_attention_mask': torch.tensor(drug2_attention_mask, dtype=torch.long),
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
    
    def __init__(self, embed_dim=256, gate_hidden=128, fine_grained_classes=FINE_GRAINED_CLASSES):
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
                    lambda_coarse=1.0, lambda_fine=0.5):
    R_coarse = torch.zeros_like(gate_action, dtype=torch.float)
    pos_mask = (label_coarse == 1)
    neg_mask = (label_coarse == 0)
    
    R_coarse[pos_mask] = pos_weight * (gate_action[pos_mask] == label_coarse[pos_mask]).float()
    R_coarse[neg_mask] = (gate_action[neg_mask] == label_coarse[neg_mask]).float()
    
    valid_mask = (gate_action == 1) & (label_coarse == 1) & label_fine_valid
    R_fine = torch.zeros_like(R_coarse)
    
    if fine_logits is not None and valid_mask.any():
        fine_pred = torch.argmax(fine_logits, dim=-1)
        # Adjust for 0-based indexing vs 1-based labels
        R_fine[valid_mask] = (fine_pred[valid_mask] == (label_fine[valid_mask] - 1)).float()
    
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

            valid_mask = (batch['label_coarse'] == 1) & batch['label_fine_valid'].cpu()
            if valid_mask.any() and outputs['fine_logits'] is not None:
                fine_logits = outputs['fine_logits'].cpu()[valid_mask]
                fine_probs = torch.softmax(fine_logits, dim=-1).numpy()
                fine_pred = np.argmax(fine_probs, axis=1) + 1 
                
                fine_preds.extend(fine_pred)
                fine_true.extend(batch['label_fine'].cpu().numpy()[valid_mask])
                
                fine_probs_all.append(fine_probs)
                
                batch_fine_true = batch['label_fine'].cpu().numpy()[valid_mask]
                for i in range(FINE_GRAINED_CLASSES):
                    class_true = (batch_fine_true == (i + 1)).astype(int)
                    class_pred = fine_probs[:, i]
                    fine_class_metrics[i]['y_true'].extend(class_true)
                    fine_class_metrics[i]['y_pred'].extend(class_pred)

    coarse_auc = roc_auc_score(coarse_true, coarse_probs)
    coarse_aupr = average_precision_score(coarse_true, coarse_probs)
    coarse_acc = accuracy_score(coarse_true, np.array(coarse_probs) > 0.5)

    fine_acc = 0.0
    fine_auc = 0.0
    fine_aupr = 0.0
    valid_classes = 0
    
    if fine_true:
        fine_acc = accuracy_score(fine_true, fine_preds)
        
        for i in range(FINE_GRAINED_CLASSES):
            y_true = np.array(fine_class_metrics[i]['y_true'])
            y_pred = np.array(fine_class_metrics[i]['y_pred'])
            
            if len(y_true) > 0 and np.sum(y_true) > 0 and np.sum(y_true) < len(y_true):
                try:
                    fine_auc += roc_auc_score(y_true, y_pred)
                    fine_aupr += average_precision_score(y_true, y_pred)
                    valid_classes += 1
                except:
                    pass
        
        if valid_classes > 0:
            fine_auc /= valid_classes
            fine_aupr /= valid_classes

    return {
        'coarse_auc': coarse_auc,
        'coarse_aupr': coarse_aupr,
        'coarse_acc': coarse_acc,
        'fine_acc': fine_acc,
        'fine_auc': fine_auc if valid_classes > 0 else 0,
        'fine_aupr': fine_aupr if valid_classes > 0 else 0
    }


def train_model(model, drug_model, data_loader, optimizer, total_epochs, device, lambda_coarse_schedule,
                lambda_fine_schedule, pos_weight, fine_pos_weight, test_loader, test_idx, drug1_smiles_list,
                drug2_smiles_list, output_path, fold_idx=None):
    drug_model.eval()
    model.train()

    for epoch in range(total_epochs):
        lambda_coarse = lambda_coarse_schedule[epoch] if epoch < len(lambda_coarse_schedule) else lambda_coarse_schedule[-1]
        lambda_fine = lambda_fine_schedule[epoch] if epoch < len(lambda_fine_schedule) else lambda_fine_schedule[-1]

        fold_prefix = f"Fold {fold_idx + 1}/5 | " if fold_idx is not None else ""
        print(f"\n{fold_prefix}Epoch {epoch + 1}/{total_epochs} | Lambda Coarse: {lambda_coarse:.1f} | Lambda Fine: {lambda_fine:.1f} | Pos Weight: {pos_weight:.1f} | Fine Pos Weight: {fine_pos_weight:.1f}")

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
                    # Adjust labels to be 0-indexed for CrossEntropyLoss
                    valid_labels = batch['label_fine'].to(device)[valid_indices] - 1
                    
                    fine_loss = nn.functional.cross_entropy(valid_fine_logits, valid_labels)
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

        print(f'Epoch {epoch + 1} | Loss: {total_loss / len(data_loader):.4f} | Coarse Acc: {np.mean(coarse_acc):.4f} | Fine Acc: {np.mean(fine_acc) if fine_acc else 0:.4f}')

        if (epoch + 1) == 30:
            print('\nEvaluation results at epoch 30:')
            metrics = evaluate_model(model, drug_model, test_loader, device)

            fold_suffix = f"_fold{fold_idx + 1}" if fold_idx is not None else ""
            output_path_epoch30 = f"{os.path.splitext(output_path)[0]}{fold_suffix}_epoch30.csv"
            save_predictions(model, drug_model, test_loader, test_idx, drug1_smiles_list, drug2_smiles_list,
                             output_path_epoch30, device)
            print(f"Prediction results at epoch 30 saved to: {output_path_epoch30}")

    print('\nFinal evaluation results:')
    metrics = evaluate_model(model, drug_model, test_loader, device)

    fold_suffix = f"_fold{fold_idx + 1}" if fold_idx is not None else ""
    output_path_fold = f"{os.path.splitext(output_path)[0]}{fold_suffix}.csv"
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
                fine_pred = -1
                fine_prob = -1
                
                if outputs['fine_logits'] is not None and outputs['gate_action'][i] == 1:
                    fine_probs = torch.softmax(outputs['fine_logits'][i], dim=-1)
                    fine_pred = int(torch.argmax(fine_probs) + 1)
                    fine_prob = float(torch.max(fine_probs))

                record = {
                    'drug1_SMILES': drug1_smiles_list[test_idx[i]],
                    'drug2_SMILES': drug2_smiles_list[test_idx[i]],
                    'true_coarse': int(batch['label_coarse'][i]),
                    'true_fine': int(batch['label_fine'][i]) if batch['label_fine_valid'][i] else -1,
                    'pred_coarse_prob': float(outputs['gate_probs'][i, 1]),
                    'pred_coarse': int(outputs['gate_action'][i]),
                    'pred_fine': fine_pred,
                    'pred_fine_prob': fine_prob
                }
                results.append(record)

    pd.DataFrame(results).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Drug-Drug Interaction Prediction (RL Gated Version)')
    parser.add_argument('--threshold', type=float, default=FINE_CLASSIFICATION_THRESHOLD, help='Fine-grained classification threshold')
    parser.add_argument('--data', default='DDI+moa.csv', help='Dataset path')
    parser.add_argument('--drug_model', default='best_drug_transformer_BN.pth', help='Drug pre-trained model path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='Coarse-grained positive sample weight for class imbalance')
    parser.add_argument('--fine_pos_weight', type=float, default=2.0, help='Fine-grained positive sample weight for class imbalance')
    parser.add_argument('--output', default='predictions.csv', help='Prediction output path')
    parser.add_argument('--no_balance', action='store_true', help='Do not use data balancing')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    df = pd.read_csv(args.data)
    drug1_smiles_list = df['drug1_SMILES'].tolist()
    drug2_smiles_list = df['drug2_SMILES'].tolist()
    label_coarse = df['label_coarse'].values
    label_fine = df['label_fine'].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(label_coarse)):
        print(f"\n===== Starting Fold {fold_idx + 1}/5 Cross-Validation =====")

        train_dataset = DrugDrugInteractionDataset(
            [drug1_smiles_list[i] for i in train_idx],
            [drug2_smiles_list[i] for i in train_idx],
            label_coarse[train_idx],
            label_fine[train_idx],
            balance_data=not args.no_balance,
            random_state=42 + fold_idx
        )
        test_dataset = DrugDrugInteractionDataset(
            [drug1_smiles_list[i] for i in test_idx],
            [drug2_smiles_list[i] for i in test_idx],
            label_coarse[test_idx],
            label_fine[test_idx],
            balance_data=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        drug_model = DrugTransformer()
        drug_model = load_pretrained_model(drug_model, args.drug_model, device)
        drug_model = drug_model.to(device)

        rl_model = RLGateModel().to(device)
        optimizer = torch.optim.Adam(rl_model.parameters(), lr=1e-4)

        lambda_coarse_schedule = [1.0] * args.epochs
        lambda_fine_schedule = [1.0] * min(30, args.epochs) + [1.0] * max(0, args.epochs - 30)

        print(f'\nStarting training for Fold {fold_idx + 1}/5...')
        metrics = train_model(rl_model, drug_model, train_loader, optimizer, args.epochs, device,
                              lambda_coarse_schedule, lambda_fine_schedule, args.pos_weight, args.fine_pos_weight,
                              test_loader, test_idx, drug1_smiles_list, drug2_smiles_list, args.output, fold_idx)

        all_metrics.append(metrics)

    print("\n===== 5-Fold Cross-Validation Summary =====")
    print("\nAverage Coarse-grained Metrics:")
    print(f"AUC: {np.mean([m['coarse_auc'] for m in all_metrics]):.4f} ± {np.std([m['coarse_auc'] for m in all_metrics]):.4f}")
    print(f"AUPR: {np.mean([m['coarse_aupr'] for m in all_metrics]):.4f} ± {np.std([m['coarse_aupr'] for m in all_metrics]):.4f}")
    print(f"Accuracy: {np.mean([m['coarse_acc'] for m in all_metrics]):.4f} ± {np.std([m['coarse_acc'] for m in all_metrics]):.4f}")

    print("\nAverage Fine-grained Metrics (Active and valid samples only):")
    print(f"Accuracy: {np.mean([m['fine_acc'] for m in all_metrics]):.4f} ± {np.std([m['fine_acc'] for m in all_metrics]):.4f}")
    print(f"Average AUC: {np.mean([m['fine_auc'] for m in all_metrics]):.4f} ± {np.std([m['fine_auc'] for m in all_metrics]):.4f}")
    print(f"Average AUPR: {np.mean([m['fine_aupr'] for m in all_metrics]):.4f} ± {np.std([m['fine_aupr'] for m in all_metrics]):.4f}")


if __name__ == "__main__":
    main()