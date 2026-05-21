import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_squared_error
from scipy import stats
import os
from tqdm import tqdm
import argparse
import math
from collections import defaultdict
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

SMILES_CHARS = [' ', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']', 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(SMILES_CHARS)}

MAX_SMILES_LEN = 128

CELL_FEATURE_DIM = 954

REGRESSION_THRESHOLD = 0.5

def parse_cell_features(cell_feature_str):
    try:
        values = cell_feature_str.strip('[]').split(', ')
        return np.array([float(v) for v in values], dtype=np.float32)
    except:
        return np.zeros(CELL_FEATURE_DIM, dtype=np.float32)

def calculate_cell_feature_stats(cell_features_list):
    all_features = np.vstack(cell_features_list)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)
    std[std < 1e-8] = 1.0
    return mean, std

def normalize_cell_features(cell_features, mean, std):
    return (cell_features - mean) / std

class DrugDrugInteractionDataset(Dataset):
    
    def __init__(self, drug1_smiles_list, drug2_smiles_list, cell_features_list, 
                 label_coarse, synergy_values, balance_data=True, random_state=42, 
                 cell_mean=None, cell_std=None):
        self.drug1_smiles_list = drug1_smiles_list
        self.drug2_smiles_list = drug2_smiles_list
        self.cell_features_list = cell_features_list
        self.label_coarse = label_coarse
        self.synergy_values = synergy_values
        
        if cell_mean is not None and cell_std is not None:
            self.cell_features_list = [normalize_cell_features(cf, cell_mean, cell_std) 
                                     for cf in self.cell_features_list]
        
        if balance_data:
            self.balance_data(random_state)
            
        self.drug1_cache = {}
        self.drug2_cache = {}
    
    def balance_data(self, random_state):
        np.random.seed(random_state)
        pos_indices = np.where(self.label_coarse == 1)[0]
        neg_indices = np.where(self.label_coarse == 0)[0]
        
        if len(pos_indices) < len(neg_indices) // 2:
            additional_pos = np.random.choice(pos_indices, 
                                            len(neg_indices) // 2 - len(pos_indices), 
                                            replace=True)
            pos_indices = np.concatenate([pos_indices, additional_pos])
        
        if len(neg_indices) > 0:
            sample_size = min(len(pos_indices), len(neg_indices))
            neg_indices = np.random.choice(neg_indices, sample_size, replace=False)
        else:
            print("Warning: No negative samples available for balancing!")
            neg_indices = np.array([], dtype=np.int64)
        
        if len(pos_indices) > 0 and len(neg_indices) > 0:
            selected_indices = np.concatenate([pos_indices, neg_indices])
            np.random.shuffle(selected_indices)
            
            self.drug1_smiles_list = [self.drug1_smiles_list[i] for i in selected_indices]
            self.drug2_smiles_list = [self.drug2_smiles_list[i] for i in selected_indices]
            self.cell_features_list = [self.cell_features_list[i] for i in selected_indices]
            self.label_coarse = self.label_coarse[selected_indices]
            self.synergy_values = self.synergy_values[selected_indices]
        else:
            print("Warning: Not enough positive or negative samples for balancing!")

    def __len__(self):
        return len(self.drug1_smiles_list)

    def __getitem__(self, idx):
        if idx not in self.drug1_cache:
            drug1_smiles = self.drug1_smiles_list[idx]
            drug1_idx_seq = [CHAR_TO_IDX.get(c, 0) for c in drug1_smiles]
            drug1_idx_seq = [CHAR_TO_IDX[' ']] + drug1_idx_seq[:MAX_SMILES_LEN-2] + [CHAR_TO_IDX[' ']]
            drug1_input_ids = drug1_idx_seq + [0] * (MAX_SMILES_LEN - len(drug1_idx_seq))
            drug1_attention_mask = [1] * len(drug1_idx_seq) + [0] * (MAX_SMILES_LEN - len(drug1_idx_seq))
            self.drug1_cache[idx] = (drug1_input_ids, drug1_attention_mask)
        
        if idx not in self.drug2_cache:
            drug2_smiles = self.drug2_smiles_list[idx]
            drug2_idx_seq = [CHAR_TO_IDX.get(c, 0) for c in drug2_smiles]
            drug2_idx_seq = [CHAR_TO_IDX[' ']] + drug2_idx_seq[:MAX_SMILES_LEN-2] + [CHAR_TO_IDX[' ']]
            drug2_input_ids = drug2_idx_seq + [0] * (MAX_SMILES_LEN - len(drug2_idx_seq))
            drug2_attention_mask = [1] * len(drug2_idx_seq) + [0] * (MAX_SMILES_LEN - len(drug2_idx_seq))
            self.drug2_cache[idx] = (drug2_input_ids, drug2_attention_mask)
        
        drug1_input_ids, drug1_attention_mask = self.drug1_cache[idx]
        drug2_input_ids, drug2_attention_mask = self.drug2_cache[idx]
        
        return {
            'drug1_input_ids': torch.tensor(drug1_input_ids, dtype=torch.long),
            'drug1_attention_mask': torch.tensor(drug1_attention_mask, dtype=torch.long),
            'drug2_input_ids': torch.tensor(drug2_input_ids, dtype=torch.long),
            'drug2_attention_mask': torch.tensor(drug2_attention_mask, dtype=torch.long),
            'cell_features': torch.tensor(self.cell_features_list[idx], dtype=torch.float32),
            'label_coarse': torch.tensor(self.label_coarse[idx], dtype=torch.long),
            'synergy': torch.tensor(self.synergy_values[idx], dtype=torch.float32)
        }

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

class DrugTransformer(nn.Module):
    def __init__(self, vocab_size=len(SMILES_CHARS), embed_dim=256, num_heads=8, 
                 num_layers=6, hidden_dim=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, MAX_SMILES_LEN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
        return x[:, 0, :]

def load_pretrained_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    
    filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
    
    if 'positional_encoding.pe' in filtered_checkpoint:
        pretrained_pe = filtered_checkpoint['positional_encoding.pe']
        model_pe = model_dict['positional_encoding.pe']
        
        if pretrained_pe.shape[0] < model_pe.shape[0]:
            model_pe[:pretrained_pe.shape[0], :] = pretrained_pe
            filtered_checkpoint['positional_encoding.pe'] = model_pe
    
    model.load_state_dict(filtered_checkpoint, strict=False)
    return model

class RLGateModel(nn.Module):
    
    def __init__(self, embed_dim=256, cell_feature_dim=CELL_FEATURE_DIM, 
                 gate_hidden=512, regression_hidden=1024):
        super().__init__()
        
        self.cell_projection = nn.Sequential(
            nn.Linear(cell_feature_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.BatchNorm1d(gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.BatchNorm1d(gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 2)
        )
        
        self.regression_network = nn.Sequential(
            nn.Linear(embed_dim, regression_hidden),
            nn.BatchNorm1d(regression_hidden),
            nn.ReLU(),
            nn.Linear(regression_hidden, regression_hidden),
            nn.BatchNorm1d(regression_hidden),
            nn.ReLU(),
            nn.Linear(regression_hidden, 1)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(embed_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )
    
    def forward(self, h_drug1, h_drug2, cell_features, train_mode=True, threshold=REGRESSION_THRESHOLD):
        h_cell = self.cell_projection(cell_features)
        
        combined = torch.cat([h_drug1, h_drug2, h_cell], dim=-1)
        fused = self.fusion_layer(combined)
        
        gate_logits = self.gate_network(fused)
        gate_probs = nn.functional.softmax(gate_logits, dim=-1)
        
        if train_mode:
            gate_action = torch.multinomial(gate_probs, 1).squeeze(-1)
        else:
            gate_action = torch.argmax(gate_probs, dim=-1)
        
        synergy_pred = self.regression_network(fused).squeeze(-1)
        
        return {
            'gate_probs': gate_probs,
            'gate_action': gate_action,
            'synergy_pred': synergy_pred,
            'value': self.value_network(fused).squeeze(-1)
        }

def compute_rewards(gate_action, label_coarse, synergy_pred, synergy_true, 
                   pos_weight=5.0, lambda_coarse=1.0, lambda_regression=0.5):
    R_coarse = torch.where(
        label_coarse == 1,
        pos_weight * (gate_action == label_coarse).float(),
        (gate_action == label_coarse).float()
    )
    
    regression_loss = nn.functional.huber_loss(
        synergy_pred, 
        synergy_true, 
        reduction='none',
        delta=1.0
    )
    R_regression = -regression_loss
    
    valid_mask = (gate_action == 1) & (label_coarse == 1)
    R_total = lambda_coarse * R_coarse
    R_total[valid_mask] += lambda_regression * R_regression[valid_mask]
    
    return R_coarse, R_regression, R_total

def evaluate_model(model, drug_model, data_loader, device):
    model.eval()
    drug_model.eval()
    
    coarse_probs = []
    coarse_true = []
    synergy_preds = []
    synergy_true = []
    valid_mask = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            h_drug1 = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            h_drug2 = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )
            outputs = model(
                h_drug1, h_drug2, 
                batch['cell_features'].to(device), 
                train_mode=False
            )
            
            coarse_probs.extend(outputs['gate_probs'][:, 1].cpu().numpy())
            coarse_true.extend(batch['label_coarse'].cpu().numpy())
            synergy_preds.extend(outputs['synergy_pred'].cpu().numpy())
            synergy_true.extend(batch['synergy'].cpu().numpy())
            valid_mask.extend((outputs['gate_action'].cpu() == 1).numpy())
    
    coarse_auc = roc_auc_score(coarse_true, coarse_probs)
    coarse_aupr = average_precision_score(coarse_true, coarse_probs)
    coarse_acc = accuracy_score(coarse_true, np.array(coarse_probs) > 0.5)
    
    valid_synergy_preds = np.array(synergy_preds)[valid_mask]
    valid_synergy_true = np.array(synergy_true)[valid_mask]
    num_valid = len(valid_synergy_preds)
    
    if num_valid > 0:
        msv = mean_squared_error(valid_synergy_true, valid_synergy_preds)
        rmse = np.sqrt(msv)
        
        residuals = valid_synergy_true - valid_synergy_preds
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        n = len(residuals)
        t_value = stats.t.ppf(0.975, n - 1)
        ci_low = mean_residual - t_value * (std_residual / np.sqrt(n))
        ci_high = mean_residual + t_value * (std_residual / np.sqrt(n))
        
        pcc = np.corrcoef(valid_synergy_true, valid_synergy_preds)[0, 1]
    else:
        msv = rmse = ci_low = ci_high = pcc = 0.0
    
    return {
        'coarse_auc': coarse_auc,
        'coarse_aupr': coarse_aupr,
        'coarse_acc': coarse_acc,
        'synergy_msv': msv,
        'synergy_rmse': rmse,
        'synergy_ci_low': ci_low,
        'synergy_ci_high': ci_high,
        'synergy_pcc': pcc,
        'num_valid_samples': num_valid
    }

def compute_gate_activation_rate(model, drug_model, data_loader, device):
    model.eval()
    drug_model.eval()
    
    total_samples = 0
    activated_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            h_drug1 = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            h_drug2 = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )
            outputs = model(
                h_drug1, h_drug2,
                batch['cell_features'].to(device),
                train_mode=False
            )
            
            total_samples += len(batch['label_coarse'])
            activated_samples += (outputs['gate_action'] == 1).sum().item()
    
    return activated_samples / total_samples if total_samples > 0 else 0.0

def print_metrics(metrics, prefix=""):
    print(f"\n{prefix}Coarse-grained Metrics:")
    print(f"AUC: {metrics['coarse_auc']:.4f}")
    print(f"AUPR: {metrics['coarse_aupr']:.4f}")
    print(f"Accuracy: {metrics['coarse_acc']:.4f}")
    
    print(f"\n{prefix}Regression Metrics:")
    print(f"MSV: {metrics['synergy_msv']:.4f}")
    print(f"RMSE: {metrics['synergy_rmse']:.4f}")
    print(f"95% Confidence Interval: [{metrics['synergy_ci_low']:.4f}, {metrics['synergy_ci_high']:.4f}]")
    print(f"PCC: {metrics['synergy_pcc']:.4f}")
    print(f"Number of Gate Activated Samples: {metrics['num_valid_samples']}")

def train_model(model, drug_model, train_loader, optimizer, total_epochs, device, 
               lambda_coarse_schedule, lambda_regression_schedule, pos_weight,
               test_loader, test_idx, drug1_smiles_list, drug2_smiles_list, 
               cell_features_list, output_path):
    drug_model.eval()
    model.train()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs, 
        eta_min=1e-6
    )
    
    best_metrics = None
    
    for epoch in range(total_epochs):
        lambda_coarse = lambda_coarse_schedule[epoch] if epoch < len(lambda_coarse_schedule) else lambda_coarse_schedule[-1]
        lambda_regression = lambda_regression_schedule[epoch] if epoch < len(lambda_regression_schedule) else lambda_regression_schedule[-1]
        
        total_loss = 0.0
        coarse_acc = []
        regression_losses = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}'):
            h_drug1 = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            h_drug2 = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )
            outputs = model(
                h_drug1, h_drug2,
                batch['cell_features'].to(device)
            )
            
            R_coarse, R_regression, R_total = compute_rewards(
                outputs['gate_action'],
                batch['label_coarse'].to(device),
                outputs['synergy_pred'],
                batch['synergy'].to(device),
                pos_weight=pos_weight,
                lambda_coarse=lambda_coarse,
                lambda_regression=lambda_regression
            )
            
            log_probs = torch.log(outputs['gate_probs'].gather(1, outputs['gate_action'].unsqueeze(1)).squeeze(1))
            advantage = R_total - outputs['value'].detach()
            policy_loss = -(log_probs * advantage).mean()
            
            value_loss = nn.functional.mse_loss(outputs['value'], R_total)
            
            regression_loss = nn.functional.huber_loss(
                outputs['synergy_pred'][batch['label_coarse'].to(device) == 1],
                batch['synergy'].to(device)[batch['label_coarse'].to(device) == 1],
                delta=1.0
            )
            
            loss = policy_loss + 0.5 * value_loss + lambda_regression * regression_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            coarse_acc.append(R_coarse.mean().item())
            regression_losses.append(regression_loss.item())
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} Training Results:")
        print(f"Total Loss: {total_loss/len(train_loader):.4f}")
        print(f"Coarse Accuracy: {np.mean(coarse_acc):.4f}")
        print(f"Regression Loss: {np.mean(regression_losses):.4f}")
        
        test_metrics = evaluate_model(model, drug_model, test_loader, device)
        print_metrics(test_metrics, prefix="Test Set ")
        
        gate_activation_rate = compute_gate_activation_rate(model, drug_model, test_loader, device)
        print(f"Gate Activation Rate: {gate_activation_rate:.4f}")
        
        if best_metrics is None or test_metrics['synergy_pcc'] > best_metrics['synergy_pcc']:
            best_metrics = test_metrics
            torch.save(model.state_dict(), "best_model.pth")
    
    output_path_final = f"{os.path.splitext(output_path)[0]}_final.csv"
    
    save_predictions(
        model, drug_model, test_loader, test_idx,
        drug1_smiles_list, drug2_smiles_list, cell_features_list,
        output_path_final, device
    )
    
    return best_metrics

def save_predictions(model, drug_model, data_loader, test_idx, drug1_smiles_list, 
                   drug2_smiles_list, cell_features_list, output_path, device):
    model.eval()
    drug_model.eval()
    
    results = []
    with torch.no_grad():
        for batch in data_loader:
            h_drug1 = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            h_drug2 = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )
            outputs = model(
                h_drug1, h_drug2,
                batch['cell_features'].to(device),
                train_mode=False
            )
            
            for i in range(len(batch['label_coarse'])):
                results.append({
                    'drug1_SMILES': drug1_smiles_list[test_idx[i]],
                    'drug2_SMILES': drug2_smiles_list[test_idx[i]],
                    'cell_features': str(cell_features_list[test_idx[i]].tolist()),
                    'true_coarse': int(batch['label_coarse'][i]),
                    'true_synergy': float(batch['synergy'][i]),
                    'pred_coarse_prob': float(outputs['gate_probs'][i, 1]),
                    'pred_coarse': int(outputs['gate_action'][i]),
                    'pred_synergy': float(outputs['synergy_pred'][i])
                })
    
    pd.DataFrame(results).to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description='Drug-Drug Interaction Prediction Model')
    parser.add_argument('--data', type=str, default='DDI_oneil_moa.csv', help='Dataset path')
    parser.add_argument('--drug_model', type=str, default='best_drug_transformer_BN.pth', help='Pre-trained drug model path')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--pos_weight', type=float, default=5.0, help='Positive sample weight')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file path')
    parser.add_argument('--no_balance', action='store_true', help='Do not use data balancing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                        help='Test set ratio')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data...")
    df = pd.read_csv(args.data)
    drug1_smiles_list = df['drug1_smiles'].tolist()
    drug2_smiles_list = df['drug2_smiles'].tolist()
    cell_features_list = [parse_cell_features(cf) for cf in df['cell_features'].tolist()]
    label_coarse = df['label_coarse'].values
    synergy_values = df['synergy'].values
    
    cell_mean, cell_std = calculate_cell_feature_stats(cell_features_list)
    
    print("\n===== Starting Random Split for Training and Test Sets =====")
    all_indices = np.arange(len(label_coarse))
    
    test_size = int(len(all_indices) * args.test_ratio)
    test_idx = np.random.choice(all_indices, size=test_size, replace=False)
    
    train_size = int(len(all_indices) * (1 - args.test_ratio))
    train_idx = np.random.choice(all_indices, size=train_size, replace=False)
    
    print(f"Training set size: {train_size}, Test set size: {test_size}")
    
    train_dataset = DrugDrugInteractionDataset(
        [drug1_smiles_list[i] for i in train_idx],
        [drug2_smiles_list[i] for i in train_idx],
        [cell_features_list[i] for i in train_idx],
        label_coarse[train_idx],
        synergy_values[train_idx],
        balance_data=not args.no_balance,
        random_state=42,
        cell_mean=cell_mean,
        cell_std=cell_std
    )
    
    test_dataset = DrugDrugInteractionDataset(
        [drug1_smiles_list[i] for i in test_idx],
        [drug2_smiles_list[i] for i in test_idx],
        [cell_features_list[i] for i in test_idx],
        label_coarse[test_idx],
        synergy_values[test_idx],
        balance_data=False,
        cell_mean=cell_mean,
        cell_std=cell_std
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    drug_model = DrugTransformer().to(device)
    drug_model = load_pretrained_model(drug_model, args.drug_model, device)
    drug_model.eval()
    
    model = RLGateModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    
    lambda_coarse_schedule = [2.0] * 20 + [1.5] * 20 + [1.0] * (args.epochs - 40)
    lambda_regression_schedule = [0.3] * 20 + [0.5] * 20 + [0.7] * (args.epochs - 40)
    
    print("\n===== Starting Model Training =====")
    metrics = train_model(
        model, drug_model, train_loader, optimizer, args.epochs, device,
        lambda_coarse_schedule, lambda_regression_schedule, args.pos_weight,
        test_loader, test_idx, drug1_smiles_list, drug2_smiles_list,
        cell_features_list, args.output
    )
    
    print("\n===== Final Results =====")
    print_metrics(metrics, prefix="Test Set ")

if __name__ == "__main__":
    main()