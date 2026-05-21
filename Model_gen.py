import os
import math
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem import rdmolops
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_squared_error

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
                 label_coarse, synergy_values, drug3_smiles_list=None, balance_data=True, 
                 random_state=42, cell_mean=None, cell_std=None):
        self.drug1_smiles_list = drug1_smiles_list
        self.drug2_smiles_list = drug2_smiles_list
        self.cell_features_list = cell_features_list
        self.label_coarse = label_coarse.astype(np.int64)
        self.synergy_values = synergy_values
        self.drug3_smiles_list = drug3_smiles_list
        
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
            if self.drug3_smiles_list is not None:
                self.drug3_smiles_list = [self.drug3_smiles_list[i] for i in selected_indices]
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
        
        item = {
            'drug1_input_ids': torch.tensor(drug1_input_ids, dtype=torch.long),
            'drug1_attention_mask': torch.tensor(drug1_attention_mask, dtype=torch.long),
            'drug2_input_ids': torch.tensor(drug2_input_ids, dtype=torch.long),
            'drug2_attention_mask': torch.tensor(drug2_attention_mask, dtype=torch.long),
            'cell_features': torch.tensor(self.cell_features_list[idx], dtype=torch.float32),
            'label_coarse': torch.tensor(self.label_coarse[idx], dtype=torch.long),
            'synergy': torch.tensor(self.synergy_values[idx], dtype=torch.float32)
        }
        
        if self.drug3_smiles_list is not None:
            item['drug3_smiles'] = self.drug3_smiles_list[idx]
        
        return item

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

class DrugVAEGenerator(nn.Module):
    
    def __init__(self, drug_feature_dim=256, cell_feature_dim=CELL_FEATURE_DIM, 
                 latent_dim=256, max_len=MAX_SMILES_LEN, vocab_size=len(SMILES_CHARS)):
        super().__init__()
        self.input_dim = drug_feature_dim * 2 + cell_feature_dim + 1
        
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        self.seq_generator = nn.Linear(1024, max_len * vocab_size)
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder(z)
        seq_logits = self.seq_generator(h).view(-1, self.max_len, self.vocab_size)
        return seq_logits
    
    def forward(self, drug1_feature, drug2_feature, cell_feature, synergy_score):
        combined = torch.cat([drug1_feature, drug2_feature, cell_feature, synergy_score.unsqueeze(1)], dim=1)
        
        if combined.size(1) != self.input_dim:
            raise ValueError(f"Input feature dimension mismatch, expected {self.input_dim}, got {combined.size(1)}")
        
        mu, logvar = self.encode(combined)
        
        z = self.reparameterize(mu, logvar)
        
        z_with_synergy = torch.cat([z, synergy_score.unsqueeze(1)], dim=1)
        seq_logits = self.decode(z_with_synergy)
        
        return seq_logits, mu, logvar
    
    def generate(self, drug1_feature, drug2_feature, cell_feature, synergy_score, temperature=1.0):
        with torch.no_grad():
            combined = torch.cat([drug1_feature, drug2_feature, cell_feature, synergy_score.unsqueeze(1)], dim=1)
            
            mu, logvar = self.encode(combined)
            z = self.reparameterize(mu, logvar)
            
            z_with_synergy = torch.cat([z, synergy_score.unsqueeze(1)], dim=1)
            seq_logits = self.decode(z_with_synergy)
            
            seq_probs = torch.nn.functional.softmax(seq_logits / temperature, dim=-1)
            
            seq_indices = torch.multinomial(seq_probs.view(-1, self.vocab_size), 1).view(-1, self.max_len)
            
            return seq_indices, seq_probs

def vae_loss(recon_x, x, mu, logvar, kl_weight=0.001):
    recon_loss = F.cross_entropy(
        recon_x.view(-1, len(SMILES_CHARS)), 
        x.view(-1), 
        reduction='sum'
    )
    
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_weight * kl_div

def smiles_to_indices(smiles, max_len=MAX_SMILES_LEN):
    indices = [CHAR_TO_IDX.get(c, 0) for c in smiles]
    indices = indices[:max_len]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long)

def indices_to_smiles(indices):
    return ''.join([SMILES_CHARS[idx] for idx in indices if idx != 0]).strip()

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
    
    if len(np.unique(coarse_true)) > 1:
        coarse_auc = roc_auc_score(coarse_true, coarse_probs)
        coarse_aupr = average_precision_score(coarse_true, coarse_probs)
    else:
        coarse_auc = coarse_aupr = 0.0
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

def evaluate_vae_model(vae_model, drug_model, rl_model, data_loader, device):
    vae_model.eval()
    drug_model.eval()
    rl_model.eval()
    
    total_samples = 0
    correct_sequences = 0
    correct_tokens = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating VAE'):
            drug1_feature = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            
            drug2_feature = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )
            
            cell_feature = batch['cell_features'].to(device)
            
            target_synergy = batch['synergy'].to(device)
            
            drug3_smiles = batch['drug3_smiles']
            drug3_indices = torch.stack([smiles_to_indices(s) for s in drug3_smiles]).to(device)
            
            generated_indices, _ = vae_model.generate(
                drug1_feature, drug2_feature, cell_feature, target_synergy
            )
            
            batch_size = drug3_indices.size(0)
            total_samples += batch_size
            
            for i in range(batch_size):
                if torch.all(generated_indices[i] == drug3_indices[i]):
                    correct_sequences += 1
            
            correct_tokens += (generated_indices == drug3_indices).sum().item()
            total_tokens += drug3_indices.numel()
    
    sequence_accuracy = correct_sequences / total_samples if total_samples > 0 else 0.0
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return {
        'sequence_accuracy': sequence_accuracy,
        'token_accuracy': token_accuracy
    }

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

def train_two_stage_model(drug_model, rl_model, vae_model, train_loader, test_loader, 
                          rl_optimizer, vae_optimizer, rl_epochs, vae_epochs, device, 
                          lambda_coarse_schedule, lambda_regression_schedule, pos_weight,
                          drug1_smiles_list, drug2_smiles_list, cell_features_list, 
                          test_idx, output_path):
    rl_best_metrics = None
    vae_best_metrics = None
    
    print(f"Two-stage training strategy:")
    print(f"Stage 1: Train RL synergy prediction model - {rl_epochs} epochs")
    print(f"Stage 2: Train VAE drug generation model - {vae_epochs} epochs")
    
    print("\n===== Starting Stage 1 Training: Synergy Prediction =====")
    drug_model.eval()
    rl_model.train()
    
    rl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        rl_optimizer, 
        T_max=rl_epochs, 
        eta_min=1e-6
    )
    
    for epoch in range(rl_epochs):
        lambda_coarse = lambda_coarse_schedule[epoch] if epoch < len(lambda_coarse_schedule) else lambda_coarse_schedule[-1]
        lambda_regression = lambda_regression_schedule[epoch] if epoch < len(lambda_regression_schedule) else lambda_regression_schedule[-1]
        
        total_loss = 0.0
        coarse_acc = []
        regression_losses = []
        
        for batch in tqdm(train_loader, desc=f'RL Training Epoch {epoch+1}/{rl_epochs}'):
            h_drug1 = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            h_drug2 = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )
            outputs = rl_model(
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
                outputs['synergy_pred'], 
                batch['synergy'].to(device),
                delta=1.0
            )
            
            loss = policy_loss + 0.5 * value_loss + regression_loss
            
            rl_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rl_model.parameters(), 1.0)
            rl_optimizer.step()
            
            total_loss += loss.item()
            coarse_acc.append((outputs['gate_action'] == batch['label_coarse'].to(device)).float().mean().item())
            regression_losses.append(regression_loss.item())
        
        avg_loss = total_loss / len(train_loader)
        avg_coarse_acc = np.mean(coarse_acc)
        avg_regression_loss = np.mean(regression_losses)
        
        print(f"Epoch {epoch+1}/{rl_epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Gate Accuracy: {avg_coarse_acc:.4f}")
        print(f"Regression Loss: {avg_regression_loss:.4f}")
        print(f"RL Learning Rate: {rl_scheduler.get_last_lr()[0]:.8f}")
        
        if (epoch + 1) % 1 == 0:
            metrics = evaluate_model(rl_model, drug_model, test_loader, device)
            print_metrics(metrics, prefix="Test Set ")
            
            if rl_best_metrics is None or metrics['synergy_rmse'] < rl_best_metrics['synergy_rmse']:
                rl_best_metrics = metrics
                torch.save(rl_model.state_dict(), f"{output_path}/best_rl_model.pth")
                print(f"Saving best RL model - epoch {epoch+1} - RMSE: {metrics['synergy_rmse']:.4f}")
        
        rl_scheduler.step()
    
    print(f"Loading best RL model - RMSE: {rl_best_metrics['synergy_rmse']:.4f}")
    rl_model.load_state_dict(torch.load(f"{output_path}/best_rl_model.pth"))
    
    print("\n===== Starting Stage 2 Training: Drug Generation Model =====")
    
    print("Loading drug generation dataset...")
    ddi_data = pd.read_csv('DDI_oneil_paired.csv')
    
    drug1_smiles_gen = ddi_data['drug1_smiles'].tolist()
    drug2_smiles_gen = ddi_data['drug2_smiles'].tolist()
    drug3_smiles_gen = ddi_data['drug3_smiles'].tolist()
    cell_features_gen = ddi_data['cell_features'].tolist()
    synergy_values_gen = ddi_data['synergy2'].astype(float).tolist()
    
    cell_features_gen = [parse_cell_features(cf) for cf in cell_features_gen]
    
    train_size = int(0.8 * len(drug1_smiles_gen))
    indices = list(range(len(drug1_smiles_gen)))
    np.random.shuffle(indices)
    train_idx_gen, test_idx_gen = indices[:train_size], indices[train_size:]
    
    train_cell_features = [cell_features_gen[i] for i in train_idx_gen]
    cell_mean, cell_std = calculate_cell_feature_stats(train_cell_features)
    
    vae_train_dataset = DrugDrugInteractionDataset(
        [drug1_smiles_gen[i] for i in train_idx_gen],
        [drug2_smiles_gen[i] for i in train_idx_gen],
        [cell_features_gen[i] for i in train_idx_gen],
        np.zeros(len(train_idx_gen)),
        np.array([synergy_values_gen[i] for i in train_idx_gen]),
        [drug3_smiles_gen[i] for i in train_idx_gen],
        balance_data=False,
        cell_mean=cell_mean,
        cell_std=cell_std
    )
    
    vae_test_dataset = DrugDrugInteractionDataset(
        [drug1_smiles_gen[i] for i in test_idx_gen],
        [drug2_smiles_gen[i] for i in test_idx_gen],
        [cell_features_gen[i] for i in test_idx_gen],
        np.zeros(len(test_idx_gen)),
        np.array([synergy_values_gen[i] for i in test_idx_gen]),
        [drug3_smiles_gen[i] for i in test_idx_gen],
        balance_data=False,
        cell_mean=cell_mean,
        cell_std=cell_std
    )
    
    vae_train_loader = DataLoader(vae_train_dataset, batch_size=64, shuffle=True)
    vae_test_loader = DataLoader(vae_test_dataset, batch_size=64, shuffle=False)
    
    vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        vae_optimizer, 
        T_max=vae_epochs, 
        eta_min=1e-6
    )
    
    for param in rl_model.parameters():
        param.requires_grad = False
    
    drug_model.eval()
    rl_model.eval()
    vae_model.train()
    
    for epoch in range(vae_epochs):
        total_loss = 0.0
        recon_losses = []
        kl_losses = []
        
        for batch in tqdm(vae_train_loader, desc=f'VAE Training Epoch {epoch+1}/{vae_epochs}'):
            drug1_feature = drug_model(
                batch['drug1_input_ids'].to(device),
                batch['drug1_attention_mask'].to(device)
            )
            
            drug2_feature = drug_model(
                batch['drug2_input_ids'].to(device),
                batch['drug2_attention_mask'].to(device)
            )
            
            cell_feature = batch['cell_features'].to(device)
            
            target_synergy = batch['synergy'].to(device)
            
            drug3_smiles = batch['drug3_smiles']
            drug3_indices = torch.stack([smiles_to_indices(s) for s in drug3_smiles]).to(device)
            
            seq_logits, mu, logvar = vae_model(
                drug1_feature, drug2_feature, cell_feature, target_synergy
            )
            
            kl_weight = min(1.0, epoch / 10.0)
            loss = vae_loss(seq_logits, drug3_indices, mu, logvar, kl_weight=kl_weight)
            
            recon_loss = F.cross_entropy(
                seq_logits.view(-1, len(SMILES_CHARS)), 
                drug3_indices.view(-1), 
                reduction='mean'
            )
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            vae_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 1.0)
            vae_optimizer.step()
            
            total_loss += loss.item()
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_div.item())
        
        avg_loss = total_loss / len(vae_train_loader)
        avg_recon_loss = np.mean(recon_losses)
        avg_kl_loss = np.mean(kl_losses)
        
        print(f"Epoch {epoch+1}/{vae_epochs}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"KL Divergence: {avg_kl_loss:.4f}")
        print(f"VAE Learning Rate: {vae_scheduler.get_last_lr()[0]:.8f}")
        
        if (epoch + 1) % 5 == 0:
            metrics = evaluate_vae_model(vae_model, drug_model, rl_model, vae_test_loader, device)
            print(f"Test Set Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
            print(f"Test Set Token Accuracy: {metrics['token_accuracy']:.4f}")
            
            if vae_best_metrics is None or metrics['sequence_accuracy'] > vae_best_metrics['sequence_accuracy']:
                vae_best_metrics = metrics
                torch.save(vae_model.state_dict(), f"{output_path}/best_vae_model.pth")
                print(f"Saving best VAE model - epoch {epoch+1}")
        
        vae_scheduler.step()
    
    print("\n===== Final Evaluation =====")
    
    print("\nEvaluating Synergy Prediction Model:")
    metrics = evaluate_model(rl_model, drug_model, test_loader, device)
    print_metrics(metrics, prefix="Test Set ")
    
    print("\nEvaluating Drug Generation Model:")
    metrics = evaluate_vae_model(vae_model, drug_model, rl_model, vae_test_loader, device)
    print(f"Sequence Accuracy: {metrics['sequence_accuracy']:.4f}")
    print(f"Token Accuracy: {metrics['token_accuracy']:.4f}")
    
    torch.save(drug_model.state_dict(), f"{output_path}/final_drug_model.pth")
    torch.save(rl_model.state_dict(), f"{output_path}/final_rl_model.pth")
    torch.save(vae_model.state_dict(), f"{output_path}/final_vae_model.pth")
    
    print(f"\nModel training completed, all models saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Drug Synergy Prediction and Generation Model')
    parser.add_argument('--synergy_data_path', type=str, default='DDI_oneil_moa.csv', help='Synergy prediction dataset path')
    parser.add_argument('--generation_data_path', type=str, default='DDI_oneil_paired.csv', help='Drug generation dataset path')
    parser.add_argument('--output_path', type=str, default='saved_models', help='Model save path')
    parser.add_argument('--pretrained_drug_model', type=str, default='best_drug_transformer_BN.pth', help='Pre-trained drug model path')
    parser.add_argument('--pretrained_rl_model', type=str, default=None, help='Pre-trained RL model path')
    parser.add_argument('--pretrained_vae_model', type=str, default=None, help='Pre-trained VAE model path')
    parser.add_argument('--rl_epochs', type=int, default=30, help='RL model training epochs')
    parser.add_argument('--vae_epochs', type=int, default=50, help='VAE model training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_rl', type=float, default=1e-4, help='RL model learning rate')
    parser.add_argument('--lr_vae', type=float, default=1e-4, help='VAE model learning rate')
    parser.add_argument('--pos_weight', type=float, default=5.0, help='Positive sample weight')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    print("Loading synergy prediction data...")
    synergy_data = pd.read_csv(args.synergy_data_path)
    
    drug1_smiles_list = synergy_data['drug1_smiles'].tolist()
    drug2_smiles_list = synergy_data['drug2_smiles'].tolist()
    cell_features_list = synergy_data['cell_features'].tolist()
    label_coarse = synergy_data['label_coarse'].astype(np.int64).values
    synergy_values = synergy_data['synergy'].astype(float).values
    
    cell_features_list = [parse_cell_features(cf) for cf in cell_features_list]
    
    print(f"Randomly splitting dataset, test ratio: {args.test_ratio}")
    all_indices = np.arange(len(synergy_data))
    
    test_size = int(len(all_indices) * args.test_ratio)
    test_idx = np.random.choice(all_indices, size=test_size, replace=False)
    
    train_size = int(len(all_indices) * (1 - args.test_ratio))
    train_idx = np.random.choice(all_indices, size=train_size, replace=False)
    
    train_cell_features = [cell_features_list[i] for i in train_idx]
    cell_mean, cell_std = calculate_cell_feature_stats(train_cell_features)
    
    train_dataset = DrugDrugInteractionDataset(
        [drug1_smiles_list[i] for i in train_idx],
        [drug2_smiles_list[i] for i in train_idx],
        [cell_features_list[i] for i in train_idx],
        label_coarse[train_idx],
        synergy_values[train_idx],
        balance_data=True,
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
    
    print(f"Training set samples: {len(train_dataset)}")
    print(f"Test set samples: {len(test_dataset)}")
    print(f"Training set positive ratio: {train_dataset.label_coarse.mean():.2f}")
    print(f"Test set positive ratio: {test_dataset.label_coarse.mean():.2f}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Initializing models...")
    drug_model = DrugTransformer()
    rl_model = RLGateModel()
    vae_model = DrugVAEGenerator(drug_feature_dim=256, cell_feature_dim=CELL_FEATURE_DIM)
    
    if args.pretrained_drug_model:
        print(f"Loading pre-trained drug model: {args.pretrained_drug_model}")
        drug_model = load_pretrained_model(drug_model, args.pretrained_drug_model, args.device)
    
    if args.pretrained_rl_model:
        print(f"Loading pre-trained RL model: {args.pretrained_rl_model}")
        rl_model.load_state_dict(torch.load(args.pretrained_rl_model, map_location=args.device))
    
    if args.pretrained_vae_model:
        print(f"Loading pre-trained VAE model: {args.pretrained_vae_model}")
        vae_model.load_state_dict(torch.load(args.pretrained_vae_model, map_location=args.device))
    
    device = torch.device(args.device)
    drug_model.to(device)
    rl_model.to(device)
    vae_model.to(device)
    
    rl_optimizer = optim.Adam(rl_model.parameters(), lr=args.lr_rl)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=args.lr_vae)
    
    lambda_coarse_schedule = [1.0] * args.rl_epochs
    lambda_regression_schedule = [0.5] * args.rl_epochs
    
    print("Starting training...")
    train_two_stage_model(
        drug_model, rl_model, vae_model, train_loader, test_loader,
        rl_optimizer, vae_optimizer, args.rl_epochs, args.vae_epochs, device,
        lambda_coarse_schedule, lambda_regression_schedule, args.pos_weight,
        drug1_smiles_list, drug2_smiles_list, cell_features_list,
        test_idx, args.output_path
    )

if __name__ == "__main__":
    main()