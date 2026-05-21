import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import math
import argparse
from sklearn.preprocessing import StandardScaler

SMILES_CHARS = [' ', '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']', 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(SMILES_CHARS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(SMILES_CHARS)}

MAX_LEN = 128


class DrugDataset(Dataset):

    def __init__(self, smiles_list, mask_prob=0.15, max_len=MAX_LEN, is_pretrain=True, scaler=None):
        self.smiles_list = smiles_list
        self.mask_prob = mask_prob
        self.max_len = max_len
        self.is_pretrain = is_pretrain
        self.scaler = scaler

        if is_pretrain:
            self.properties = self._calculate_properties()
            if self.scaler is None:
                self._fit_scaler()
            self._scale_properties()

    def _calculate_properties(self):
        properties = []
        for smiles in tqdm(self.smiles_list, desc="Calculating physicochemical properties"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    props = {
                        'mol_weight': 0.0,
                        'logp': 0.0,
                        'tpsa': 0.0,
                        'hba': 0,
                        'hbd': 0
                    }
                else:
                    props = {
                        'mol_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'tpsa': rdMolDescriptors.CalcTPSA(mol),
                        'hba': rdMolDescriptors.CalcNumHBA(mol),
                        'hbd': rdMolDescriptors.CalcNumHBD(mol)
                    }
                properties.append(props)
            except:
                properties.append({
                    'mol_weight': 0.0,
                    'logp': 0.0,
                    'tpsa': 0.0,
                    'hba': 0,
                    'hbd': 0
                })
        return properties

    def _fit_scaler(self):
        all_props = []
        for props in self.properties:
            all_props.append([
                props['mol_weight'],
                props['logp'],
                props['tpsa'],
                props['hba'],
                props['hbd']
            ])
        
        self.scaler = StandardScaler()
        self.scaler.fit(all_props)
        
        print("Physicochemical property standardization parameters:")
        print(f"Mean: {self.scaler.mean_}")
        print(f"Standard deviation: {np.sqrt(self.scaler.var_)}")

    def _scale_properties(self):
        all_props = []
        for props in self.properties:
            all_props.append([
                props['mol_weight'],
                props['logp'],
                props['tpsa'],
                props['hba'],
                props['hbd']
            ])
        
        scaled_props = self.scaler.transform(all_props)
        
        for i, props in enumerate(scaled_props):
            self.properties[i] = {
                'mol_weight': props[0],
                'logp': props[1],
                'tpsa': props[2],
                'hba': props[3],
                'hbd': props[4]
            }

    def _smiles_to_idx(self, smiles):
        idx_seq = [CHAR_TO_IDX.get(c, 0) for c in smiles]
        return idx_seq

    def _mask_sequence(self, idx_seq):
        masked_seq = idx_seq.copy()
        mask_labels = []

        for i in range(len(masked_seq)):
            if random.random() < self.mask_prob:
                mask_labels.append(masked_seq[i])

                prob = random.random()
                if prob < 0.8:
                    masked_seq[i] = CHAR_TO_IDX[' ']
                elif prob < 0.9:
                    masked_seq[i] = random.randint(1, len(SMILES_CHARS) - 1)
            else:
                mask_labels.append(-1)

        return masked_seq, mask_labels

    def _get_bond_matrix(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((self.max_len, self.max_len), dtype=np.float32)

            bond_matrix = np.zeros((self.max_len, self.max_len), dtype=np.float32)

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                if i < self.max_len and j < self.max_len:
                    bond_matrix[i, j] = 1.0
                    bond_matrix[j, i] = 1.0

            return bond_matrix
        except:
            return np.zeros((self.max_len, self.max_len), dtype=np.float32)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        idx_seq = self._smiles_to_idx(smiles)

        if len(idx_seq) > self.max_len - 2:
            idx_seq = idx_seq[:self.max_len - 2]

        idx_seq = [CHAR_TO_IDX[' ']] + idx_seq + [CHAR_TO_IDX[' ']]
        original_seq = idx_seq.copy()

        masked_seq, mask_labels = self._mask_sequence(idx_seq)

        padding_length = self.max_len - len(masked_seq)
        masked_seq = masked_seq + [0] * padding_length
        original_seq = original_seq + [0] * padding_length
        mask_labels = mask_labels + [-1] * padding_length

        attention_mask = [1] * len(idx_seq) + [0] * padding_length

        item = {
            'input_ids': torch.tensor(masked_seq, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'original_ids': torch.tensor(original_seq, dtype=torch.long),
            'mask_labels': torch.tensor(mask_labels, dtype=torch.long),
            'smiles': smiles
        }

        if self.is_pretrain:
            props = self.properties[idx]
            item['mol_weight'] = torch.tensor(props['mol_weight'], dtype=torch.float)
            item['logp'] = torch.tensor(props['logp'], dtype=torch.float)
            item['tpsa'] = torch.tensor(props['tpsa'], dtype=torch.float)
            item['hba'] = torch.tensor(props['hba'], dtype=torch.float)
            item['hbd'] = torch.tensor(props['hbd'], dtype=torch.float)

            bond_matrix = self._get_bond_matrix(smiles)
            item['bond_matrix'] = torch.tensor(bond_matrix, dtype=torch.float)

        return item


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=MAX_LEN):
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

    def __init__(self, vocab_size=len(SMILES_CHARS), embed_dim=256, num_heads=8, num_layers=6, hidden_dim=1024):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )

        self.mol_weight_head = nn.Linear(embed_dim, 1)
        self.logp_head = nn.Linear(embed_dim, 1)
        self.tpsa_head = nn.Linear(embed_dim, 1)
        self.hba_head = nn.Linear(embed_dim, 1)
        self.hbd_head = nn.Linear(embed_dim, 1)

        self.bond_head = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids, attention_mask):
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())

        cls_repr = x[:, 0, :]

        mlm_logits = self.mlm_head(x)

        mol_weight = self.mol_weight_head(cls_repr).squeeze()
        logp = self.logp_head(cls_repr).squeeze()
        tpsa = self.tpsa_head(cls_repr).squeeze()
        hba = self.hba_head(cls_repr).squeeze()
        hbd = self.hbd_head(cls_repr).squeeze()

        batch_size, seq_len, embed_dim = x.shape

        x_expanded1 = x.unsqueeze(2).expand(batch_size, seq_len, seq_len, embed_dim)
        x_expanded2 = x.unsqueeze(1).expand(batch_size, seq_len, seq_len, embed_dim)

        pair_repr = torch.cat([x_expanded1, x_expanded2], dim=-1)

        bond_logits = self.bond_head(pair_repr).squeeze(-1)

        return {
            'mlm_logits': mlm_logits,
            'mol_weight': mol_weight,
            'logp': logp,
            'tpsa': tpsa,
            'hba': hba,
            'hbd': hbd,
            'bond_logits': bond_logits,
            'embedding': cls_repr
        }


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    property_criterion = nn.MSELoss()
    bond_criterion = nn.BCELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_mlm_loss = 0.0
        total_property_loss = 0.0
        total_bond_loss = 0.0

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mask_labels = batch['mask_labels'].to(device)
            mol_weight = batch['mol_weight'].to(device)
            logp = batch['logp'].to(device)
            tpsa = batch['tpsa'].to(device)
            hba = batch['hba'].to(device)
            hbd = batch['hbd'].to(device)
            bond_matrix = batch['bond_matrix'].to(device)

            outputs = model(input_ids, attention_mask)

            mlm_loss = mlm_criterion(
                outputs['mlm_logits'].view(-1, len(SMILES_CHARS)),
                mask_labels.view(-1)
            )

            property_loss = (
                    property_criterion(outputs['mol_weight'], mol_weight) +
                    property_criterion(outputs['logp'], logp) +
                    property_criterion(outputs['tpsa'], tpsa) +
                    property_criterion(outputs['hba'], hba) +
                    property_criterion(outputs['hbd'], hbd)
            )

            seq_len = input_ids.size(1)
            valid_mask = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
            valid_bond_matrix = bond_matrix * valid_mask
            valid_bond_logits = outputs['bond_logits'] * valid_mask

            bond_loss = bond_criterion(
                valid_bond_logits.view(-1),
                valid_bond_matrix.view(-1)
            )

            total_loss = mlm_loss + property_loss + bond_loss

            optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += total_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_property_loss += property_loss.item()
            total_bond_loss += bond_loss.item()

            train_progress.set_postfix({
                'mlm_loss': mlm_loss.item(),
                'property_loss': property_loss.item(),
                'bond_loss': bond_loss.item(),
                'total_loss': total_loss.item()
            })

        avg_train_loss = total_train_loss / len(train_loader)
        avg_mlm_loss = total_mlm_loss / len(train_loader)
        avg_property_loss = total_property_loss / len(train_loader)
        avg_bond_loss = total_bond_loss / len(train_loader)

        print(f'Epoch {epoch + 1}/{epochs} [Train] | MLM Loss: {avg_mlm_loss:.4f} | Property Loss: {avg_property_loss:.4f} | Bond Loss: {avg_bond_loss:.4f} | Total Loss: {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0.0
        total_val_mlm_loss = 0.0
        total_val_property_loss = 0.0
        total_val_bond_loss = 0.0

        val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                mask_labels = batch['mask_labels'].to(device)
                mol_weight = batch['mol_weight'].to(device)
                logp = batch['logp'].to(device)
                tpsa = batch['tpsa'].to(device)
                hba = batch['hba'].to(device)
                hbd = batch['hbd'].to(device)
                bond_matrix = batch['bond_matrix'].to(device)

                outputs = model(input_ids, attention_mask)

                mlm_loss = mlm_criterion(
                    outputs['mlm_logits'].view(-1, len(SMILES_CHARS)),
                    mask_labels.view(-1)
                )

                property_loss = (
                        property_criterion(outputs['mol_weight'], mol_weight) +
                        property_criterion(outputs['logp'], logp) +
                        property_criterion(outputs['tpsa'], tpsa) +
                        property_criterion(outputs['hba'], hba) +
                        property_criterion(outputs['hbd'], hbd)
                )

                seq_len = input_ids.size(1)
                valid_mask = attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1)
                valid_bond_matrix = bond_matrix * valid_mask
                valid_bond_logits = outputs['bond_logits'] * valid_mask

                bond_loss = bond_criterion(
                    valid_bond_logits.view(-1),
                    valid_bond_matrix.view(-1)
                )

                total_loss = mlm_loss + property_loss + bond_loss

                total_val_loss += total_loss.item()
                total_val_mlm_loss += mlm_loss.item()
                total_val_property_loss += property_loss.item()
                total_val_bond_loss += bond_loss.item()

                val_progress.set_postfix({
                    'mlm_loss': mlm_loss.item(),
                    'property_loss': property_loss.item(),
                    'bond_loss': bond_loss.item(),
                    'total_loss': total_loss.item()
                })

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mlm_loss = total_val_mlm_loss / len(val_loader)
        avg_val_property_loss = total_val_property_loss / len(val_loader)
        avg_val_bond_loss = total_val_bond_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{epochs} [Val] | MLM Loss: {avg_val_mlm_loss:.4f} | Property Loss: {avg_val_property_loss:.4f} | Bond Loss: {avg_val_bond_loss:.4f} | Total Loss: {avg_val_loss:.4f}')

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_drug_transformer_BN.pth')
            print('Model saved!')


def extract_embeddings(model, data_loader, device):
    model.eval()
    all_smiles = []
    all_embeddings = []

    progress = tqdm(data_loader, desc="Extracting drug embeddings")
    with torch.no_grad():
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            smiles = batch['smiles']

            outputs = model(input_ids, attention_mask)
            embeddings = outputs['embedding'].cpu().numpy()

            all_smiles.extend(smiles)
            all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)

    return all_smiles, all_embeddings


def save_embeddings_to_csv(smiles_list, embeddings, output_file):
    df = pd.DataFrame({
        'SMILES': smiles_list,
        'embedding': [','.join([f'{x:.6f}' for x in embed]) for embed in embeddings]
    })

    df.to_csv(output_file, index=False)
    print(f"Drug embeddings saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Drug Pre-training Model')
    parser.add_argument('--pretrain_data', default='DTI+moa.csv', help='Pre-training dataset')
    parser.add_argument('--extract_data', default='DTI+moa.csv', help='Dataset for embedding extraction')
    parser.add_argument('--output', default='drug_embeddings.csv', help='Output CSV file path for embeddings')
    parser.add_argument('--column', default='SMILES', help='Column name containing SMILES sequences')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_path', default='best_drug_transformer_BN.pth', help='Model save/load path')

    args = parser.parse_args()

    try:
        print(f"===== Starting pre-training phase, using dataset: {args.pretrain_data} =====")

        pretrain_df = pd.read_csv(args.pretrain_data)
        if args.column not in pretrain_df.columns:
            raise ValueError(f"Column '{args.column}' does not exist in pre-training data")

        pretrain_smiles = pretrain_df[args.column].tolist()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        pretrain_dataset = DrugDataset(pretrain_smiles)

        train_size = int(0.8 * len(pretrain_dataset))
        val_size = len(pretrain_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(pretrain_dataset, [train_size, val_size])

        train_dataset.scaler = pretrain_dataset.scaler
        val_dataset.dataset.scaler = pretrain_dataset.scaler

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = DrugTransformer().to(device)

        train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs
        )

        print(f"\n===== Starting feature extraction phase, using dataset: {args.extract_data} =====")

        extract_df = pd.read_csv(args.extract_data)
        if args.column not in extract_df.columns:
            raise ValueError(f"Column '{args.column}' does not exist in extraction data")

        extract_smiles = extract_df[args.column].tolist()

        extract_dataset = DrugDataset(extract_smiles, is_pretrain=False, scaler=pretrain_dataset.scaler)
        extract_loader = DataLoader(extract_dataset, batch_size=args.batch_size)

        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded: {args.model_path}")

        smiles, embeddings = extract_embeddings(model, extract_loader, device)

        save_embeddings_to_csv(smiles, embeddings, args.output)

        print("\n===== Processing complete =====")
        print(f"1. Pre-trained model saved to: {args.model_path}")
        print(f"2. Drug embeddings saved to: {args.output}")

    except Exception as e:
        print(f"Program execution error: {str(e)}")


if __name__ == "__main__":
    main()