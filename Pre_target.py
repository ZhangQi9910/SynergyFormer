import pandas as pd
import numpy as np
from collections import Counter
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def standardize_features(df, feature_names):
    for feature in feature_names:
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / (std if std > 0 else 1)
        df.attrs[f'{feature}_mean'] = mean
        df.attrs[f'{feature}_std'] = std
    return df

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

DIPEPTIDES = [aa1 + aa2 for aa1 in AA_PROPERTIES.keys() for aa2 in AA_PROPERTIES.keys()]

AA_DICT = {aa: idx + 1 for idx, aa in enumerate(sorted(AA_PROPERTIES.keys()))}
AA_DICT['[PAD]'] = 0
AA_DICT['[CLS]'] = len(AA_PROPERTIES) + 1
AA_DICT['[MASK]'] = len(AA_PROPERTIES) + 2


def calculate_hydrophobicity(sequence):
    return sum(AA_PROPERTIES.get(aa, {'hydrophobicity': 0})['hydrophobicity'] for aa in sequence) / len(sequence)


def calculate_polarity(sequence):
    return sum(AA_PROPERTIES.get(aa, {'polarity': 0})['polarity'] for aa in sequence) / len(sequence)


def calculate_volume(sequence):
    return sum(AA_PROPERTIES.get(aa, {'volume': 0})['volume'] for aa in sequence)


def calculate_molecular_weight(sequence):
    return sum(AA_PROPERTIES.get(aa, {'mw': 0})['mw'] for aa in sequence) - (len(sequence) - 1) * 18.02


def calculate_dipeptide_composition(sequence):
    dipeptide_counts = Counter([sequence[i:i + 2] for i in range(len(sequence) - 1)])
    return {dipeptide: dipeptide_counts.get(dipeptide, 0) / (len(sequence) - 1) for dipeptide in DIPEPTIDES}


def extract_features(sequence):
    features = {
        'hydrophobicity': calculate_hydrophobicity(sequence),
        'polarity': calculate_polarity(sequence),
        'volume': calculate_volume(sequence),
        'molecular_weight': calculate_molecular_weight(sequence)
    }

    dipeptide_features = calculate_dipeptide_composition(sequence)
    features.update(dipeptide_features)

    return features


def process_csv(input_file, output_file, fasta_column='fasta'):
    df = pd.read_csv(input_file)

    if fasta_column not in df.columns:
        raise ValueError(f"Column '{fasta_column}' does not exist in the CSV file")

    all_features = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting traditional features for {input_file}"):
        sequence = row[fasta_column]
        try:
            features = extract_features(sequence)
            all_features.append(features)
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            all_features.append({})

    features_df = pd.DataFrame(all_features)

    result_df = pd.concat([df, features_df], axis=1)
    
    if 'volume' in result_df.columns and 'molecular_weight' in result_df.columns:
        result_df = standardize_features(result_df, ['volume', 'molecular_weight'])

    result_df.to_csv(output_file, index=False)
    print(f"Feature extraction complete, results saved to {output_file}")

    return result_df


class ProteinDataset(Dataset):

    def __init__(self, sequences, features_df=None, max_len=512, is_pretrain=True):
        self.sequences = sequences
        self.features_df = features_df
        self.max_len = max_len
        self.is_pretrain = is_pretrain

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        seq_indices = [AA_DICT['[CLS]']] + [AA_DICT.get(aa, 0) for aa in sequence]

        if len(seq_indices) > self.max_len:
            seq_indices = seq_indices[:self.max_len]
        else:
            seq_indices = seq_indices + [AA_DICT['[PAD]']] * (self.max_len - len(seq_indices))

        item = {
            'input_ids': torch.tensor(seq_indices, dtype=torch.long),
            'attention_mask': torch.tensor([1 if i != AA_DICT['[PAD]'] else 0 for i in seq_indices], dtype=torch.long),
            'sequence': sequence
        }

        if self.is_pretrain and self.features_df is not None:
            row = self.features_df.iloc[idx]
            item['hydrophobicity'] = torch.tensor(row['hydrophobicity'], dtype=torch.float)
            item['polarity'] = torch.tensor(row['polarity'], dtype=torch.float)
            item['volume'] = torch.tensor(row['volume'], dtype=torch.float)
            item['molecular_weight'] = torch.tensor(row['molecular_weight'], dtype=torch.float)
            item['dipeptide'] = torch.tensor([row[dp] for dp in DIPEPTIDES], dtype=torch.float)

        return item


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]


class ProteinTransformer(nn.Module):

    def __init__(self, vocab_size=24, embed_dim=256, num_heads=8, num_layers=6, hidden_dim=1024, max_len=512):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.hydrophobicity_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.polarity_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.volume_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.molecular_weight_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.dipeptide_head = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(DIPEPTIDES))
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

        hydrophobicity = self.hydrophobicity_head(cls_repr).squeeze()
        polarity = self.polarity_head(cls_repr).squeeze()
        volume = self.volume_head(cls_repr).squeeze()
        molecular_weight = self.molecular_weight_head(cls_repr).squeeze()
        dipeptide = self.dipeptide_head(cls_repr)

        return {
            'hydrophobicity': hydrophobicity,
            'polarity': polarity,
            'volume': volume,
            'molecular_weight': molecular_weight,
            'dipeptide': dipeptide,
            'embedding': cls_repr
        }


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4):
    mse_loss = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_hydro_loss = 0.0
        train_polar_loss = 0.0
        train_volume_loss = 0.0
        train_weight_loss = 0.0
        train_dipeptide_loss = 0.0

        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hydrophobicity = batch['hydrophobicity'].to(device)
            polarity = batch['polarity'].to(device)
            volume = batch['volume'].to(device)
            molecular_weight = batch['molecular_weight'].to(device)
            dipeptide = batch['dipeptide'].to(device)

            outputs = model(input_ids, attention_mask)

            loss_hydro = mse_loss(outputs['hydrophobicity'], hydrophobicity)
            loss_polar = mse_loss(outputs['polarity'], polarity)
            loss_volume = mse_loss(outputs['volume'], volume)
            loss_weight = mse_loss(outputs['molecular_weight'], molecular_weight)
            loss_dipeptide = mse_loss(outputs['dipeptide'], dipeptide)

            total_loss = loss_hydro + loss_polar + loss_volume + loss_weight + loss_dipeptide

            optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += total_loss.item()
            train_hydro_loss += loss_hydro.item()
            train_polar_loss += loss_polar.item()
            train_volume_loss += loss_volume.item()
            train_weight_loss += loss_weight.item()
            train_dipeptide_loss += loss_dipeptide.item()

            train_progress.set_postfix({
                'total_loss': total_loss.item(),
                'hydro_loss': loss_hydro.item(),
                'polar_loss': loss_polar.item(),
                'volume_loss': loss_volume.item(),
                'weight_loss': loss_weight.item(),
                'dipeptide_loss': loss_dipeptide.item()
            })

        avg_train_total_loss = total_train_loss / len(train_loader)
        avg_train_hydro_loss = train_hydro_loss / len(train_loader)
        avg_train_polar_loss = train_polar_loss / len(train_loader)
        avg_train_volume_loss = train_volume_loss / len(train_loader)
        avg_train_weight_loss = train_weight_loss / len(train_loader)
        avg_train_dipeptide_loss = train_dipeptide_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        val_hydro_loss = 0.0
        val_polar_loss = 0.0
        val_volume_loss = 0.0
        val_weight_loss = 0.0
        val_dipeptide_loss = 0.0

        val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                hydrophobicity = batch['hydrophobicity'].to(device)
                polarity = batch['polarity'].to(device)
                volume = batch['volume'].to(device)
                molecular_weight = batch['molecular_weight'].to(device)
                dipeptide = batch['dipeptide'].to(device)

                outputs = model(input_ids, attention_mask)

                loss_hydro = mse_loss(outputs['hydrophobicity'], hydrophobicity)
                loss_polar = mse_loss(outputs['polarity'], polarity)
                loss_volume = mse_loss(outputs['volume'], volume)
                loss_weight = mse_loss(outputs['molecular_weight'], molecular_weight)
                loss_dipeptide = mse_loss(outputs['dipeptide'], dipeptide)

                total_loss = loss_hydro + loss_polar + loss_volume + loss_weight + loss_dipeptide

                total_val_loss += total_loss.item()
                val_hydro_loss += loss_hydro.item()
                val_polar_loss += loss_polar.item()
                val_volume_loss += loss_volume.item()
                val_weight_loss += loss_weight.item()
                val_dipeptide_loss += loss_dipeptide.item()

                val_progress.set_postfix({
                    'total_loss': total_loss.item(),
                    'hydro_loss': loss_hydro.item(),
                    'polar_loss': loss_polar.item(),
                    'volume_loss': loss_volume.item(),
                    'weight_loss': loss_weight.item(),
                    'dipeptide_loss': loss_dipeptide.item()
                })

        avg_val_total_loss = total_val_loss / len(val_loader)
        avg_val_hydro_loss = val_hydro_loss / len(val_loader)
        avg_val_polar_loss = val_polar_loss / len(val_loader)
        avg_val_volume_loss = val_volume_loss / len(val_loader)
        avg_val_weight_loss = val_weight_loss / len(val_loader)
        avg_val_dipeptide_loss = val_dipeptide_loss / len(val_loader)

        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Training Loss:")
        print(f"  Total Loss: {avg_train_total_loss:.4f}")
        print(f"  Hydrophobicity Loss: {avg_train_hydro_loss:.4f}")
        print(f"  Polarity Loss: {avg_train_polar_loss:.4f}")
        print(f"  Volume Loss: {avg_train_volume_loss:.4f}")
        print(f"  Molecular Weight Loss: {avg_train_weight_loss:.4f}")
        print(f"  Dipeptide Composition Loss: {avg_train_dipeptide_loss:.4f}")
        
        print(f"Validation Loss:")
        print(f"  Total Loss: {avg_val_total_loss:.4f}")
        print(f"  Hydrophobicity Loss: {avg_val_hydro_loss:.4f}")
        print(f"  Polarity Loss: {avg_val_polar_loss:.4f}")
        print(f"  Volume Loss: {avg_val_volume_loss:.4f}")
        print(f"  Molecular Weight Loss: {avg_val_weight_loss:.4f}")
        print(f"  Dipeptide Composition Loss: {avg_val_dipeptide_loss:.4f}")

        if avg_val_total_loss < best_val_loss:
            best_val_loss = avg_val_total_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'volume_mean': train_loader.dataset.features_df.attrs.get('volume_mean', 0),
                'volume_std': train_loader.dataset.features_df.attrs.get('volume_std', 1),
                'weight_mean': train_loader.dataset.features_df.attrs.get('molecular_weight_mean', 0),
                'weight_std': train_loader.dataset.features_df.attrs.get('molecular_weight_std', 1)
            }, 'best_protein_transformer_BN.pth')
            print('Model saved!')


def extract_pretrained_features(model, data_loader, device, volume_mean=0, volume_std=1, weight_mean=0, weight_std=1):
    model.eval()
    all_sequences = []
    all_embeddings = []
    all_predictions = []

    progress = tqdm(data_loader, desc="Extracting embedding features")
    with torch.no_grad():
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sequences = batch['sequence']

            outputs = model(input_ids, attention_mask)

            embeddings = outputs['embedding'].cpu().numpy()
            
            predictions = {
                'hydrophobicity': outputs['hydrophobicity'].cpu().numpy(),
                'polarity': outputs['polarity'].cpu().numpy(),
                'volume': outputs['volume'].cpu().numpy() * volume_std + volume_mean,
                'molecular_weight': outputs['molecular_weight'].cpu().numpy() * weight_std + weight_mean
            }

            all_sequences.extend(sequences)
            all_embeddings.append(embeddings)
            all_predictions.append(predictions)

    all_embeddings = np.vstack(all_embeddings)
    
    combined_predictions = {
        key: np.concatenate([p[key] for p in all_predictions])
        for key in all_predictions[0].keys()
    }

    return all_sequences, all_embeddings, combined_predictions


def save_embeddings_to_csv(sequences, embeddings, output_file):
    embedding_strings = []
    for embed in embeddings:
        embed_str = ','.join([f'{x:.6f}' for x in embed])
        embedding_strings.append(embed_str)

    df = pd.DataFrame({
        'fasta': sequences,
        'pre_p': embedding_strings
    })

    df.to_csv(output_file, index=False)
    print(f"Embedding features saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Protein Sequence Feature Extraction and Pre-training Tool')
    parser.add_argument('--pretrain_data', default='DTI+moa.csv', help='Dataset for pre-training')
    parser.add_argument('--extract_data', default='DTI+moa.csv', help='Dataset for feature extraction')
    parser.add_argument('--output', default='Embedding_protein.csv', help='Output CSV file path for embedding features')
    parser.add_argument('--column', default='fasta', help='Column name containing protein sequences')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--max_len', type=int, default=1000, help='Maximum sequence length')
    parser.add_argument('--model_path', default='best_protein_transformer_BN.pth', help='Model save/load path')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        print(f"===== Starting pre-training phase, using dataset: {args.pretrain_data} =====")

        pretrain_features_df = process_csv(args.pretrain_data, 'pretrain_features_temp.csv', args.column)

        volume_mean = pretrain_features_df.attrs.get('volume_mean', 0)
        volume_std = pretrain_features_df.attrs.get('volume_std', 1)
        weight_mean = pretrain_features_df.attrs.get('molecular_weight_mean', 0)
        weight_std = pretrain_features_df.attrs.get('molecular_weight_std', 1)
        
        print(f"Standardization parameters - Volume: mean={volume_mean:.4f}, std={volume_std:.4f}")
        print(f"Standardization parameters - Molecular Weight: mean={weight_mean:.4f}, std={weight_std:.4f}")

        pretrain_sequences = pretrain_features_df[args.column].tolist()

        train_seqs, val_seqs, train_indices, val_indices = train_test_split(
            pretrain_sequences, range(len(pretrain_sequences)), test_size=0.2, random_state=42
        )

        train_features = pretrain_features_df.iloc[train_indices]
        val_features = pretrain_features_df.iloc[val_indices]

        train_dataset = ProteinDataset(train_seqs, train_features, max_len=args.max_len)
        val_dataset = ProteinDataset(val_seqs, val_features, max_len=args.max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        model = ProteinTransformer(
            vocab_size=len(AA_DICT),
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            hidden_dim=1024,
            max_len=args.max_len
        ).to(device)

        train_model(
            model,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=1e-4
        )

        print(f"\n===== Starting feature extraction phase, using dataset: {args.extract_data} =====")

        extract_df = process_csv(args.extract_data, 'extract_features_temp.csv', args.column)

        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        volume_mean = checkpoint.get('volume_mean', 0)
        volume_std = checkpoint.get('volume_std', 1)
        weight_mean = checkpoint.get('weight_mean', 0)
        weight_std = checkpoint.get('weight_std', 1)
        
        print(f"Loading standardization parameters - Volume: mean={volume_mean:.4f}, std={volume_std:.4f}")
        print(f"Loading standardization parameters - Molecular Weight: mean={weight_mean:.4f}, std={weight_std:.4f}")

        print(f"Model loaded: {args.model_path}")

        extract_sequences = extract_df[args.column].tolist()

        dataset = ProteinDataset(extract_sequences, None, max_len=args.max_len, is_pretrain=False)
        data_loader = DataLoader(dataset, batch_size=args.batch_size)

        sequences, embeddings, predictions = extract_pretrained_features(
            model, data_loader, device, volume_mean, volume_std, weight_mean, weight_std
        )

        save_embeddings_to_csv(sequences, embeddings, args.output)
        
        predictions_df = pd.DataFrame({
            'fasta': sequences,
            'predicted_hydrophobicity': predictions['hydrophobicity'],
            'predicted_polarity': predictions['polarity'],
            'predicted_volume': predictions['volume'],
            'predicted_molecular_weight': predictions['molecular_weight']
        })
        predictions_df.to_csv('predicted_properties.csv', index=False)
        print(f"Predicted physicochemical properties saved to predicted_properties.csv")

        print("\n===== Processing complete =====")
        print(f"1. Pre-trained model saved to: {args.model_path}")
        print(f"2. Protein embedding features saved to: {args.output}")
        print(f"3. Predicted physicochemical properties saved to: predicted_properties.csv")

    except Exception as e:
        print(f"Program execution error: {str(e)}")


if __name__ == "__main__":
    main()