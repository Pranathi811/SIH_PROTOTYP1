"""Machine learning models for satellite anomaly detection (Isolation Forest & Autoencoder).

Implements FR-04, FR-05, and FR-06.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except (ImportError, Exception):
    HAS_TORCH = False

from .data_loader import CHANNELS


if HAS_TORCH:
    class AutoencoderNN(nn.Module):
        """PyTorch Autoencoder for multivariate telemetry reconstruction."""
        def __init__(self, input_dim: int = 5, latent_dim: int = 3):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, 8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 8),
                nn.LeakyReLU(0.1),
                nn.Linear(8, 16),
                nn.LeakyReLU(0.1),
                nn.Linear(16, input_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed


class SatelliteAutoencoder:
    """
    Robust Autoencoder implementation supporting PyTorch backend with fallback
    to multi-layer neural network autoencoder.
    """
    def __init__(self, input_dim: int = 5, latent_dim: int = 3, lr: float = 0.005, random_state: int = 42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.random_state = random_state
        self.use_torch = HAS_TORCH
        self.torch_model: Optional[Any] = None
        self.sklearn_model: Optional[Any] = None
        
        if self.use_torch:
            torch.manual_seed(random_state)
            self.torch_model = AutoencoderNN(input_dim=input_dim, latent_dim=latent_dim)
        else:
            self.sklearn_model = MLPRegressor(
                hidden_layer_sizes=(16, 8, latent_dim, 8, 16),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=64,
                learning_rate_init=lr,
                max_iter=250,
                random_state=random_state,
                early_stopping=True
            )

    def fit(self, X: np.ndarray, epochs: int = 80, batch_size: int = 64, verbose: bool = False) -> "SatelliteAutoencoder":
        if self.use_torch and self.torch_model is not None:
            self.torch_model.train()
            dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = optim.AdamW(self.torch_model.parameters(), lr=self.lr, weight_decay=1e-5)
            criterion = nn.MSELoss()
            
            for epoch in range(epochs):
                total_loss = 0.0
                for (batch_x,) in loader:
                    optimizer.zero_grad()
                    reconstructed = self.torch_model(batch_x)
                    loss = criterion(reconstructed, batch_x)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(batch_x)
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"Autoencoder Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(X):.6f}")
        else:
            # Fit MLP autoencoder (target is X itself)
            self.sklearn_model.fit(X, X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed multivariate features."""
        if self.use_torch and self.torch_model is not None:
            self.torch_model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32)
                reconstructed = self.torch_model(tensor_x)
                return reconstructed.cpu().numpy()
        else:
            return self.sklearn_model.predict(X)

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error per sample (anomaly score)."""
        recon = self.predict(X)
        errors = np.mean((X - recon) ** 2, axis=1)
        return errors

    def save(self, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_torch and self.torch_model is not None:
            torch.save(self.torch_model.state_dict(), path)
        # Always dump wrapper/sklearn model for fallback portability
        alt_path = path.with_suffix(".pkl")
        joblib.dump(self, alt_path)

    def load(self, filepath: str | Path) -> "SatelliteAutoencoder":
        path = Path(filepath)
        if path.suffix == ".pth" and HAS_TORCH:
            if self.torch_model is None:
                self.torch_model = AutoencoderNN(input_dim=self.input_dim, latent_dim=self.latent_dim)
            self.torch_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.torch_model.eval()
            self.use_torch = True
        elif path.suffix == ".pkl" or path.with_suffix(".pkl").exists():
            pkl_file = path if path.suffix == ".pkl" else path.with_suffix(".pkl")
            loaded = joblib.load(pkl_file)
            return loaded
        return self


def train_isolation_forest(X_train: np.ndarray, random_state: int = 42) -> IsolationForest:
    """Train Isolation Forest anomaly detector on scaled training features."""
    forest = IsolationForest(
        n_estimators=200,
        contamination=0.04,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1
    )
    forest.fit(X_train)
    return forest


def train_all_models(
    X_train: np.ndarray,
    random_state: int = 42,
    epochs: int = 80
) -> Dict[str, Any]:
    """Train both Isolation Forest and Neural Autoencoder."""
    print("-> Training Isolation Forest model...")
    iso_forest = train_isolation_forest(X_train, random_state=random_state)
    
    print("-> Training Neural Autoencoder model...")
    autoencoder = SatelliteAutoencoder(input_dim=X_train.shape[1], random_state=random_state)
    autoencoder.fit(X_train, epochs=epochs, verbose=False)
    
    return {
        "isolation_forest": iso_forest,
        "autoencoder": autoencoder
    }


def build_models(frames: Dict[str, Any]) -> Dict[str, Any]:
    """Build and load anomaly detection models from dataset frames or pre-trained models."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    
    # Prepare data
    train_df = frames["train"]
    test_df = frames["test"]
    labels_df = frames["labels"]
    
    # Extract features and scale
    X_train = train_df[CHANNELS].values
    X_test = test_df[CHANNELS].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Try to load pre-trained models, otherwise train with minimal resources
    models_dir = Path("models")
    if models_dir.exists() and (models_dir / "isolation_forest.pkl").exists():
        print("Loading pre-trained models...")
        iso_forest = joblib.load(models_dir / "isolation_forest.pkl")
        autoencoder = joblib.load(models_dir / "autoencoder.pkl")
    else:
        print("Training models with minimal resources (this may take a moment)...")
        # Train with reduced memory footprint
        iso_forest = IsolationForest(
            n_estimators=50,  # Reduced from 200
            contamination=0.04,
            max_samples=min(256, len(X_train_scaled)),  # Limit sample size
            random_state=42,
            n_jobs=1  # Single thread to reduce memory usage
        )
        iso_forest.fit(X_train_scaled)
        
        autoencoder = SatelliteAutoencoder(input_dim=X_train_scaled.shape[1], random_state=42)
        autoencoder.fit(X_train_scaled, epochs=10, batch_size=32, verbose=False)  # Reduced epochs and batch size
    
    # Generate predictions and scores
    # Isolation Forest predictions (1 = anomaly, -1 = normal)
    iso_predictions_raw = iso_forest.predict(X_test_scaled)
    iso_predictions = (iso_predictions_raw == -1).astype(int)  # Convert to 0=normal, 1=anomaly
    iso_scores = -iso_forest.score_samples(X_test_scaled)  # Anomaly scores (higher = more anomalous)
    
    # Autoencoder predictions
    ae_scores = autoencoder.get_reconstruction_errors(X_test_scaled)
    ae_threshold = np.percentile(ae_scores, 96)  # Use 96th percentile as threshold
    ae_predictions = (ae_scores > ae_threshold).astype(int)
    
    # Get true labels
    true_labels = test_df.timestamp.map(labels_df.set_index("timestamp")["anomaly_type"]).notna().astype(int)
    
    # Calculate metrics
    metrics = {}
    for model_name, predictions in [("isolation_forest", iso_predictions), ("autoencoder", ae_predictions)]:
        metrics[model_name] = {
            "f1_score": f1_score(true_labels, predictions, zero_division=0),
            "precision": precision_score(true_labels, predictions, zero_division=0),
            "recall": recall_score(true_labels, predictions, zero_division=0),
        }
    
    # Calculate thresholds based on test set scores
    iso_threshold = np.percentile(iso_scores, 96)
    ae_threshold_val = np.percentile(ae_scores, 96)
    
    # Flatten structure for API access
    return {
        "predictions": {
            "isolation_forest": iso_predictions,
            "autoencoder": ae_predictions
        },
        "scores": {
            "isolation_forest": iso_scores,
            "autoencoder": ae_scores
        },
        "metrics": metrics,
        "scaler": scaler,
        "isolation_forest": iso_forest,
        "autoencoder": autoencoder,
        "thresholds": {
            "isolation_forest": iso_threshold,
            "autoencoder": ae_threshold_val
        },
        "models": {"isolation_forest": iso_forest, "autoencoder": autoencoder}
    }


def channel_deviations(models: Dict[str, Any], values: list) -> list:
    """Calculate channel deviations from expected ranges."""
    import numpy as np
    # Calculate z-scores for each channel based on standard deviation
    deviations = []
    for i, channel in enumerate(CHANNELS):
        if i < len(values):
            # Simple deviation calculation
            value = values[i]
            # Assume mean of 50 and std of 15 for scaling purposes
            mean = 50.0
            std = 15.0
            z_score = (value - mean) / std if std > 0 else 0
            deviation = {
                "channel": channel,
                "value": value,
                "z_score": float(z_score),
                "absolute_z_score": float(abs(z_score)),
                "deviation": abs(z_score)
            }
            deviations.append(deviation)
    return deviations


def per_type_metrics(frames: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate metrics per anomaly type."""
    from sklearn.metrics import f1_score
    
    test_df = frames["test"]
    labels_df = frames["labels"]
    
    true_labels = test_df.timestamp.map(labels_df.set_index("timestamp")["anomaly_type"])
    anomaly_types = labels_df["anomaly_type"].unique()
    
    metrics = {}
    for atype in anomaly_types:
        mask = (true_labels == atype)
        if mask.sum() > 0:
            metrics[atype] = {
                "count": int(mask.sum()),
                "detected": int((models["predictions"]["isolation_forest"][mask] == 1).sum())
            }
    
    return metrics


def save_trained_models(
    models: Dict[str, Any],
    scaler: StandardScaler,
    output_dir: str | Path = "models"
) -> None:
    """Serialize all models and scaler to the models directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save Isolation Forest
    joblib.dump(models["isolation_forest"], path / "isolation_forest.pkl")
    
    # Save Autoencoder
    models["autoencoder"].save(path / "autoencoder.pth")
        
    # Save Scaler
    joblib.dump(scaler, path / "scaler.pkl")
    print(f"[OK] Serialized models to {path.resolve()}/ (isolation_forest.pkl, autoencoder.pth, scaler.pkl)")
