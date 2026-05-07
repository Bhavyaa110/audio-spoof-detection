"""
dataset.py — ASVspoof 2019 Dataset Loader + RawBoost Augmentation

Matches this exact local folder structure:
    asvspoof_dataset/
    ├── ASVspoof2019_LA_asv_protocols/
    ├── ASVspoof2019_LA_asv_scores/
    ├── ASVspoof2019_LA_cm_protocols/
    │     ├── ASVspoof2019.LA.cm.train.trn.txt
    │     ├── ASVspoof2019.LA.cm.dev.trl.txt
    │     └── ASVspoof2019.LA.cm.eval.trl.txt
    ├── ASVspoof2019_LA_dev/
    │     └── flac/   ← audio files here
    ├── ASVspoof2019_LA_eval/
    │     └── flac/
    ├── ASVspoof2019_LA_train/
    │     └── flac/
    └── README.LA.txt
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as AF


# ─────────────────────────────────────────────
# Path resolver — matches YOUR folder layout
# ─────────────────────────────────────────────

class ASVspoofPaths:
    """
    Single source of truth for all paths.
    Pass data_root = path to your 'asvspoof_dataset' folder.

    Example:
        paths = ASVspoofPaths(r'C:/datasets/asvspoof_dataset')
        paths.validate()   # prints ✓/✗ for every expected path
    """

    def __init__(self, data_root: str):
        self.root = data_root

    # ── CM protocol files (spoof detection) ──────────────────────
    @property
    def train_protocol(self):
        return os.path.join(self.root,
            'ASVspoof2019_LA_cm_protocols',
            'ASVspoof2019.LA.cm.train.trn.txt')

    @property
    def dev_protocol(self):
        return os.path.join(self.root,
            'ASVspoof2019_LA_cm_protocols',
            'ASVspoof2019.LA.cm.dev.trl.txt')

    @property
    def eval_protocol(self):
        return os.path.join(self.root,
            'ASVspoof2019_LA_cm_protocols',
            'ASVspoof2019.LA.cm.eval.trl.txt')

    # ── ASV protocol files (for joint tDCF evaluation) ────────────
    @property
    def asv_train_protocol(self):
        return os.path.join(self.root,
            'ASVspoof2019_LA_asv_protocols',
            'ASVspoof2019.LA.asv.train.trn.txt')

    @property
    def asv_eval_protocol(self):
        return os.path.join(self.root,
            'ASVspoof2019_LA_asv_protocols',
            'ASVspoof2019.LA.asv.eval.trl.txt')

    # ── Audio directories ─────────────────────────────────────────
    def audio_dir(self, split: str) -> str:
        """
        Returns the flac/ folder for a given split.
        split: 'train' | 'dev' | 'eval'
        """
        folder = {
            'train': 'ASVspoof2019_LA_train',
            'dev':   'ASVspoof2019_LA_dev',
            'eval':  'ASVspoof2019_LA_eval',
        }[split]
        return os.path.join(self.root, folder, 'flac')

    def validate(self) -> bool:
        """Print a quick sanity-check of all expected paths."""
        checks = {
            'CM train protocol': self.train_protocol,
            'CM dev protocol':   self.dev_protocol,
            'CM eval protocol':  self.eval_protocol,
            'train audio dir':   self.audio_dir('train'),
            'dev audio dir':     self.audio_dir('dev'),
            'eval audio dir':    self.audio_dir('eval'),
        }
        all_ok = True
        for name, path in checks.items():
            ok     = os.path.exists(path)
            symbol = '✓' if ok else '✗ MISSING'
            print(f"  [{symbol}]  {name}:\n          {path}")
            if not ok:
                all_ok = False
        return all_ok


# ─────────────────────────────────────────────
# RawBoost Augmentation  (Tak et al. 2022)
# ─────────────────────────────────────────────

def _LnL_convolutive_noise(x: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """Linear and Non-Linear convolutive noise injection."""
    n     = len(x)
    X     = np.fft.rfft(x, n=n)
    nfreq = len(np.fft.rfftfreq(n))
    H     = np.random.randn(nfreq) + 1j * np.random.randn(nfreq)
    H     = np.abs(H) ** (-beta / 2.0)
    noise = np.fft.irfft(X * H, n=n)
    return x + 0.05 * noise / (np.std(noise) + 1e-8)


def _ISD_additive_noise(x: np.ndarray, P: float = 10.0, g_sd: float = 2.0) -> np.ndarray:
    """Impulsive signal-dependent additive noise."""
    beta      = np.random.randn() * g_sd
    y         = x / (np.max(np.abs(x)) + 1e-8)
    y_clipped = np.clip(y, -P / 100, P / 100)
    return x + beta * (x - y_clipped)


def _SSI_additive_noise(x: np.ndarray, SNRmin: int = 10, SNRmax: int = 40) -> np.ndarray:
    """Stationary signal-independent additive noise."""
    noise   = np.random.randn(len(x))
    snr     = 10 ** (np.random.uniform(SNRmin, SNRmax) / 20)
    sig_pow = np.sqrt(np.mean(x ** 2)) + 1e-8
    return x + noise / (np.std(noise) + 1e-8) * sig_pow / snr


def rawboost(x: np.ndarray, algo: int = None) -> np.ndarray:
    """
    Apply RawBoost noise augmentation.

    algo:
      0 = LnL only
      1 = ISD only
      2 = SSI only
      3 = LnL + ISD
      4 = LnL + SSI
      5 = ISD + SSI
      6 = LnL + ISD + SSI  (all three)
      None = pick one of the above randomly each call
    """
    if algo is None:
        algo = random.randint(0, 6)
    if algo == 0: return _LnL_convolutive_noise(x)
    if algo == 1: return _ISD_additive_noise(x)
    if algo == 2: return _SSI_additive_noise(x)
    if algo == 3: return _ISD_additive_noise(_LnL_convolutive_noise(x))
    if algo == 4: return _SSI_additive_noise(_LnL_convolutive_noise(x))
    if algo == 5: return _SSI_additive_noise(_ISD_additive_noise(x))
    return _SSI_additive_noise(_ISD_additive_noise(_LnL_convolutive_noise(x)))


# ─────────────────────────────────────────────
# Protocol parser
# ─────────────────────────────────────────────

def parse_protocol(protocol_path: str) -> list:
    """
    Parse an ASVspoof 2019 LA CM protocol .txt file.

    File format (5 space-separated columns):
        SPEAKER_ID   FILENAME   -   ATTACK_TYPE   KEY
        LA_0079      LA_T_1138215   -   A10   spoof
        LA_0079      LA_T_1138211   -   -     bonafide

    Returns:
        list of dicts, each with keys:
          'filename'  : str  — audio file stem (no extension)
          'speaker'   : str  — speaker ID, e.g. 'LA_0079'
          'attack'    : str  — 'A01'..'A19' or '-' for bonafide
          'label'     : int  — 0 = bonafide, 1 = spoof
    """
    entries = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker  = parts[0]
            filename = parts[1]
            attack   = parts[3]          # '-' for bonafide
            key      = parts[4]          # 'bonafide' or 'spoof'
            label    = 0 if key == 'bonafide' else 1
            entries.append({
                'filename': filename,
                'speaker':  speaker,
                'attack':   attack,
                'label':    label,
            })
    return entries


def build_speaker_map(entries: list) -> dict:
    """Map speaker string IDs → consecutive integer indices."""
    speakers = sorted(set(e['speaker'] for e in entries))
    return {spk: idx for idx, spk in enumerate(speakers)}


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ASVspoofDataset(Dataset):
    """
    ASVspoof 2019 LA Dataset.

    Args:
        paths        : ASVspoofPaths instance pointing to your dataset folder
        split        : 'train' | 'dev' | 'eval'
        max_len      : waveform length in samples  (default 64000 = 4 s @ 16kHz)
        augment      : apply RawBoost + speed perturbation (use True for train only)
        speaker_map  : str→int speaker map; pass train_ds.speaker_map to dev/eval
                       so all splits share the same speaker indices
        sample_rate  : target sample rate (ASVspoof 2019 LA is 16kHz)
    """

    def __init__(self, paths: ASVspoofPaths, split: str,
                 max_len: int = 64000, augment: bool = False,
                 speaker_map: dict = None, sample_rate: int = 16000):

        proto_file = {
            'train': paths.train_protocol,
            'dev':   paths.dev_protocol,
            'eval':  paths.eval_protocol,
        }[split]

        self.audio_dir   = paths.audio_dir(split)
        self.max_len     = max_len
        self.augment     = augment
        self.sample_rate = sample_rate
        self.split       = split

        self.entries     = parse_protocol(proto_file)
        self.speaker_map = speaker_map or build_speaker_map(self.entries)
        self.n_speakers  = len(self.speaker_map)

        n_bon  = sum(1 for e in self.entries if e['label'] == 0)
        n_spf  = sum(1 for e in self.entries if e['label'] == 1)
        print(f"  [{split:>5}]  {len(self.entries):6,} samples  "
              f"bonafide={n_bon:,}  spoof={n_spf:,}  speakers={self.n_speakers}")

    def __len__(self):
        return len(self.entries)

    # ── Audio helpers ────────────────────────────────────────────

    def _load(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.audio_dir, filename + '.flac')
        wav, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            wav = AF.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)   # [T]

    def _pad_or_trim(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.size(0)
        if T >= self.max_len:
            # Random crop during training; centre crop otherwise
            start = random.randint(0, T - self.max_len) if self.augment \
                    else (T - self.max_len) // 2
            return wav[start: start + self.max_len]
        # Repeat-pad short clips
        reps = (self.max_len // T) + 1
        return wav.repeat(reps)[:self.max_len]

    def _speed_perturb(self, wav: torch.Tensor) -> torch.Tensor:
        factor = random.choice([0.9, 1.0, 1.1])
        if factor == 1.0:
            return wav
        arr     = wav.numpy()
        n_out   = int(len(arr) / factor)
        indices = np.linspace(0, len(arr) - 1, n_out)
        arr     = np.interp(indices, np.arange(len(arr)), arr).astype(np.float32)
        return torch.from_numpy(arr)

    # ── Main getter ──────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        wav   = self._load(entry['filename'])

        if self.augment:
            # Speed perturbation — 50% probability
            if random.random() < 0.5:
                wav = self._speed_perturb(wav)
            # RawBoost — 70% probability
            if random.random() < 0.7:
                arr = wav.numpy().astype(np.float64)
                arr = rawboost(arr)
                wav = torch.from_numpy(arr.astype(np.float32))

        wav = self._pad_or_trim(wav)
        wav = wav / (wav.abs().max() + 1e-8)  # peak normalize to [-1, 1]

        speaker_id = self.speaker_map.get(entry['speaker'], 0)

        return {
            'waveform':    wav,                                              # [max_len]
            'spoof_label': torch.tensor(entry['label'],     dtype=torch.long),
            'speaker_id':  torch.tensor(speaker_id,         dtype=torch.long),
            'attack_type': entry['attack'],   # 'A01'..'A19' or '-'
            'filename':    entry['filename'],
        }


# ─────────────────────────────────────────────
# DataLoader factory  — main entry point
# ─────────────────────────────────────────────

def get_dataloaders(data_root: str,
                    max_len:     int = 64000,
                    batch_size:  int = 24,
                    num_workers: int = 4):
    """
    Build train / dev / eval DataLoaders directly from your local folder.

    Args:
        data_root   : path to your 'asvspoof_dataset' folder
                      e.g. r'C:/datasets/asvspoof_dataset'
                           '/home/user/data/asvspoof_dataset'
        max_len     : waveform length in samples  (4 s = 64000 @ 16kHz)
        batch_size  : samples per batch
        num_workers : DataLoader worker processes (0 = main process only)

    Returns:
        train_loader, dev_loader, eval_loader, speaker_map
    """
    paths = ASVspoofPaths(data_root)

    print("\n── Validating dataset paths ─────────────────────────")
    ok = paths.validate()
    if not ok:
        raise FileNotFoundError(
            "\nOne or more dataset paths are missing.\n"
            "Make sure data_root points to the folder that contains:\n"
            "  ASVspoof2019_LA_train/\n"
            "  ASVspoof2019_LA_dev/\n"
            "  ASVspoof2019_LA_eval/\n"
            "  ASVspoof2019_LA_cm_protocols/\n"
        )
    print("── Building datasets ────────────────────────────────")

    # Train split first — defines the global speaker map
    train_ds = ASVspoofDataset(paths, split='train',
                               max_len=max_len, augment=True)
    dev_ds   = ASVspoofDataset(paths, split='dev',
                               max_len=max_len, augment=False,
                               speaker_map=train_ds.speaker_map)
    eval_ds  = ASVspoofDataset(paths, split='eval',
                               max_len=max_len, augment=False,
                               speaker_map=train_ds.speaker_map)
    print("─────────────────────────────────────────────────────\n")

    _pw = num_workers > 0   # persistent_workers needs num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        drop_last=True, persistent_workers=_pw,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=_pw,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=_pw,
    )

    return train_loader, dev_loader, eval_loader, train_ds.speaker_map


# ─────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './asvspoof_dataset'

    train_loader, dev_loader, eval_loader, spk_map = get_dataloaders(
        data_root=root, batch_size=4, num_workers=0
    )

    print(f"Speaker map size : {len(spk_map)}")
    print(f"Train batches    : {len(train_loader)}")
    print(f"Dev   batches    : {len(dev_loader)}")
    print(f"Eval  batches    : {len(eval_loader)}")

    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  waveform     : {batch['waveform'].shape}")
    print(f"  spoof_label  : {batch['spoof_label'].tolist()}")
    print(f"  speaker_id   : {batch['speaker_id'].tolist()}")
    print(f"  attack_type  : {batch['attack_type']}")
    print(f"  filename     : {batch['filename']}")