import numpy as np

feats = np.load("data/crunchbase_tgn.npy")

print("="*70)
print("🔬 DIAGNOSTIC DES FEATURES")
print("="*70)
print(f"Shape: {feats.shape}")
print(f"\nFeature 0 (raised_amount):")
print(f"  Min: {feats[:, 0].min():.0f}, Max: {feats[:, 0].max():.0f}")
print(f"  Mean: {feats[:, 0].mean():.0f}, Std: {feats[:, 0].std():.0f}")
print(f"  Zeros: {(feats[:, 0] == 0).sum()} / {len(feats)} ({100*(feats[:, 0] == 0).sum()/len(feats):.1f}%)")

print(f"\nFeature 1 (num_rounds):")
print(f"  Min: {feats[:, 1].min():.0f}, Max: {feats[:, 1].max():.0f}")
print(f"  Mean: {feats[:, 1].mean():.2f}, Std: {feats[:, 1].std():.2f}")
print(f"  Valeur la plus fréquente: {np.bincount(feats[:, 1].astype(int)).argmax()}")
print("="*70)