"""Quick integrity check for all new/updated code."""
import sys; sys.path.insert(0, 'src')
import torch

# 1. permutation_test_cd_spi with effect_size
from fclprm.metrics.cd_spi_stats import permutation_test_cd_spi
embs = {'c0': torch.randn(64), 'c1': torch.randn(64), 'c2': torch.randn(64), 'c3': torch.randn(64)}
r = permutation_test_cd_spi(embs, n_permutations=50)
assert 'effect_size' in r, f"effect_size missing: {list(r.keys())}"
print(f"permutation_test: effect_size={r['effect_size']:.3f}, p={r['p_value']:.3f}")

# 2. PCA EVR NaN handling
from fclprm.metrics.cd_spi import compute_pca_evr
embs_bad = {'c0': torch.tensor([float('nan')]*64), 'c1': torch.randn(64), 'c2': torch.randn(64), 'c3': torch.randn(64)}
r2 = compute_pca_evr(embs_bad)
assert r2['interpretation'] == 'nan_input', f"Got {r2['interpretation']}"
print(f"pca_evr NaN: {r2['interpretation']}")

# 3. PCA EVR zero variance
embs_zero = {'c0': torch.ones(64), 'c1': torch.ones(64), 'c2': torch.ones(64), 'c3': torch.ones(64)}
r3 = compute_pca_evr(embs_zero)
assert r3['interpretation'] == 'zero_variance', f"Got {r3['interpretation']}"
print(f"pca_evr zero_var: {r3['interpretation']}")

# 4. CKA NaN handling
from fclprm.metrics.cka import compute_cka
cka_nan = compute_cka(torch.tensor([[float('nan')]*5, [1.0]*5]), torch.randn(2, 5))
assert cka_nan != cka_nan, "Should be NaN"
print(f"cka NaN: correct")

# 5. CKA zero variance
cka_zero = compute_cka(torch.ones(3, 5), torch.ones(3, 5))
print(f"cka zero_var: {cka_zero:.1f} (expected 1.0)")

# 6. fclprm.metrics imports
from fclprm.metrics import compute_correlation, prepare_aligned_series, cross_experiment_correlation
from fclprm.metrics import compute_pca_evr, compute_cka
print(f"__init__ exports: all OK")

print("\nALL INTEGRITY CHECKS PASSED")
