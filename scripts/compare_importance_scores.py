"""
Compare importance score methods for unstructured pruning.
Single GPU, no DDP. Reuses pruning config format.

Usage:
    python scripts/compare_importance_scores.py configs/prune/taylor.toml
    python scripts/compare_importance_scores.py configs/prune/taylor.toml --amounts 0.1 0.3 0.5
"""

import copy
import argparse

import toml
import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.vit import ViT
from src.models.simple_vit import SimpleViT
from src.data.py2d_dataset import Py2DDataset
from src.training.utils.importance_scores import compute_importance_scores

METRICS = ['l1', 'taylor', 'fisher', 'random']

def load_model(config, device):
    model_config = config['model']
    if 'encoder_embed_dim' in model_config:
        model = ViT(**model_config)
    else:
        model = SimpleViT(**model_config)

    state_dict = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
    model_state = state_dict.get('model_state', state_dict)
    model.load_state_dict(model_state)
    model.to(device)
    return model


@torch.no_grad()
def evaluate(model, dataset, device, n_samples=4096, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    total_loss, seen = 0., 0
    for x, y_true in dataloader:
        if seen >= n_samples:
            break
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        total_loss += F.mse_loss(y_pred, y_true).item() * x.size(0)
        seen += x.size(0)
    return total_loss / max(seen, 1)


def get_flat_mask(model):
    """Concatenated binary mask across all prunable weights (deterministic order)."""
    parts = []
    for module, name in model.get_weights():
        mask = getattr(module, name + '_mask', torch.ones_like(getattr(module, name)))
        parts.append(mask.flatten())
    return torch.cat(parts)


def get_flat_scores(scores_dict, model):
    """Flatten scores dict into a single tensor, aligned with get_flat_mask ordering."""
    parts = []
    for module, name in model.get_weights():
        parts.append(scores_dict[(module, name)].flatten())
    return torch.cat(parts)


def score_stats(flat):
    return {
        'mean': flat.mean().item(),
        'std': flat.std().item(),
        'min': flat.min().item(),
        'max': flat.max().item(),
        'median': flat.median().item(),
        'p5': flat.quantile(0.05).item(),
        'p95': flat.quantile(0.95).item(),
        'nonzero': (flat != 0).float().mean().item(),
    }


def print_stats(label, stats):
    print(f"  {label}")
    print(f"    mean={stats['mean']:.2e}  std={stats['std']:.2e}  median={stats['median']:.2e}")
    print(f"    min={stats['min']:.2e}  max={stats['max']:.2e}")
    print(f"    p5={stats['p5']:.2e}  p95={stats['p95']:.2e}  nonzero={stats['nonzero']:.1%}")


def rank_correlation(a, b):
    """Spearman rank correlation between two flat tensors."""
    rank_a = a.argsort().argsort().float()
    rank_b = b.argsort().argsort().float()
    ra = rank_a - rank_a.mean()
    rb = rank_b - rank_b.mean()
    return (ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12)


def mask_overlap(flat_a, flat_b):
    """Jaccard index between two pruned sets."""
    pruned_a = (flat_a == 0)
    pruned_b = (flat_b == 0)
    intersection = (pruned_a & pruned_b).sum().item()
    union = (pruned_a | pruned_b).sum().item()
    return intersection / max(union, 1)


def main():
    parser = argparse.ArgumentParser(description='Compare pruning importance score methods')
    parser.add_argument('config', help='TOML config (reuses pruning config format)')
    parser.add_argument('--amounts', nargs='+', type=float, default=[0.1, 0.3, 0.5],
                        help='Pruning amounts to test (default: 0.1 0.3 0.5)')
    parser.add_argument('--eval_samples', type=int, default=4096)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    args = parser.parse_args()

    config = toml.load(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(config, device)
    print(f"Parameters: {model.n_parameters():,}")

    val_dataset = Py2DDataset(**config['validation_dataset'])
    train_dataset = Py2DDataset(**config['train_dataset'])

    baseline_loss = evaluate(model, val_dataset, device, args.eval_samples, args.eval_batch_size)
    print(f"Baseline loss: {baseline_loss:.6e}\n")

    # ----- Score distributions on the unpruned model -----

    print("=" * 70)
    print("IMPORTANCE SCORE DISTRIBUTIONS (unpruned model)")
    print("=" * 70)

    # Compute all score dicts once — reused for distributions, correlation, and per-layer table
    scored_metrics = ['l1', 'taylor', 'fisher']
    scores_dicts = {}
    flat_scores = {}

    scores_dicts['l1'] = {(m, n): getattr(m, n).detach().abs() for m, n in model.get_weights()}
    flat_scores['l1'] = get_flat_scores(scores_dicts['l1'], model)
    print_stats('L1 (|weight|)', score_stats(flat_scores['l1']))

    for metric in ['taylor', 'fisher']:
        scores_dicts[metric] = compute_importance_scores(metric, model, device, dataset=train_dataset)
        flat_scores[metric] = get_flat_scores(scores_dicts[metric], model)
        print_stats(metric.upper(), score_stats(flat_scores[metric]))
        model.zero_grad(set_to_none=True)

    # Rank correlation matrix
    print(f"\n  Rank correlation (Spearman):")
    header = "           " + "  ".join(f"{m:>8s}" for m in scored_metrics)
    print(header)
    for m1 in scored_metrics:
        row = f"  {m1:>8s}"
        for m2 in scored_metrics:
            if m1 == m2:
                row += f"  {'1.000':>8s}"
            else:
                rho = rank_correlation(flat_scores[m1], flat_scores[m2]).item()
                row += f"  {rho:>8.3f}"
        print(row)
    print()

    # Per-layer score comparison
    print("  Per-layer mean scores:")
    weights = model.get_weights()
    header = f"  {'layer':<40s}" + "".join(f"  {m:>10s}" for m in scored_metrics)
    print(header)
    print("  " + "-" * (40 + 12 * len(scored_metrics)))
    for i, (module, name) in enumerate(weights):
        for attr_name, attr_val in model.named_modules():
            if attr_val is module:
                layer_name = f"{attr_name}.{name}"
                break
        else:
            layer_name = f"[{i}].{name}"
        row = f"  {layer_name:<40s}"
        for metric in scored_metrics:
            val = scores_dicts[metric][(module, name)].mean().item()
            row += f"  {val:>10.2e}"
        print(row)
    print()

    # ----- Pruning comparison -----

    for amount in args.amounts:
        print("=" * 70)
        print(f"PRUNING AT {amount:.0%} SPARSITY")
        print("=" * 70)

        flat_masks = {}
        results = {}

        for metric in METRICS:
            model_copy = copy.deepcopy(model)

            scores = compute_importance_scores(metric, model_copy, device, dataset=train_dataset)
            pruning_method = prune.RandomUnstructured if metric == 'random' else prune.L1Unstructured

            prune.global_unstructured(
                model_copy.get_weights(),
                pruning_method=pruning_method,
                importance_scores=scores,
                amount=amount,
            )

            post_loss = evaluate(model_copy, val_dataset, device, args.eval_samples, args.eval_batch_size)
            pruned = model_copy.n_pruned_parameters()
            total = model_copy.n_parameters()
            delta = post_loss - baseline_loss

            flat_masks[metric] = get_flat_mask(model_copy)
            results[metric] = post_loss

            print(f"  {metric.upper():8s}  loss={post_loss:.6e}  "
                  f"delta={delta:+.2e} ({delta/baseline_loss:+.1%})  "
                  f"sparsity={pruned/total:.1%}")

            del model_copy
            torch.cuda.empty_cache()

        # Mask overlap
        print(f"\n  Mask overlap (Jaccard):")
        header = "           " + "  ".join(f"{m:>8s}" for m in METRICS)
        print(header)
        for m1 in METRICS:
            row = f"  {m1:>8s}"
            for m2 in METRICS:
                if m1 == m2:
                    row += f"  {'---':>8s}"
                else:
                    ov = mask_overlap(flat_masks[m1], flat_masks[m2])
                    row += f"  {ov:>7.1%} "
            print(row)
        print()


if __name__ == '__main__':
    main()
