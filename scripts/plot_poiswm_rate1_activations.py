from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import stable_worldmodel as swm
import stable_pretraining as spt

OUT = Path('outputs/poisson_activations_rate1')
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    ('poiswm_rate1_epoch2', 'poiswm_rate1_embed384_20260604/weights_epoch_2.pt', 'poiswm'),
]

IMAGENET_MEAN = torch.tensor(spt.data.dataset_stats.ImageNet['mean']).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(spt.data.dataset_stats.ImageNet['std']).view(1, 3, 1, 1)


def pick_sequence():
    ds = swm.data.load_dataset(
        'pusht_expert_train.h5',
        transform=None,
        cache_dir=None,
        num_steps=12,
        frameskip=5,
        keys_to_load=['pixels', 'action', 'proprio', 'state'],
        keys_to_cache=['action', 'proprio', 'state'],
    )
    idx = min(25000, len(ds) - 1)
    row = ds[idx]
    px_u8 = row['pixels']
    px01 = px_u8.float() / 255.0
    px_norm = (px01 - IMAGENET_MEAN) / IMAGENET_STD
    return idx, px01, px_norm


def encode_rates(model, pixels_norm):
    model.eval()
    with torch.no_grad():
        out = model.encode({'pixels': pixels_norm.unsqueeze(0)})
    u = out['emb'][0].detach().cpu().float()
    u_clamped = u.clamp(-20.0, 5.0)
    rates = torch.exp(u_clamped)
    stats = {
        'T': int(rates.shape[0]),
        'D': int(rates.shape[1]),
        'lograte_mean': float(u.mean()),
        'lograte_std': float(u.std()),
        'lograte_min': float(u.min()),
        'lograte_max': float(u.max()),
        'rate_mean': float(rates.mean()),
        'rate_std': float(rates.std()),
        'rate_min': float(rates.min()),
        'rate_p50': float(torch.quantile(rates.flatten(), 0.50)),
        'rate_p90': float(torch.quantile(rates.flatten(), 0.90)),
        'rate_p95': float(torch.quantile(rates.flatten(), 0.95)),
        'rate_p99': float(torch.quantile(rates.flatten(), 0.99)),
        'rate_max': float(rates.max()),
        'hi_sat_frac_lograte_ge_5': float((u >= 5.0 - 1e-3).float().mean()),
        'lo_sat_frac_lograte_le_neg20': float((u <= -20.0 + 1e-3).float().mean()),
    }
    return u.numpy(), rates.numpy(), stats


def plot_frames(px01, path):
    T = px01.shape[0]
    fig, axes = plt.subplots(1, T, figsize=(T * 1.45, 1.6), constrained_layout=True)
    for t, ax in enumerate(np.ravel(axes)):
        img = px01[t].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f't={t}', fontsize=8)
        ax.axis('off')
    fig.suptitle('PushT sequence used for activation diagnostics', fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model(name, u, rates, stats):
    T, D = rates.shape
    order = np.argsort(-rates.mean(axis=0))
    top = order[:min(10, D)]
    rng = np.random.default_rng(0)
    remaining = np.setdiff1d(np.arange(D), top, assume_unique=False)
    random_ch = rng.choice(remaining, size=min(10, len(remaining)), replace=False)
    selected = np.concatenate([top, random_ch])
    selected_labels = [f'top {i + 1}\nch {ch}' for i, ch in enumerate(top)]
    selected_labels += [f'rand {i + 1}\nch {ch}' for i, ch in enumerate(random_ch)]

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[0, :2])
    im = ax.imshow(rates[:, selected].T, aspect='auto', interpolation='nearest', cmap='magma')
    ax.set_title(f'{name}: rates exp(clamped log-rate), top 10 + random 10 channels')
    ax.set_xlabel('time in PushT snippet')
    ax.set_ylabel('latent channel')
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected_labels, fontsize=7)
    fig.colorbar(im, ax=ax, label='rate lambda')

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(rates.flatten(), bins=80, color='#375a7f')
    ax.set_yscale('log')
    ax.set_title('rate distribution')
    ax.set_xlabel('lambda')
    ax.set_ylabel('count, log scale')

    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(u[:, selected].T, aspect='auto', interpolation='nearest', cmap='coolwarm', vmin=-5, vmax=5)
    ax.set_title('raw log-rates u, same channels')
    ax.set_xlabel('time in PushT snippet')
    ax.set_ylabel('latent channel')
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected_labels, fontsize=7)
    fig.colorbar(im, ax=ax, label='log-rate u')

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(u.flatten(), bins=80, color='#7f4f37')
    ax.set_yscale('log')
    ax.axvline(5, color='crimson', linestyle='--', linewidth=1, label='rate clamp hi')
    ax.set_title('log-rate distribution')
    ax.set_xlabel('u')
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, :2])
    for ch in top:
        ax.plot(np.arange(T), rates[:, ch], linewidth=1.7, label=f'top ch {ch}')
    for ch in random_ch:
        ax.plot(np.arange(T), rates[:, ch], linewidth=1.1, linestyle='--', alpha=0.75, label=f'rand ch {ch}')
    ax.set_title('rate traces: top 10 solid, random 10 dashed')
    ax.set_xlabel('time')
    ax.set_ylabel('lambda')
    ax.legend(ncol=5, fontsize=7)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    lines = [
        f'D={stats["D"]}, T={stats["T"]}',
        f'rate mean={stats["rate_mean"]:.3g}',
        f'rate std={stats["rate_std"]:.3g}',
        f'rate p95={stats["rate_p95"]:.3g}',
        f'rate p99={stats["rate_p99"]:.3g}',
        f'rate max={stats["rate_max"]:.3g}',
        f'lograte mean={stats["lograte_mean"]:.3g}',
        f'lograte min/max={stats["lograte_min"]:.3g}/{stats["lograte_max"]:.3g}',
        f'frac u>=5={stats["hi_sat_frac_lograte_ge_5"]:.3g}',
    ]
    ax.text(0.02, 0.98, '\n'.join(lines), va='top', family='monospace', fontsize=10)

    path = OUT / f'{name}_top10_random10_activation_summary.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    idx, px01, px_norm = pick_sequence()
    plot_frames(px01, OUT / 'pusht_sequence_frames.png')
    all_stats = {'dataset_index': idx, 'models': {}}

    for name, ckpt, _kind in RUNS:
        print(f'loading {name}: {ckpt}', flush=True)
        model = swm.wm.utils.load_pretrained(ckpt)
        u, rates, stats = encode_rates(model, px_norm)
        all_stats['models'][name] = stats
        np.savez(OUT / f'{name}_arrays.npz', lograte=u, rate=rates)
        path = plot_model(name, u, rates, stats)
        print(f'wrote {path}', flush=True)

    with (OUT / 'activation_stats.json').open('w') as f:
        json.dump(all_stats, f, indent=2)
    print(f'wrote {OUT / "activation_stats.json"}', flush=True)


if __name__ == '__main__':
    main()
