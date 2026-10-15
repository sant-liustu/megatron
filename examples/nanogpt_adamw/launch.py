"""Launch the official GPT pretraining entry with the w768 AdamW recipe."""
import argparse
import hashlib
import json
import platform
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=8, help='GPUs per node')
    parser.add_argument('--micro-batch-size', type=int, default=64, help='Sequences per GPU per microbatch')
    parser.add_argument('--transformer-impl', choices=('local', 'transformer_engine'), default='transformer_engine')
    parser.add_argument('--data-dir', type=Path, required=True, help='Directory produced by prepare_data.py')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--exit-interval', type=int, help='Save and stop after this many iterations')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    layers, width, heads, vocab, seq = 12, 768, 6, 50304, 1024
    steps, batch, warmup, decay = 5100, 512, 250, 1450
    args.data_dir = args.data_dir.resolve()
    manifest_path = args.data_dir / 'manifest.json'
    if not manifest_path.is_file():
        parser.error('--data-dir requires prepare_data.py manifest.json')
    data_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (data_manifest.get('format') != 'megatron_indexed_uint16'
            or data_manifest.get('eod_id') != 50256
            or data_manifest.get('model_vocab_size') != 50304):
        parser.error('Unsupported data manifest')
    for split in ('train', 'valid'):
        for suffix in ('.bin', '.idx'):
            if not (args.data_dir / (split + suffix)).is_file():
                parser.error(f'Missing {split}{suffix}')
    impl = args.transformer_impl
    nnodes = int(os.environ.get('NNODES', '1'))
    node_rank = int(os.environ.get('NODE_RANK', '0'))
    if min(args.gpus, args.micro_batch_size, nnodes) < 1:
        parser.error('GPU, node and micro batch counts must be positive')
    if batch % (args.gpus * nnodes * args.micro_batch_size):
        parser.error('global batch must be divisible by GPUs * nodes * micro batch')
    out = args.output.resolve()
    checkpoint = out / 'checkpoints'
    train_args = [
        '--num-layers', str(layers), '--hidden-size', str(width),
        '--num-attention-heads', str(heads), '--ffn-hidden-size', str(4 * width),
        '--seq-length', str(seq), '--max-position-embeddings', str(seq),
        '--normalization', 'RMSNorm', '--norm-epsilon', '1e-5', '--qk-layernorm',
        '--squared-relu', '--disable-bias-linear', '--position-embedding-type', 'rope',
        '--rotary-base', '10000', '--rotary-percent', '1.0',
        '--hidden-dropout', '0', '--attention-dropout', '0', '--init-method-std', '0.02',
        '--tensor-model-parallel-size', '1', '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1', '--transformer-impl', impl,
        '--micro-batch-size', str(args.micro_batch_size), '--global-batch-size', str(batch),
        '--train-iters', str(steps), '--optimizer', 'adam', '--lr', '0.0018',
        '--decoupled-lr', '0.0036', '--min-lr', '0', '--decoupled-min-lr', '0',
        '--adam-beta1', '0.9', '--adam-beta2', '0.95', '--adam-eps', '1e-8',
        '--weight-decay', '0.1', '--clip-grad', '0',
        '--lr-decay-style', 'WSD', '--lr-warmup-iters', str(warmup),
        '--lr-decay-iters', str(steps), '--lr-wsd-decay-iters', str(decay),
        '--lr-wsd-decay-style', 'linear',
        '--tokenizer-type', 'NullTokenizer', '--vocab-size', str(vocab),
        '--make-vocab-size-divisible-by', '1', '--num-workers', '0',
        '--seed', '1234', '--log-interval', '10',
        '--eval-iters', '20', '--eval-interval', '500',
        '--save-interval', '1000', '--save', str(checkpoint),
        '--ckpt-format', 'torch', '--data-cache-path', str(out / 'data_cache'),
        '--no-gradient-accumulation-fusion', '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion', '--no-bias-dropout-fusion', '--no-rope-fusion',
        '--no-persist-layer-norm',
    ]
    train_args += ['--train-data-path', str(args.data_dir / 'train'),
                   '--valid-data-path', str(args.data_dir / 'valid'),
                   '--null-tokenizer-eod-id', '50256', '--bf16']
    if args.exit_interval is not None:
        if args.exit_interval < 1:
            parser.error('--exit-interval must be positive')
        train_args += ['--exit-interval', str(args.exit_interval)]
    if args.resume:
        original = out / ('launch.json' if nnodes == 1 else f'launch-node{node_rank}.json')
        if original.is_file():
            previous = json.loads(original.read_text(encoding='utf-8'))
            if previous.get('data_manifest') != data_manifest:
                parser.error('Resume dataset differs from the original launch')
        if not (checkpoint / 'latest_checkpointed_iteration.txt').is_file():
            parser.error('--resume requires an existing checkpoint tracker')
        train_args += ['--load', str(checkpoint)]
    elif checkpoint.exists() and any(checkpoint.iterdir()):
        parser.error('checkpoint directory is nonempty; use --resume or a new output')
    distributed = (['--standalone'] if nnodes == 1 else [
        '--nnodes', str(nnodes), '--node-rank', str(node_rank),
        '--master-addr', os.environ['MASTER_ADDR'],
        '--master-port', os.environ.get('MASTER_PORT', '6000')])
    cmd = [sys.executable, '-m', 'torch.distributed.run', *distributed,
           '--nproc-per-node', str(args.gpus), str(ROOT / 'pretrain_gpt.py'), *train_args]
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    label = 'resume' if args.resume else 'launch'
    if nnodes > 1:
        label += f'-node{node_rank}'
    (out / f'{label}.diff').write_text(subprocess.check_output(
        ['git', 'diff', 'HEAD'], cwd=ROOT, text=True, encoding='utf-8'), encoding='utf-8')
    source_dir = out / f'{label}_source'
    source_dir.mkdir(exist_ok=True)
    hashes = {}
    for source in (Path(__file__), Path(__file__).with_name('prepare_data.py')):
        content = source.read_bytes()
        hashes[source.name] = hashlib.sha256(content).hexdigest()
        (source_dir / source.name).write_bytes(content)
    (out / f'{label}.json').write_text(json.dumps({
        'argv': cmd, 'recipe': 'w768_adamw', 'data': str(args.data_dir),
        'data_manifest': data_manifest,
        'python': sys.version, 'platform': platform.platform(), 'source_sha256': hashes,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
    }, indent=2), encoding='utf-8')
    with (out / f'{label}.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
