"""Convert llm.c/modded-nanoGPT GPT-2 shards to native Megatron IndexedDataset.

Each source shard is one packed stream entry, not a recovered web document.
No tokenization, EOD insertion/removal, or token reordering is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_shard(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        header = np.frombuffer(stream.read(1024), dtype='<i4')
    if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
        raise ValueError(f'{path}: expected llm.c magic 20240520 and version 1')
    count = int(header[2])
    if count < 2 or path.stat().st_size != 1024 + count * 2:
        raise ValueError(f'{path}: payload size does not match header token count')
    tokens = np.memmap(path, dtype='<u2', mode='r', offset=1024, shape=(count,))
    maximum = int(tokens.max())
    if maximum > 50256:
        raise ValueError(f'{path}: token {maximum} outside GPT-2 vocabulary')
    return tokens, {'path': str(path), 'sha256': sha256(path), 'tokens': count,
                    'min_token': int(tokens.min()), 'max_token': maximum,
                    'eod_count': int(np.count_nonzero(tokens == 50256))}


def convert(paths, prefix):
    builder = IndexedDatasetBuilder(str(prefix) + '.bin', dtype=np.uint16)
    sources = []
    for path in paths:
        tokens, info = inspect_shard(path)
        builder.add_document(tokens, [len(tokens)])
        sources.append(info)
        del tokens
    builder.finalize(str(prefix) + '.idx')
    # Verify every token through the official reader, with bounded comparison memory.
    dataset = IndexedDataset(str(prefix))
    if len(dataset) != len(paths):
        raise AssertionError('Unexpected sequence count')
    for index, path in enumerate(paths):
        tokens, _ = inspect_shard(path)
        if int(dataset.sequence_lengths[index]) != len(tokens):
            raise AssertionError('Unexpected sequence length')
        for start in range(0, len(tokens), 1_000_000):
            length = min(1_000_000, len(tokens) - start)
            if not np.array_equal(dataset.get(index, offset=start, length=length), tokens[start:start+length]):
                raise AssertionError(f'Token mismatch at {path}:{start}')
        del tokens
    return {'sources': sources, 'token_count': sum(s['tokens'] for s in sources),
            'bin_sha256': sha256(str(prefix) + '.bin'),
            'idx_sha256': sha256(str(prefix) + '.idx'), 'all_tokens_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--valid', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve() for p in args.train + args.valid]
    if len(set(paths)) != len(paths):
        parser.error('Duplicate shards or train/validation path overlap')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory; refusing to overwrite data')
    # Check both splits before writing any output.
    hashes = []
    for path in paths:
        tokens, info = inspect_shard(path)
        hashes.append(info['sha256'])
        del tokens
    if len(set(hashes)) != len(hashes):
        parser.error('Duplicate content detected across input shards')
    out.mkdir(parents=True, exist_ok=True)
    result = {'format': 'megatron_indexed_uint16', 'tokenizer': 'gpt2',
              'eod_id': 50256, 'model_vocab_size': 50304,
              'packing': 'one unchanged packed token stream per source shard',
              'train': convert(args.train, out / 'train'),
              'valid': convert(args.valid, out / 'valid')}
    (out / 'manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
