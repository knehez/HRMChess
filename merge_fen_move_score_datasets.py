import torch
import time
import argparse
import os

# --- Chess dataset merger ---
def merge_fen_move_score_datasets(paths, output_path):
    """
    Merge multiple fen_move_score_dataset.pt files into one, deduplicating by FEN.
    Args:
        paths: list of input .pt file paths
        output_path: output .pt file path
    """
    merged_dict = {}
    sources = []
    total_loaded = 0
    for path in paths:
        print(f"Loading: {path}")
        data = torch.load(path,weights_only=False)
        vec = data['data']
        info = data.get('info', {})
        sources.append(info.get('source', path))
        total_loaded += len(vec)
        for fen, score in vec:
            merged_dict[fen] = score  # last score wins if duplicate
    merged_vec = [(fen, score) for fen, score in merged_dict.items()]
    print(f"\nLoaded {total_loaded:,} positions from {len(paths)} files.")
    print(f"Deduplicated: {len(merged_vec):,} unique positions.")
    merged_info = {
        'data': merged_vec,
        'info': {
            'created': time.time(),
            'source': 'Merged: ' + ' + '.join(sources),
            'total_positions': len(merged_vec),
            'data_format': '(fen, score)',
            'merged_files': paths
        }
    }
    torch.save(merged_info, output_path)
    print(f"\n✅ Merged dataset saved to: {output_path}")
    print(f"   Positions: {len(merged_vec):,}")
    print(f"   Sources: {sources}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge fen_move_score_dataset.pt files.")
    parser.add_argument('inputs', nargs='+', help='Input .pt files to merge')
    parser.add_argument('-o', '--output', default='fen_move_score_dataset_merged.pt', help='Output .pt file')
    args = parser.parse_args()
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            exit(1)
    merge_fen_move_score_datasets(args.inputs, args.output)
