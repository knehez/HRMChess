
import torch
import time
import argparse
import os

def merge_game_history_datasets(paths, output_path):
    """
    Merge multiple game_history_dataset.pt files (new format) into one, concatenating their 'data' lists.
    Args:
        paths: list of input .pt file paths
        output_path: output .pt file path
    """
    all_data = []
    sources = []
    history_lengths = set()
    total_loaded = 0
    for path in paths:
        print(f"Loading: {path}")
        d = torch.load(path, weights_only=False)
        if 'data' in d:
            all_data.extend(d['data'])
            total_loaded += len(d['data'])
            info = d.get('info', {})
            sources.append(info.get('source', path))
            hl = info.get('history_length', None)
            if hl is not None:
                history_lengths.add(hl)
        else:
            print(f"Warning: {path} does not contain 'data' key!")
    
    # Deduplicate based on starting_fen and moves
    print("\n🔍 Deduplicating positions...")
    print(f"   Before deduplication: {len(all_data):,} positions")
    
    unique_positions = {}
    for pos in all_data:
        # Create unique key from starting_fen and moves
        key = (pos['starting_fen'], tuple(pos['moves']))
        if key not in unique_positions:
            unique_positions[key] = pos
    
    deduplicated_data = list(unique_positions.values())
    duplicates_removed = len(all_data) - len(deduplicated_data)
    
    print(f"   After deduplication: {len(deduplicated_data):,} positions")
    print(f"   Duplicates removed: {duplicates_removed:,} ({duplicates_removed/len(all_data)*100:.1f}%)")
    
    merged_info = {
        'created': time.time(),
        'source': 'MERGED + DEDUPLICATED: ' + ', '.join(sources),
        'total_positions': len(deduplicated_data),
        'original_positions_before_dedup': len(all_data),
        'duplicates_removed': duplicates_removed,
        'history_length': history_lengths.pop() if len(history_lengths) == 1 else list(history_lengths),
        'data_format': 'compact_game_history (with embedded score)',
        'merged_files': paths
    }
    torch.save({'data': deduplicated_data, 'info': merged_info}, output_path)
    print(f"\n✅ Merged and deduplicated dataset saved to: {output_path}")
    print(f"   Final positions: {len(deduplicated_data):,}")
    print(f"   Sources: {sources}")
    print(f"   File size reduction: {duplicates_removed/len(all_data)*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge game_history_dataset.pt files.")
    parser.add_argument('inputs', nargs='+', help='Input .pt files to merge')
    parser.add_argument('-o', '--output', default='merged_game_history_dataset.pt', help='Output .pt file')
    args = parser.parse_args()
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            exit(1)
    merge_game_history_datasets(args.inputs, args.output)
