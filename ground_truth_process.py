import argparse
import sys
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Groundtruth.csv from a tab-delimited input file. "
                    "Required columns: contig, position, mod_ratio. "
                    "If y is present, it will be used; otherwise it will be computed "
                    "as y = 1 if mod_ratio > 0, else y = 0.")
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the input tab-delimited CSV file.')
    parser.add_argument('-o', '--output', default='Groundtruth.csv',
                        help='Output CSV file name. Default: Groundtruth.csv')
    return parser.parse_args()


def normalize_columns(df):
    # Normalize column names to canonical lowercase identifiers
    canonical_names = ['contig', 'position', 'mod_ratio', 'y']
    for canonical in canonical_names:
        matches = [c for c in df.columns if c.lower() == canonical]
        if matches:
            found = matches[0]
            if found != canonical:
                df.rename(columns={found: canonical}, inplace=True)
    return df


def main():
    args = parse_args()

    # Read input as tab-delimited CSV
    try:
        df = pd.read_csv(args.input, sep='\t', header=0)
    except Exception as e:
        sys.exit(f"Error reading input file: {e}")

    df = normalize_columns(df)

    # Validate required columns
    cols = set(df.columns)
    required = {'contig', 'position', 'mod_ratio'}
    if not required.issubset(cols):
        sys.exit("Input file must contain columns: contig, position, and mod_ratio "
                 "(case-insensitive).")

    # If y exists, use it; otherwise compute from mod_ratio
    if 'y' in df.columns:
        out_df = df[['contig', 'position', 'mod_ratio', 'y']].copy()
    else:
        df['mod_ratio'] = pd.to_numeric(df['mod_ratio'], errors='coerce')
        df['y'] = (df['mod_ratio'] > 0.5).astype(int)
        out_df = df[['contig', 'position', 'mod_ratio', 'y']].copy()

    # Determine output path: if not provided, save in the input file's directory

    input_dir = os.path.dirname(os.path.abspath(args.input))
    output_path = os.path.join(input_dir, 'Groundtruth.csv')

    # Write output to CSV (default comma separator)
    try:
        out_df.to_csv(output_path, sep='\t',index=False)
        print(f"Groundtruth saved to: {output_path}")
    except Exception as e:
        sys.exit(f"Error writing output file: {e}")


if __name__ == '__main__':
    main()
