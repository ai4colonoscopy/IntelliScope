import pandas as pd
import os
import shutil
import argparse


def organized_dataset(csv_path, base_original_path, base_target_path):
    df = pd.read_csv(csv_path)

    df['original'] = df['original'].apply(lambda x: os.path.normpath(x))
    df['target'] = df['target'].apply(lambda x: os.path.normpath(x))

    missing_files = []

    for index, row in df.iterrows():
        original_rel_path = row['original']
        target_rel_path = row['target']

        original_file = os.path.join(base_original_path, original_rel_path)
        target_file = os.path.join(base_target_path, target_rel_path)

        if not os.path.isfile(original_file):
            print(f"File not found: {original_file}")
            missing_files.append(original_file)
            continue

        target_dir = os.path.dirname(target_file)
        os.makedirs(target_dir, exist_ok=True)

        shutil.copy2(original_file, target_file)

        if os.path.basename(original_file) != os.path.basename(target_file):
            target_new_file = os.path.join(target_dir, os.path.basename(target_file))
            os.rename(target_file, target_new_file)

    if missing_files:
        print("\nThe following original files were not found:")
        for file in missing_files:
            print(file)
    else:
        print("\nAll files were copied successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize the data into the structure of ColonINST")
    parser.add_argument("--csv", type=str, default='cache/data/data_reorganize.csv')
    parser.add_argument("--base_original", type=str, default='cache/data/ori_dataset_download')
    parser.add_argument("--base_target", type=str, default='cache/data/ColonINST/Positive-images')
    args = parser.parse_args()

    organized_dataset(args.csv, args.base_original, args.base_target)