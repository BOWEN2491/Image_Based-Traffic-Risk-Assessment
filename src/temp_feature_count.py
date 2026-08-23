"""Print the row count of a feature CSV."""

import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    print(len(pd.read_csv(args.csv_path)))
