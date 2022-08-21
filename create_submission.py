import argparse
import warnings
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

import utils

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Create the submission")

parser.add_argument("--data_dir", type=str, default=".")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--random_state", type=int, default=47)

args = parser.parse_args()

utils.set_seed(args.random_state)

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
data_dir = Path(args.data_dir)


logger.info("loading multi targets")
test_multi_target_df = pd.read_feather(output_dir / "test_multi_target.feather")
logger.info("melting multi targets")
test_multi_target_row_wise_df = pd.melt(
    test_multi_target_df, id_vars=["cell_id"], var_name="gene_id", value_name="target"
)

logger.info("loading cite targets")
test_cite_target_df = pd.read_feather(output_dir / "test_cite_target.feather")
logger.info("melting cite targets")
test_cite_target_row_wise_df = pd.melt(
    test_cite_target_df, id_vars=["cell_id"], var_name="gene_id", value_name="target"
)

logger.info("merging multi and cite targets")
pred_df = pd.concat(
    [test_cite_target_row_wise_df, test_multi_target_row_wise_df], ignore_index=True
)

logger.info("garbage collection")
test_cite_target_row_wise_df = None
test_multi_target_row_wise_df = None
test_cite_target_df = None
test_multi_target_df = None
gc.collect()

logger.info("loading the evaluation ids")
df_evaluation = pd.read_csv(data_dir / "evaluation_ids.csv")

logger.info("setting index for df_evaluation")
df_evaluation = df_evaluation.set_index(["cell_id", "gene_id"])

logger.info("setting index for pred_df")
pred_df = pred_df.set_index(["cell_id", "gene_id"])

logger.info("joining pred_df and df_evaluation")
eval_df = df_evaluation.join(pred_df)

logger.info("Creating the submission")
eval_df.to_csv(output_dir / "submission.csv", index=False)

logger.info("Finished")
