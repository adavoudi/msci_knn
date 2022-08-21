import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

import utils

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(
    description="Calculates the targets for the evaluation"
)

parser.add_argument("--data_dir", type=str, default=".")
parser.add_argument("--tenchnology", type=str, default="multi")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--batch_size", type=int, default=10000)
parser.add_argument("--num_neighbours", type=int, default=1000)
parser.add_argument("--reduce_by", type=int, default=0)
parser.add_argument("--reduce_func", type=str, default="max")
parser.add_argument("--ivis_dim", type=int, default=2)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--ivis_unsupervised", action="store_true")

parser.add_argument("--random_state", type=int, default=47)


args = parser.parse_args()

utils.set_seed(args.random_state)

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
data_dir = Path(args.data_dir)

TRAIN_INPUTS_PATH = data_dir / f"train_{args.tenchnology}_inputs.h5"
TRAIN_TARGETS_PATH = data_dir / f"train_{args.tenchnology}_targets.h5"
TEST_INPUTS_PATH = data_dir / f"test_{args.tenchnology}_inputs.h5"

logger.info("loading the evaluation ids")
df_evaluation = pd.read_csv(data_dir / "evaluation_ids.csv")

logger.info("loading the metadata")
metadata = pd.read_csv(data_dir / "metadata.csv")

logger.info("loading the train inputs")
train_inputs_cell_ids, train_inputs_embed = utils.read_cell_ids(
    TRAIN_INPUTS_PATH,
    df_evaluation=df_evaluation,
    batch_size=args.batch_size,
    reduce_dimension=args.reduce_by > 0,
    reduce_by=args.reduce_by,
    reduce_function=args.reduce_func,
)

metadata_df = train_inputs_cell_ids.set_index("cell_id").join(
    metadata.set_index("cell_id")
)

logger.info("training the ivis model")
ivis_model, train_embeddings = utils.train_ivis(
    train_inputs_embed if train_inputs_embed is not None else TRAIN_INPUTS_PATH,
    metadata_df,
    supervised=not args.ivis_unsupervised,
    ivis_dim=args.ivis_dim,
    epochs=args.epochs,
    k=150,
    build_index_on_disk=True,
    precompute=True,
    n_trees=50,
)

logger.info("transforming test inputs")
test_embeddings = utils.transform(
    TEST_INPUTS_PATH,
    ivis_model,
    df_evaluation=df_evaluation,
    reduce_dimension=args.reduce_by > 0,
    reduce_by=args.reduce_by,
    reduce_function=args.reduce_func,
)

logger.info("calculating the kneighbors for the test inputs")
kneighbors_test = utils.find_knn(
    train_embeddings, test_embeddings, n_neighbors=args.num_neighbours
)

logger.info("loading test cell ids")
test_inputs_cell_ids, _ = utils.read_cell_ids(
    TEST_INPUTS_PATH,
    df_evaluation=df_evaluation,
    batch_size=args.batch_size,
    only_exist_in_eval=True,
)

logger.info("calculating the test targets")
test_target_df = utils.calculate_target(TRAIN_TARGETS_PATH, kneighbors_test)
test_target_df["cell_id"] = test_inputs_cell_ids

logger.info("saving the test targets")
test_target_df.to_feather(output_dir / f"test_{args.tenchnology}_target.feather")

logger.info("Finished")
