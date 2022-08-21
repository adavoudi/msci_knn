# Open Problems - Multimodal Single-Cell Integration

## KNN Solution - LB 0.808

### Parameters

```python
NUM_NEIGHBOURS = 100
REDUCE_BY = 20
IVIS_DIM = 100
EPOCHS = 10
REDUCE_FUNC = "max"
```

### Compute multi targets

```bash
python calculate_targets.py --tenchnology multi --batch_size 1000 --num_neighbours {NUM_NEIGHBOURS} --reduce_by {REDUCE_BY} --reduce_func {REDUCE_FUNC} --ivis_dim {IVIS_DIM} --epochs {EPOCHS}
```

### Compute cite targets

```bash
!python calculate_targets.py --tenchnology cite --batch_size 10000 --num_neighbours {NUM_NEIGHBOURS} --ivis_dim {IVIS_DIM} --epochs {EPOCHS}
```

### Create submission

```bash
!python create_submission.py 
```
