
#### Usage

```shell
python3 -m src.train
```

#### Dataset

Unfortunately we need to break abstractions a little, in that the dataset needs to know the model anchors in order to produce valid targets.
The advantage of doing the target transformation in the dataset is that we can take advantage of PyTorch multi-processing.

Further notes go here.
