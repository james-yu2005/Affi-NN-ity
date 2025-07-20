import deepchem as dc

tasks, datasets, transformers = dc.molnet.load_pdbbind(
    featurizer='raw',
    set_name='refined',
    splitter='random',
    reload=True
)
train_dataset, valid_dataset, test_dataset = datasets
print(len(train_dataset), len(valid_dataset), len(test_dataset))

# apply DeepChem’s transformers so y is zero‑mean, unit‑variance
for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
    valid_dataset = transformer.transform(valid_dataset)
    test_dataset  = transformer.transform(test_dataset)
