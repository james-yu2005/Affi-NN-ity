import deepchem as dc
import numpy as np
from .cleaning import get_forbidden_pairs, remove_invalid_molecules

def load_and_merge_set(set_name):
    print(f"Loading '{set_name}' set...")
    try:
        tasks, datasets, transformers = dc.molnet.load_pdbbind(
            featurizer='raw', set_name=set_name, splitter='index', reload=True
        )
        train, valid, test = datasets

        if len(transformers) > 0:
            transformer = transformers[0]
            def un_transform(ds):
                if len(ds) == 0: return ds
                return dc.data.NumpyDataset(X=ds.X, y=transformer.untransform(ds.y), w=ds.w, ids=ds.ids)
            train, valid, test = un_transform(train), un_transform(valid), un_transform(test)

        X_list = [ds.X for ds in [train, valid, test] if len(ds) > 0]
        y_list = [ds.y for ds in [train, valid, test] if len(ds) > 0]
        w_list = [ds.w for ds in [train, valid, test] if len(ds) > 0]
        id_list = [ds.ids for ds in [train, valid, test] if len(ds) > 0]

        if not X_list: return None, None, None, None
        return (np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0),
                np.concatenate(w_list, axis=0), np.concatenate(id_list, axis=0))

    except Exception as e:
        print(f"Error loading {set_name}: {e}")
        return None, None, None, None

def load_pdbbind_datasets():
    core_data = load_and_merge_set('core')
    refined_data = load_and_merge_set('refined')
    general_data = load_and_merge_set('general')

    core_ds = dc.data.NumpyDataset(X=core_data[0], y=core_data[1], w=core_data[2], ids=core_data[3])
    forbidden = get_forbidden_pairs(core_dataset=core_ds)

    pool_X = np.concatenate([d[0] for d in [refined_data, general_data] if d[0] is not None], axis=0)
    pool_y = np.concatenate([d[1] for d in [refined_data, general_data] if d[1] is not None], axis=0)
    pool_w = np.concatenate([d[2] for d in [refined_data, general_data] if d[2] is not None], axis=0)
    pool_ids = np.concatenate([d[3] for d in [refined_data, general_data] if d[3] is not None], axis=0)

    # Filter out IDs present in Core set
    core_id_set = set(core_data[3])
    unique_indices = [i for i, pid in enumerate(pool_ids) if pid not in core_id_set]
    
    train_ds = dc.data.NumpyDataset(
        X=pool_X[unique_indices], y=pool_y[unique_indices], 
        w=pool_w[unique_indices], ids=pool_ids[unique_indices]
    )

    clean_train = remove_invalid_molecules(train_ds, "Train", forbidden_pairs=forbidden)
    clean_test = remove_invalid_molecules(core_ds, "Test (Core)", forbidden_pairs=None)

    return clean_train, clean_test