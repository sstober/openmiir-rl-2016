import logging
log = logging.getLogger('deepthought.datasets')


# Helper function to generate hdf5 subsets
def get_dataset(hdf5name, selectors=None, sources=('features', 'targets', 'subjects')):
    if selectors is None:
        selectors = {}

    # load metadata
    import deepthought.util.fs_util as fs_util
    base_meta = fs_util.load(hdf5name + '.meta.pklz')

    # build lookup structure
    from deepthought.datasets.selection import DatasetMetaDB
    metadb = DatasetMetaDB(base_meta, selectors.keys())

    # get selected trial IDs
    selected_trial_ids = metadb.select(selectors)
    log.debug('selectors: {}'.format(selectors))
    log.debug('selected trials: {}'.format(selected_trial_ids))
    log.debug('selected sources: {}'.format(sources))

    # load data and generate metadata
    from fuel.datasets.hdf5 import H5PYDataset
    hdf5 = H5PYDataset(hdf5name,
                       which_sets=('all',), subset=selected_trial_ids,
                       load_in_memory=True, sources=sources
                       )
    meta = [base_meta[i] for i in selected_trial_ids]

    log.debug('number of examples: {}'.format(hdf5.num_examples))

    return hdf5, meta
