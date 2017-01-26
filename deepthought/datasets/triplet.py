# generate triplet index datasets
from fuel.datasets import Dataset
import numbers
import numpy as np
import logging
log = logging.getLogger('deepthought.datasets')


class TripletsIndexDataset(Dataset):
    def __init__(self,
                 dataset, dataset_metadata,
                 base_selectors=None, ext_selectors=None,
                 targets_source='targets',
                 group_attribute=None, allow_self_comparison=False,
                 additional_sources=None,
                 **kwargs):

        if base_selectors is None:
            base_selectors = {}

        if additional_sources is None:
            additional_sources = []

        # get selected trial IDs
        from deepthought.datasets.selection import DatasetMetaDB
        metadb = DatasetMetaDB(dataset_metadata, base_selectors.keys())
        base_trial_ids = metadb.select(base_selectors)
        log.debug('base selectors: {}'.format(base_selectors))
        log.debug('selected base trials: {}'.format(base_trial_ids))

        if ext_selectors is not None:
            split_index = len(base_trial_ids)
            metadb = DatasetMetaDB(dataset_metadata, ext_selectors.keys())
            ext_trial_ids = metadb.select(ext_selectors)
        else:
            split_index = 0
            ext_trial_ids = []

        log.debug('ext selectors: {}'.format(ext_selectors))
        log.debug('selected ext trials: {}'.format(ext_trial_ids))

        # indices = np.concatenate((base_trial_ids, ext_trial_ids))
        indices = base_trial_ids + ext_trial_ids
        metadata = [dataset_metadata[i] for i in indices]

        # load targets from dataset
        state = dataset.open()
        targets = dataset.get_data(state=state, request=indices)[dataset.sources.index(targets_source)]
        dataset.close(state)
        # print targets

        # split data into partitions according to
        groups = dict()
        if group_attribute is not None:
            for i, meta in enumerate(metadata):
                group = meta[group_attribute]
                if group not in groups:
                    groups[group] = []
                groups[group].append(i)
        else:
            # default: all in one group
            groups['default'] = np.arange(len(metadata))
        # print groups

        from itertools import product
        pairs = []
        others = []
        # add group-wise
        for group_ids in groups.values():
            for i in range(targets.shape[-1]):
                # 1st trial candidates
                if split_index > 0:
                    trial_ids = np.where(targets[:split_index, i] == 1)[0]
                else:
                    trial_ids = np.where(targets[:, i] == 1)[0]
                # 2nd trial candidates (same class)
                trial_ids2 = np.where(targets[:, i] == 1)[0]
                # others candidates (different class)
                others_ids = np.where(targets[:, i] == 0)[0]

                # only retain ids within the group
                trial_ids = np.intersect1d(trial_ids, group_ids)
                trial_ids2 = np.intersect1d(trial_ids2, group_ids)
                others_ids = np.intersect1d(others_ids, group_ids)

                # combine with permutation
                new_pairs = [tuple(pair) for pair in product(trial_ids, trial_ids2)]
                if not allow_self_comparison:  # remove repetitions
                    to_remove = []
                    for pair in new_pairs:
                        if pair[0] == pair[1]:
                            to_remove.append(pair)
                    for pair in to_remove:
                        new_pairs.remove(pair)
                    # print 'removed', to_remove
                    new_pairs = sorted(new_pairs)
                    # print pairs

                # combine all pairs with all other trials
                for pair, other in product(new_pairs, others_ids):
                    # print pair, other
                    pairs.append(pair)
                    others.append([other])

        # NOTE: triplets uses internal ids
        #   (refencing into indices which contains hdfs-specific ids)
        self.triplets = np.concatenate([pairs, others], axis=1)
        self.indices = np.asarray(indices, dtype=np.int16)
        # indices = indices.reshape((len(indices), 1)) # make 2D for VectorSpace
        log.debug('triplets.shape={} indices.shape={}'.format(self.triplets.shape, self.indices.shape))

        sources = ['targets', '0_indices', '1_indices', '2_indices']

        self.data_per_source = dict()
        for source in additional_sources:
            # load source data from hdf5 dataset
            # hdf5 = H5PYDataset(hdf5name, which_sets=('all',),
            #                   load_in_memory=True, sources=(source,)
            #                )
            # state = hdf5.open()
            # self.data_per_source[source] = hdf5.get_data(request=indices)[0]
            # hdf5.close(state)

            # load source data from dataset
            state = dataset.open()
            self.data_per_source[source] = dataset.get_data(state=state, request=indices)[dataset.sources.index(source)]
            dataset.close(state)

            for i in range(3):
                sources.append('{}_{}'.format(i, source))

        self.sources = tuple(sources)
        self.provides_sources = self.sources
        log.debug('sources: {}'.format(self.sources))
        super(TripletsIndexDataset, self).__init__(**kwargs)

    @property
    def num_examples(self):
        return len(self.triplets)

    # state is ignored, request provides (internal, not hdf5-specific) indices
    def get_data(self, state=None, request=None):
        if not isinstance(request, (numbers.Integral, slice, list)):
            raise ValueError()
        if isinstance(request, numbers.Integral):
            request = [request]
        elif type(request) is slice:
            request = np.arange(request.start, request.stop)

        rval = []
        for so in self.sources:
            # print so
            if so == 'targets':
                # 1st pair always belongs to the same class (if not shuffled)
                y = np.zeros((len(request), 2), dtype=np.int)
                y[:, 0] = 1
                batch = y
            else:
                split = so.index('_')
                i = int(so[:split])
                s = so[split + 1:]

                # print indexes
                trial_ids = self.triplets[request, i]
                # print i, s, trial_ids

                if s == 'indices':
                    batch = self.indices[trial_ids]
                else:
                    # print self.data_per_source[s].shape
                    batch = self.data_per_source[s][trial_ids]
                    # print batch.shape

            # print so, batch.shape
            rval.append(batch)
        return tuple(rval)
