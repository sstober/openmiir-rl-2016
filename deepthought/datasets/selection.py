import collections
import logging
log = logging.getLogger('deepthought.datasets')


class DatasetMetaDB(object):
    def __init__(self, metadata, attributes):

        def multi_dimensions(n, dtype):
            """ Creates an n-dimension dictionary where the n-th dimension is of type 'type'
            """
            if n == 0:
                return dtype()
            return collections.defaultdict(lambda: multi_dimensions(n - 1, dtype))

        metadb = multi_dimensions(len(attributes), list)

        for i, meta in enumerate(metadata):
            def add_entry(subdb, remaining_attributes):
                if len(remaining_attributes) == 0:
                    subdb.append(i)
                else:
                    key = meta[remaining_attributes[0]]
                    #                 print remaining_attributes[0], key
                    add_entry(subdb[key], remaining_attributes[1:])

            add_entry(metadb, attributes)

        self.metadb = metadb
        self.attributes = attributes

    def select(self, selectors_dict):

        def _apply_selectors(sel, node):
            if isinstance(node, dict):
                selected = []
                keepkeys = sel[0]
                for key, value in node.items():
                    if keepkeys == 'all' or key in keepkeys:
                        selected.extend(_apply_selectors(sel[1:], value))
                return selected
            else:
                return node  # [node]

        selectors = []
        for attribute in self.attributes:
            if attribute in selectors_dict:
                selectors.append(selectors_dict[attribute])
            else:
                selectors.append('all')

        return sorted(_apply_selectors(selectors, self.metadb))


def generate_selector_map(dataset, source, name='selector_map', verbose=False):
    import numpy as np
    import theano

    state = dataset.open()
    request = slice(0, dataset.num_examples)
    selector_values = dataset.get_data(request=request)[dataset.sources.index(source)]
    dataset.close(state)

    # print selector_values
    selector_values = np.unique(selector_values)
    if verbose:
        log.debug('selector values for source "{}": {}'.format(source, selector_values))

    selector_map = np.empty(selector_values.max() + 1, selector_values.dtype)
    selector_map[:] = -1  # on purpose to generate indexing errors (hopefully)
    for i, v in enumerate(selector_values):
        selector_map[v] = i
    selector_map = theano.shared(selector_map, name=name, borrow=False)

    if verbose:
        log.debug('generated selector map "{}"={}, type={}'
                  .format(selector_map, selector_map.get_value(), selector_map.type))
    return selector_map
