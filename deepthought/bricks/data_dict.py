import logging
log = logging.getLogger(__name__)


def generate_data_dict(dataset, source, name='dict', verbose=False):
    import numpy as np
    import theano
    dtype = theano.config.floatX

    # get data into a dict, need to use the full dataset (no subset!)
    state = dataset.open()
    request = slice(0, dataset.num_examples)
    data_dict = dataset.get_data(request=request)[dataset.sources.index(source)]
    dataset.close(state)

    # FIXME: move this to original dataset generator code
    #data_dict = np.rollaxis(data_dict, 3, 1)  # convert b01c format into bc01 format

    shape = data_dict.shape
    data_dict = theano.shared(theano._asarray(data_dict, dtype=dtype),  # for GPU usage
                              name=name, borrow=False)

    if verbose:
        log.debug('generated data dict "{}", shape={}, type={}'
                  .format(data_dict, shape, data_dict.type))
    return data_dict
