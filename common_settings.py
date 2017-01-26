base_selectors = dict()

SEED = 42
hdf5name = 'data/OpenMIIR-Perception-512Hz.hdf5'

# defaults
hyper_params = dict(
    only_1_inner_fold=False,  # only use this for dev !
    use_ext_dataset_for_validation=True,
    
    #group_attribute='subject',
    group_attribute='within_subject_tuples_group', # smaller
    #group_attribute='cross_subject_tuples_group', # many more combinations    
    
    pretrain_target_source='targets',
    classification_target_source='targets',
        
    num_components=1,
    filter_width_time=1,
    filter_width_freq=1,
    pool_width_time=1,
    pool_stride_time=1,
    pool_width_freq=1,
    pool_stride_freq=1,
    use_bias=False,
    
    max_epochs=10,
    batch_size=1000,
    step_rule='Adam',
#     step_rule='RMSProp',
    learning_rate=0.01,  # Adam default=0.002,
    l1wdecay=0,
    
    classifier_l1wdecay=0.,
    classifier_max_epochs=100,
    classifier_batch_size=120, #120,
    classifier_pool_width=1,
    classifier_pool_stride=1,
    
    blocks_extensions_printing=False,  # disable for less verbose output
    blocks_extensions_progressbar=True,
    blocks_extensions_train_monitoring_channels=[],
    blocks_extensions_valid_monitoring_channels=[],
    
    blocks_extensions_bokeh=False,  # enable for plotting with bokeh
    blocks_extensions_bokeh_plot_title='',
    blocks_extensions_bokeh_channels=[['cost_train','cost_valid'],['total_gradient_norm_train']],
)

verbose = True

import theano
#print("using device: ".format(theano.config.device))

import logging
logging.basicConfig(level=logging.INFO)
#logging.getLogger('deepthought.datasets').setLevel(logging.DEBUG) # debug dataset
logging.getLogger('blocks').setLevel(logging.WARN)
log = logging.getLogger('deepthought')

import numpy as np
np.set_printoptions(precision=4)

# print the blocks configuration
from blocks.config import config
config.default_seed = SEED
print 'blocks default_seed:', config.config['default_seed']

if hyper_params['blocks_extensions_bokeh']:
    # log into bokeh server
    from bokeh.plotting import Session
    session = Session(root_url='http://localhost:5006/')
    session.login('user', 'password')  # FIXME
    
def print_hyper_params():
    for k in sorted(hyper_params.keys()):
        print k, '=', hyper_params[k]