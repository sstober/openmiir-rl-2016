def get_function(model, input_sources=None, output_sources=None, **kwargs):
    from theano import function

    if input_sources is None:
        inputs = model.inputs
    else:
        inputs = get_inputs(model, input_sources)

    if output_sources is None:
        outputs = model.outputs
    else:
        outputs = get_outputs(model, output_sources)
        
    if len(outputs) == 1:
        outputs = outputs[0]  # unwrapping for convenience
        
    fn = function(inputs=inputs, outputs=outputs, **kwargs)    
    
    return fn
        
        
def get_inputs(model, sources):
    names = [v.name for v in model.inputs]
    idx = [names.index(s) for s in sources]
    return [model.inputs[i] for i in idx]


def get_outputs(model, sources):
    names = [v.name for v in model.outputs]
    idx = [names.index(s) for s in sources]
    return [model.outputs[i] for i in idx]


def process_dataset(dataset, fn, input_sources,
                    indices=None, batch_size=100,                     
                    target_source=None):
    import numpy as np
    from fuel.schemes import SequentialScheme

    if indices is None:
        scheme = SequentialScheme(dataset.num_examples, batch_size=batch_size)
    else:
        scheme = SequentialScheme(indices, batch_size=batch_size)
    
    source_idx = [dataset.sources.index(s) for s in input_sources]
        
    if target_source is not None: 
        target_idx = dataset.sources.index(target_source)
        output_targets = []
    else:
        target_idx = None
        output_targets = None

    output = []
    state = dataset.open()
    for request in scheme.get_request_iterator():
        data = dataset.get_data(state=state, request=request)
        output.append(fn(*[data[i] for i in source_idx]))
        if target_source is not None: 
            output_targets.append(data[target_idx])
    dataset.close(state)
    # print [o.shape for o in output]
    output = np.concatenate(output)
    
    if target_source is not None: 
        return output, np.concatenate(output_targets)
    return output
