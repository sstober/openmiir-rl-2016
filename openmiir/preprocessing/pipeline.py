__author__ = 'sstober'

import logging
log = logging.getLogger(__name__)

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mne
from mne.io import read_raw_edf
from mne.channels import rename_channels
from mne.preprocessing import ICA, read_ica
from mne.viz.topomap import plot_topomap

import deepthought
from deepthought.util.fs_util import ensure_parent_dir_exists
from deepthought.datasets.eeg.biosemi64 import Biosemi64Layout
from openmiir.eeg import recording_has_mastoid_channels
from openmiir.events import decode_event_id
from openmiir.preprocessing.events import \
    merge_trial_and_audio_onsets, generate_beat_events, \
    simple_beat_event_id_generator, extract_events_from_raw
from openmiir.metadata import get_stimuli_version, load_stimuli_metadata
from mneext.resample import fast_resample_mne

RAW_EOG_CHANNELS = [u'EXG1', u'EXG2', u'EXG3', u'EXG4']
MASTOID_CHANNELS = [u'EXG5', u'EXG6']

def load_raw_info(subject,
             mne_data_root=None,
             verbose=False):

    if mne_data_root is None:
        # use default data root
        import deepthought
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')
        mne_data_root = os.path.join(data_root, 'eeg', 'mne')

    mne_data_filepath = os.path.join(mne_data_root, '{}-raw.fif'.format(subject))

    log.info('Loading raw data info for subject "{}" from {}'.format(subject, mne_data_filepath))
    raw = mne.io.Raw(mne_data_filepath, preload=False, verbose=verbose)
    return raw.info


def load_raw(subject, **args):
    return _load_raw(subject=subject, has_mastoid_channels=recording_has_mastoid_channels, **args)


def _load_raw(subject,
             mne_data_root=None,
             verbose=False,
             onsets=None,
             interpolate_bad_channels=False,
             has_mastoid_channels=None, # None=True, False, or callable(subject) returning True/False
             apply_reference=True, # by default, reference the data
             reference_mastoids=True):

    if mne_data_root is None:
        # use default data root
        import deepthought
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')
        mne_data_root = os.path.join(data_root, 'eeg', 'mne')

    mne_data_filepath = os.path.join(mne_data_root, '{}-raw.fif'.format(subject))

    log.info('Loading raw data for subject "{}" from {}'.format(subject, mne_data_filepath))
    raw = mne.io.Raw(mne_data_filepath, preload=True, verbose=verbose)

    if apply_reference:	    
        if has_mastoid_channels is None \
            or has_mastoid_channels is True \
            or has_mastoid_channels(subject) is True:
            ## referencing to mastoids
            if reference_mastoids:
                log.info('Referencing to mastoid channels: {}'.format(MASTOID_CHANNELS))
                mne.io.set_eeg_reference(raw, MASTOID_CHANNELS, copy=False) # inplace
            else:
                log.info('This recording has unused mastoid channels: {} '
                         'To use them, re-run with reference_mastoids=True.'.format(MASTOID_CHANNELS))
            raw.drop_channels(MASTOID_CHANNELS)
        else:
            ## referencing to average
            log.info('Referencing to average.')
            mne.io.set_eeg_reference(raw, copy=False)

    ## optional event merging
    if onsets == 'audio':
        merge_trial_and_audio_onsets(raw,
                                     use_audio_onsets=True,
                                     inplace=True,
                                     stim_channel='STI 014',
                                     verbose=verbose)
    elif onsets == 'trials':
        merge_trial_and_audio_onsets(raw,
                                     use_audio_onsets=True,
                                     inplace=True,
                                     stim_channel='STI 014',
                                     verbose=verbose)
    # else: keep both

    bads = raw.info['bads']
    if bads is not None and len(bads) > 0:
        if interpolate_bad_channels:
            log.info('Interpolating bad channels: {}'.format(bads))
            raw.interpolate_bads()
        else:
            log.info('This file contains some EEG channels marked as bad: {}\n'
                     'To interpolate bad channels run load_raw() with interpolate_bad_channels=True.'
                     ''.format(bads))

    return raw

def interpolate_bad_channels(inst):
    bads = inst.info['bads']
    if bads is not None and len(bads) > 0:
        log.info('Interpolating bad channels...')
        inst.interpolate_bads()
    else:
        log.info('No channels marked as bad. Nothing to interpolate.')


def load_ica(subject, description, ica_data_root=None):
    if ica_data_root is None:
        # use default data root
        import deepthought
        data_root = os.path.join(deepthought.DATA_PATH, 'OpenMIIR')
        ica_data_root = os.path.join(data_root, 'eeg', 'preprocessing', 'ica')

    ica_filepath = os.path.join(ica_data_root,
                                '{}-{}-ica.fif'.format(subject, description))
    return read_ica(ica_filepath)


def import_and_process_metadata(biosemi_data_root, mne_data_root, subject, verbose=True, overwrite=False):

    ## check whether output already exists
    output_filepath = os.path.join(mne_data_root,
                                   '{}-raw.fif'.format(subject))

    if os.path.exists(output_filepath):
        if not overwrite:
            log.info('Skipping existing {}'.format(output_filepath))
            return

    ## import raw BDF file from biosemi
    bdf_filepath = os.path.join(biosemi_data_root, '{}.bdf'.format(subject))

    ## NOTE: marks EXT1-4 channels as EOG channels during import
    log.info('Importing raw BDF data from: {}'.format(bdf_filepath))
    raw = read_raw_edf(bdf_filepath, eog=RAW_EOG_CHANNELS, preload=True, verbose=verbose)
    log.info('Imported raw data: {}'.format(raw))

    sfreq = raw.info['sfreq']
    if sfreq != 512:
        log.warn('Unexpected sample rate: {} Hz'.format(sfreq))
        log.warn('Re-sampling to 512 Hz')
        fast_resample_mne(raw, 512, res_type='sinc_best', preserve_events=True, verbose=True)

    ## mark all unused channels as bad
    raw.info['bads'] += [u'C1', u'C2', u'C3', u'C4', u'C5', u'C6', u'C7', u'C8', u'C9', u'C10',
                u'C11', u'C12', u'C13', u'C14', u'C15', u'C16', u'C17', u'C18', u'C19', u'C20',
                u'C21', u'C22', u'C23', u'C24', u'C25', u'C26', u'C27', u'C28', u'C29', u'C30',
                u'C31', u'C32', u'D1', u'D2', u'D3', u'D4', u'D5', u'D6', u'D7', u'D8',
                u'D9', u'D10', u'D11', u'D12', u'D13', u'D14', u'D15', u'D16', u'D17', u'D18',
                u'D19', u'D20', u'D21', u'D22', u'D23', u'D24', u'D25', u'D26', u'D27', u'D28',
                u'D29', u'D30', u'D31', u'D32', u'E1', u'E2', u'E3', u'E4', u'E5', u'E6',
                u'E7', u'E8', u'E9', u'E10', u'E11', u'E12', u'E13', u'E14', u'E15',
                u'E16', u'E17', u'E18', u'E19', u'E20', u'E21', u'E22', u'E23', u'E24',
                u'E25', u'E26', u'E27', u'E28', u'E29', u'E30', u'E31', u'E32', u'F1',
                u'F2', u'F3', u'F4', u'F5', u'F6', u'F7', u'F8', u'F9', u'F10', u'F11',
                u'F12', u'F13', u'F14', u'F15', u'F16', u'F17', u'F18', u'F19', u'F20',
                u'F21', u'F22', u'F23', u'F24', u'F25', u'F26', u'F27', u'F28', u'F29',
                u'F30', u'F31', u'F32', u'G1', u'G2', u'G3', u'G4', u'G5', u'G6', u'G7',
                u'G8', u'G9', u'G10', u'G11', u'G12', u'G13', u'G14', u'G15', u'G16', u'G17',
                u'G18', u'G19', u'G20', u'G21', u'G22', u'G23', u'G24', u'G25', u'G26', u'G27',
                u'G28', u'G29', u'G30', u'G31', u'G32', u'H1', u'H2', u'H3', u'H4', u'H5',
                u'H6', u'H7', u'H8', u'H9', u'H10', u'H11', u'H12', u'H13', u'H14', u'H15',
                u'H16', u'H17', u'H18', u'H19', u'H20', u'H21', u'H22', u'H23', u'H24', u'H25',
                u'H26', u'H27', u'H28', u'H29', u'H30', u'H31', u'H32',
                u'EXG7', u'EXG8',
                u'GSR1', u'GSR2', u'Erg1', u'Erg2', u'Resp', u'Plet', u'Temp']
    log.info('Marked unused channels as bad: {}'.format(raw.info['bads']))

    if not recording_has_mastoid_channels(subject):
        raw.info['bads'] += [u'EXG5', u'EXG6']

    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True, stim=True, exclude='bads')

    ## process events
    markers_filepath = os.path.join(biosemi_data_root, '{}_EEG_Data.mat'.format(subject))
    log.info('Processing events, external source: {}'.format(markers_filepath))
    events = extract_events_from_raw(raw, markers_filepath, subject, verbose)
    raw._data[-1,:].fill(0)     # delete data in stim channel
    raw.add_events(events)

    # crop to first event - 1s ... last event + 20s (longer than longest trial)
    onesec = raw.info['sfreq']
    tmin, tmax = raw.times[[events[0,0]-onesec, events[-1,0]+20*onesec]]
    log.info('Cropping raw inplace to {:.3f}s - {:.3f}s'.format(tmin, tmax))
    raw.crop(tmin=tmin, tmax=tmax, copy=False)
    # fix sample offser -> 0
    raw.last_samp -= raw.first_samp
    raw.first_samp = 0

    ensure_parent_dir_exists(output_filepath)
    log.info('Saving raw fif data to: {}'.format(output_filepath))
    raw.save(output_filepath, picks=picks, overwrite=overwrite, verbose=False)

    del raw

    raw = fix_channel_infos(output_filepath, verbose=verbose)

    log.info('Imported {}'.format(raw))
    log.info('Metadata: {}'.format(raw.info))

def fix_channel_infos(mne_data_filepath, verbose=True):

    log.info('Loading raw fif data from: {}'.format(mne_data_filepath))
    raw = mne.io.Raw(mne_data_filepath, preload=True, verbose=verbose)

    raw.info['bads'] = []   # reset bad channels as they have been removed already

    montage = Biosemi64Layout().as_montage()
    log.info('Applying channel montage: {}'.format(montage))

    ## change EEG channel names
    mapping = dict()
    bdf_channel_names = raw.ch_names
    for i, channel_name in enumerate(montage.ch_names):
        log.debug('renaming channel {}: {} -> {}'.format(
            i, bdf_channel_names[i], channel_name))
        mapping[bdf_channel_names[i]] = channel_name
    rename_channels(raw.info, mapping)

    # mne.channels.apply_montage(raw.info, montage) # in mne 0.9
    raw.set_montage(montage) # in mne 0.9
    log.info('Saving raw fif data to: {}'.format(mne_data_filepath))
    raw.save(mne_data_filepath, overwrite=True, verbose=False)

    return raw

def clean_data(mne_data_root, subject, verbose=True, overwrite=False):

    ## check whether output already exists
    output_filepath = os.path.join(mne_data_root,
                                   '{}_filtered-raw.fif'.format(subject))

    if os.path.exists(output_filepath):
        if not overwrite:
            log.info('Skipping existing {}'.format(output_filepath))
            return

    input_filepath = os.path.join(mne_data_root,
                                   '{}-raw.fif'.format(subject))

    raw = mne.io.Raw(input_filepath, preload=True, verbose=verbose)

    ## apply bandpass filter
    raw.filter(0.5, 30, filter_length='10s',
                l_trans_bandwidth=0.1, h_trans_bandwidth=0.5,
                method='fft', iir_params=None,
                picks=None, n_jobs=1, verbose=verbose)

    ensure_parent_dir_exists(output_filepath)
    raw.save(output_filepath, overwrite=overwrite, verbose=False)

