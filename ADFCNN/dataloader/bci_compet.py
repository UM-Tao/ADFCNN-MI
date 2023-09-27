from glob import glob
from tqdm import tqdm
import mne
import torch
import scipy
import numpy as np
from braindecode.datautil.preprocess import exponential_moving_standardize
from dataloader.augmentation import cutcat,  cutcat_2
from typing import List, Union
import numpy as np
from mne.filter import resample
from torch.utils.data import Dataset
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import Preprocessor
from braindecode.datautil.preprocess import preprocess
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.windowers import create_windows_from_events
from filters import load_filterbank, butter_fir_filter


class BCICompet2aIV(torch.utils.data.Dataset):
    def __init__(self, args):
        
        '''
        * 769: Left
        * 770: Right
        * 771: foot
        * 772: tongue
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
        
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/*T*.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E*.gdf'))
        
        label_filelist = sorted(glob(f'{self.base_path}/*T.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if idx != self.target_subject: continue
                    
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)
            
            raw.load_data()
            raw.filter(0., 40., fir_design='firwin')
            raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0, 3
            if not self.is_test:
                event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10}) if idx != 3 \
                else dict({'769': 5,'770': 6,'771': 7,'772': 8})
            else:
                event_id = dict({'783': 7})
            
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 4))
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]
            
            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)



        return data, label
    

    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=8)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=8)
        return data, label
    
    
class BCICompet2bIV(torch.utils.data.Dataset):
    def __init__(self, args):
        '''
        * 769: left
        * 770: right
        '''
        
        import warnings
        warnings.filterwarnings('ignore')
        
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args
        
        self.data, self.label = self.get_brain_data()
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]
        
        if not self.is_test:
            data, label = self.augmentation(data, label)
        
        sample = {'data': data, 'label': label}
        
        return sample
    
    
    def get_brain_data(self):
        filelist = sorted(glob(f'{self.base_path}/*T.gdf')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E.gdf'))
        
        label_filelist = sorted(glob(f'{self.base_path}/*T.mat')) if not self.is_test \
        else sorted(glob(f'{self.base_path}/*E.mat'))
        
        data = []
        label = []
        
        for idx, filename in enumerate(tqdm(filelist)):
            
            if not self.is_test:
                if idx // 3 != self.target_subject: continue
            else:
                if idx // 2 != self.target_subject: continue
                        
            print(f'LOG >>> Filename: {filename}')
            
            raw = mne.io.read_raw_gdf(filename, preload=True)
            events, annot = mne.events_from_annotations(raw)

            raw.load_data()
            raw.filter(0., 40., fir_design='firwin')
            raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
            
            picks = mne.pick_types(raw.info,
                                    meg=False,
                                    eeg=True,
                                    eog=False,
                                    stim=False,
                                    exclude='bads')
            
            tmin, tmax = 0., 3.
            if not self.is_test: event_id = dict({'769': annot['769'], '770': annot['770']})
            else: event_id = dict({'783': annot['783']})
                
            epochs = mne.Epochs(raw,
                                events,
                                event_id,
                                tmin,
                                tmax,
                                proj=True,
                                picks=picks,
                                baseline=None,
                                preload=True)
            
            if self.downsampling != 0:
                epochs = epochs.resample(self.downsampling)
            self.fs = epochs.info['sfreq']
            
            epochs_data = epochs.get_data() * 1e6
            splited_data = []
            for epoch in epochs_data:
                normalized_data = exponential_moving_standardize(epoch, init_block_size=int(raw.info['sfreq'] * 4))
                splited_data.append(normalized_data)
            splited_data = np.stack(splited_data)
            splited_data = splited_data[:, np.newaxis, ...]

            label_list = scipy.io.loadmat(label_filelist[idx])['classlabel'].reshape(-1) - 1
            
            if len(data) == 0:
                data = splited_data
                label = label_list
            else:
                data = np.concatenate((data, splited_data), axis=0)
                label = np.concatenate((label, label_list), axis=0)
        
        return data, label
            
    
    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=10)
        return data, label
    
class OpenBMI(torch.utils.data.Dataset):
    """
    Not supported subject-independent manner not yet.
    Therefore, we recommend session-to-session manner with single subject.
    """

    def __init__(self, args):
        import warnings
        warnings.filterwarnings('ignore')
        self.base_path = args.BASE_PATH
        self.target_subject = args.target_subject
        self.is_test = args.is_test
        self.downsampling = args.downsampling
        self.args = args

        self.data, self.label = self.get_brain_data()

    def get_brain_data(self):

        x_bundle, y_bundle = [], []
        for (low_hz, high_hz) in [[0, 40]]:
            x_list = []
            y_list = []
            # Load data from MOABBDataset
            dataset = MOABBDataset(dataset_name="Lee2019_MI", subject_ids=self.target_subject+1)

            # Preprocess data
            factor_new = 1e-3
            init_block_size = 1000

        preprocessors = [
            # Keep only EEG sensors
            Preprocessor(fn='pick_types', eeg=True, meg=False, stim=False, apply_on_array=True),
            # Convert from volt to microvolt
            Preprocessor(fn=lambda x: x * 1e+06, apply_on_array=True),
            # Apply bandpass filtering
            Preprocessor(fn='filter', l_freq=low_hz, h_freq=high_hz, apply_on_array=True),
            # Apply exponential moving standardization
            Preprocessor(fn=exponential_moving_standardize, factor_new=factor_new,
                         init_block_size=init_block_size, apply_on_array=True)
        ]
        preprocess(dataset, preprocessors)

        # Check sampling frequency
        sfreq = dataset.datasets[0].raw.info['sfreq']
        if not all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets]):
            raise ValueError("Not match sampling rate.")

        # Divide data by trial
        trial_start_offset_samples = int(0 * sfreq)

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True
        )


        # Make session-to-session data (subject dependent)
        if self.is_test == False:
            for trial in windows_dataset.split('session')['session_1']:
                x_list.append(trial[0])
                y_list.append(trial[1])
        else:
            for trial in windows_dataset.split('session')['session_2']:
                x_list.append(trial[0])
                y_list.append(trial[1])

        # Return numpy array
        x_list = np.array(x_list)
        y_list = np.array(y_list)

        # Cut time points
        tmin, tmax = 0., 3.0
        x_list = x_list[..., int(tmin * sfreq): int(tmax * sfreq)]

        # Resampling
        if self.args.downsampling is not None:
            x_list = resample(np.array(x_list, dtype=np.float64), self.args.downsampling / sfreq)

        x_bundle.append(x_list)
        y_bundle.append(y_list)

        data = np.stack(x_bundle, axis=1)
        data = data[:, :, 20:40, :]
        label = np.array(y_bundle[0])
        return data, label

    def augmentation(self, data, label):

        negative_data_indices = np.where(self.label != label)[0]
        negative_data_index = np.random.choice(negative_data_indices)
        # data, label = cutcat(data, label, self.data[negative_data_index, ...], self.label[negative_data_index], self.args.num_classes, ratio=10)
        data, label = cutcat_2(data, label, self.data[negative_data_index, ...], self.label[negative_data_index],
                             self.args.num_classes, ratio=10)
        return data, label
    def __len__(self):
        return len(self.data)

    # def __getitem__(self, idx):
    #     sample = [self.x[idx], self.y[idx]]
    #     return sample

    def __getitem__(self, idx):
        data = self.data[idx, ...]
        label = self.label[idx]

        if not self.is_test:
            data, label = self.augmentation(data, label)

        sample = {'data': data, 'label': label}

        return sample



def get_dataset(config_name, args):
    
    if 'bcicompet2a_config' in config_name:
        dataset = BCICompet2aIV(args)
        if args['filter_bank']:
            #### FBCNet####
            # data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1], len(args['bank']),
            #                             dataset.data.shape[2], dataset.data.shape[3]))
            #
            # for num, Fband in enumerate(args['bank']):
            #     bw = np.array(Fband)
            #     filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
            #     X_filtered = np.zeros_like(dataset.data)
            #     for i, trial in enumerate(dataset.data):
            #         # filtering
            #         trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            #         trail_filter = trail_filter.reshape(1, 22, 751)
            #         X_filtered[i, :, :, :] = trail_filter
            #     data_filterbank[:, :, num, :, :] = X_filtered

            #### IFNet####
            data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1],2*dataset.data.shape[2], dataset.data.shape[3]))

            for num, Fband in enumerate(args['bank']):
                bw = np.array(Fband)
                filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
                X_filtered = np.zeros_like(dataset.data)
                for i, trial in enumerate(dataset.data):
                    # filtering
                    trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
                    trail_filter = trail_filter.reshape(1, 22, 751)
                    X_filtered[i, :, :, :] = trail_filter
                data_filterbank[:, :, num*dataset.data.shape[2]: (num+1)*dataset.data.shape[2], :] = X_filtered
            dataset.data = data_filterbank
        else:
            dataset = dataset
    elif 'bcicompet2b_config' in config_name:
        dataset = BCICompet2bIV(args)
        if args['filter_bank']:
            #### FBCNet####
            # data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1], len(args['bank']),
            #                             dataset.data.shape[2], dataset.data.shape[3]))
            #
            # for num, Fband in enumerate(args['bank']):
            #     bw = np.array(Fband)
            #     filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
            #     X_filtered = np.zeros_like(dataset.data)
            #     for i, trial in enumerate(dataset.data):
            #         # filtering
            #         trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            #         trail_filter = trail_filter.reshape(1, 3, 751)
            #         X_filtered[i, :, :, :] = trail_filter
            #     data_filterbank[:, :, num, :, :] = X_filtered

            #### IFNet####
            data_filterbank = np.zeros(
                (dataset.data.shape[0], dataset.data.shape[1], 2 * dataset.data.shape[2], dataset.data.shape[3]))

            for num, Fband in enumerate(args['bank']):
                bw = np.array(Fband)
                filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
                X_filtered = np.zeros_like(dataset.data)
                for i, trial in enumerate(dataset.data):
                    # filtering
                    trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
                    trail_filter = trail_filter.reshape(1, 3, 751)
                    X_filtered[i, :, :, :] = trail_filter
                data_filterbank[:, :, num * dataset.data.shape[2]: (num + 1) * dataset.data.shape[2], :] = X_filtered

            dataset.data = data_filterbank
        else:
            dataset = dataset

    elif 'KUMI_config' in config_name:
        dataset = OpenBMI(args)
        if args['filter_bank']:
            #### FBCNet####
            # data_filterbank = np.zeros((dataset.data.shape[0], dataset.data.shape[1], len(args['bank']),
            #                             dataset.data.shape[2], dataset.data.shape[3]))
            #
            # for num, Fband in enumerate(args['bank']):
            #     bw = np.array(Fband)
            #     filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
            #     X_filtered = np.zeros_like(dataset.data)
            #     for i, trial in enumerate(dataset.data):
            #         # filtering
            #         trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
            #         trail_filter = trail_filter.reshape(1, 22, 751)
            #         X_filtered[i, :, :, :] = trail_filter
            #     data_filterbank[:, :, num, :, :] = X_filtered

            #### IFNet####
            data_filterbank = np.zeros(
                (dataset.data.shape[0], dataset.data.shape[1], 2 * dataset.data.shape[2], dataset.data.shape[3]))

            for num, Fband in enumerate(args['bank']):
                bw = np.array(Fband)
                filter_coef = load_filterbank(bw, 250, order=4, max_freq=40, ftype='butter')
                X_filtered = np.zeros_like(dataset.data)
                for i, trial in enumerate(dataset.data):
                    # filtering
                    trail_filter = butter_fir_filter(np.squeeze(trial), filter_coef[0])
                    trail_filter = trail_filter.reshape(1, 20, 751)
                    X_filtered[i, :, :, :] = trail_filter
                data_filterbank[:, :, num * dataset.data.shape[2]: (num + 1) * dataset.data.shape[2], :] = X_filtered

            dataset.data = data_filterbank
        else:
            dataset = dataset

    else:
        raise Exception('get_dataset function Wrong dataset input!!!')

    return dataset

