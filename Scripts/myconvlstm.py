"""
This script is intended to govern theself.model construction, training,
testing, and prediction of data generated from SSTPreparer. Neural
network is developed with tensorflow.
"""
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, MaxPool3D
from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks importself.modelCheckpoint, EarlyStopping
import sstpreparer

class MyConvLSTM:

    def __init__(self, config_json: str=None, Processor: SSTPreparer=None):
        self.processor = Processor

        with open(config_json) as json_file:
            config = json.load(json_file)

        self.seq_len = config['seq_len']
        self.interval = config['interval']
        self.lead_time = config['lead_time']
        self.split_year = config['split_year']
        self.model_outpath = config['model_outpath']
        self.batch_size = config['batch_size']
        self.steps_per_epoch = config['steps_per_epoch']
        self.epochs = config['epochs']

        self.model = Sequential()
        self.history = None

    def generator(self, Processor: SSTPreparer=None, batch_size: int=None, train: bool=True):
        batch_features = np.zeros((batch_size,self.seq_len,self.processor.lat_len,self.processor.lon_len,self.processor.channel))
        batch_labels = np.zeros((batch_size,3))

        while True:
            for i in range(batch_size):
                if train:
                    target_date = np.random.choice(self.processor.target_dates[np.where(self.processor.target_dates.year < self.split_year)[0]])
                else:
                    target_date = np.random.choice(self.processor.target_dates[np.where(self.processor.target_dates.year >= self.split_year)[0]])
                target_idx = np.where(self.processor.target_dates==target_date)[0]
                start_date = target_date - pd.Timedelta(days = self.lead_time+(self.seq_len-1)*self.interval)
                if start_date.month==2 and start_date.day==29:
                    start_date -= pd.Timedelta(days=1)
                start_idx = np.where(self.processor.data_dates == start_date)[0]
                idx_list = range(start_idx[0],start_idx[0]+self.seq_len*self.interval,self.interval)
                batch_features[i] = self.processor.data[idx_list].values
                batch_labels[i] = self.processor.target[target_idx].values
            yield batch_features, batch_labels

    def get_model(self):
        self.model.add(ConvLSTM2D(filters=32,
                                      kernel_size=(5,5),
                                      input_shape=(self.seq_len,self.processor.lat_len,self.processor.lon_len,self.processor.channel),
                                      data_format='channels_last',
                                      padding='same',
                                      return_sequences=True))
        self.model.add(MaxPool3D(pool_size=(1,4,4),
                                     padding='valid',
                                     data_format='channels_last'))
        self.model.add(ConvLSTM2D(filters=16,
                                       kernel_size=(3,3),
                                       data_format='channels_last',
                                       padding='same',
                                       return_sequences=True))
        self.model.add(MaxPool3D(pool_size=(1,4,4),
                                     padding='valid',
                                     data_format='channels_last'))
        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(len(self.processor.target_levels)))

        loss = binary_crossentropy
        opt = Adadelta()
        mets = categorical_accuracy
        self.model.compile(loss=loss,optimizer=opt,metrics=[mets])

    def fit(self):
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(self.model_outpath, save_best_only=True, monitor='val_loss', mode='min')

        self.history = model.fit_generator(generator=self.generator(self.processor,self.batch_size,True),
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.epochs,
                                      callbacks=[earlyStopping, mcp_save],
                                      validation_data=self.generator(self.processor,self.validation_size,False),
                                      validation_steps=1,
                                      shuffle=False,
                                      initial_epoch=0)


if __name__ == '__main__':
    processor = SSTPreparer('Configs/sstpreparer_config.json',
                            download_raw_sst = False,
                            combine_raw_sst = False,
                            load_raw_t95 = False)
    constructor = MyConvLSTM('Configs/myconvlstm_config.json',
                             Procesor = processor)
    constructor.get_model()
    constructor.fit()
