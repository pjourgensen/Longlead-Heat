
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, MaxPool3D
from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks importself.modelCheckpoint, EarlyStopping

class MyConvLSTM:
    """
    A class used for the initialization, construction, training, and prediction for a ConvLSTM model.
    ...

    Attributes
    ----------
    processor: SSTPreparer
        an SSTPreparer object used to facilitate data management
    model: tensorflow.python.keras.engine.sequential.Sequential
        Keras based sequential model
    history: tensorflow.python.keras.callbacks.History
        Keras callback history of trained model

    Config Attributes
    -----------------
    seq_len: int
        number of days used for forecasting
    interval: int
        number of days between chosen days for forecasting
    lead_time: int
        number of days into the future to forecast
    split_yeat: int
        year for train/test split. Included in test set
    model_outpath: str
        file path to store final model
    batch_size: int
        batch_size to be used for training
    steps_per_epoch: int
        number of batches to include per epoch. Typically total data size / batch_size
    epochs: int
        number of epochs for model training
    validation_size: int
        number of test samples to use for validation each epoch

    Methods
    -------
    fit()
        runs training procedure on model
    """

    def __init__(self, config_json: str=None, Processor: SSTPreparer=None):
        """
        Parameters
        ----------
        config_json: str
            file path to a config file that contains "Config Attributes"
        Processor: SSTPreparer
            an SSTPreparer object used to facilitate data management
        """

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
        self.validation_size = config['validation_size']

        self.model = self._get_model()
        self.history = None

    def _generator(self, Processor: SSTPreparer=None, batch_size: int=None, train: bool=True):
        """Loads batch data and target into working memory for each training iteration"""

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

    def _get_model(self):
        """Initializes keras sequential neural network model"""

        self.model = Sequential()
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
        """Runs training procedure on model

        Model is saved to model_outpath and callback history is stored in history attribute. Early
        stopping and model checkpoint are used to ensure optimal model is saved over the course
        of training.

        Attributes
        ----------
        early_stopping: tensorflow.python.keras.callbacks.EarlyStopping
            stops model training when performance drop is signaled
        mcp_save: tensorflow.python.keras.callbacks.ModelCheckpoint
            periodically saves model with each improvement in performance
        model_outpath: str
            file path to store final model
        history: tensorflow.python.keras.callbacks.History
            keras callback history of model
        processor: SSTPreparer
            an SSTPreparer object used to facilitate data management
        batch_size: int
            batch_size to be used for training
        steps_per_epoch: int
            number of batches to include per epoch. Typically total data size / batch_size
        epochs: int
            number of epochs for model training
        validation_size: int
            number of test samples to use for validation each epoch
        """

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(self.model_outpath, save_best_only=True, monitor='val_loss', mode='min')

        self.history = model.fit_generator(generator=self._generator(self.processor,self.batch_size,True),
                                      steps_per_epoch=self.steps_per_epoch,
                                      epochs=self.epochs,
                                      callbacks=[early_stopping, mcp_save],
                                      validation_data=self._generator(self.processor,self.validation_size,False),
                                      validation_steps=1,
                                      shuffle=False,
                                      initial_epoch=0)
