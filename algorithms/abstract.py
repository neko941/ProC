class AbstractAlgorithm:
    def __init__(self, configs, save_dir='.'):
        self.history = None
        self.time_used = '0s'
        self.model = None
        self.configs = yaml_load(configs)

        self.dir_log          = 'logs'
        self.dir_plot         = 'plots'
        self.dir_weight       = 'weights'
        self.mkdirs(path=save_dir)

        self.best_weight = None

    def mkdirs(self, path):
        path = Path(path)
        self.path_log          = path / self.dir_log
        self.path_plot         = path / self.dir_plot
        self.path_weight       = path / self.dir_weight

        for p in [self.path_log, self.path_plot, self.path_weight]: 
            p.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build(self, *inputs):
        raise NotImplementedError 

    @abstractmethod
    def preprocessing(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def fit(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def save(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def load(self, *inputs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *inputs):
        raise NotImplementedError

    def plot(self, save_dir, y, yhat, dataset):
        try:
            save_plot(filename=os.path.join(self.path_plot, f'{self.__class__.__name__}-{dataset}.png'),
                      data=[{'data': [range(len(y)), y],
                             'color': 'green',
                             'label': 'y'},
                            {'data': [range(len(yhat)), yhat],
                             'color': 'red',
                             'label': 'yhat'}],
                      xlabel='Sample',
                      ylabel='Value')
        except: pass

    def score(self, 
              y, 
              yhat, 
              scaler=None,
              r=-1):
        return score(y=y, 
                     yhat=yhat, 
                     scaler=scaler,
                     r=r)