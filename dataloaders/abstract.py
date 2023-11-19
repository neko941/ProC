import os
from utils.visuals import progress_bar
from pathlib import Path
import polars as pl
import numpy as np
from utils.npy import NpyFileAppend
from utils.general import list_uniqifier
import math
from datetime import timedelta
import json

class AbstractLoader():
    def __init__(self, low_memory, normalization):
        self.low_memory = low_memory
        self.normalization = normalization

        self.extensions = ('.csv')
        self.delimiter = ","
        self.has_header = True
        self.infer_schema_length:int|None = 10_000
        self.try_parse_dates:bool = True
        self.file_name_as_feature:str = None
    
    def _get_paths(self):
        if not isinstance(self.path, list): 
            data_paths = [self.path]
        else:
            data_paths = self.path
        file_paths = []
        with progress_bar() as progress:
            for path in progress.track(data_paths, description=' Getting files'):
                if os.path.isdir(path): [file_paths.append(Path(root, file)) for root, _, files in os.walk(path) for file in files if file.endswith(self.extensions)]
                elif path.endswith(self.extensions) and os.path.exists(path): file_paths.append(path)
        assert len(file_paths) > 0, 'No csv file(s)'

        return file_paths

    def _read_csvs(self, csvs:list) -> pl.DataFrame | None:
        if not isinstance(csvs, list): csvs = [csvs] 
        csvs = [os.path.abspath(csv) for csv in csvs]  

        all_data = []
        with progress_bar() as progress:
            for csv in progress.track(csvs, description='  Reading data'):
                data = pl.read_csv(source=csv, 
                                   separator=self.delimiter,  
                                   has_header=self.has_header, 
                                   try_parse_dates=self.try_parse_dates, 
                                   low_memory=self.low_memory, 
                                   infer_schema_length=self.infer_schema_length)
                if self.file_name_as_feature is not None: 
                    d = '.'.join(os.path.basename(os.path.abspath(csv)).split('.')[:-1])
                    if d.isdigit():
                        data = data.with_columns(pl.lit(int(d)).alias(self.file_name_as_feature))
                    else:
                        # TODO: handle the case where values are categorical not integers
                        print('To be implemented!!!!!!!!!!!!!')
                        exit()
                all_data.append(data)
        return pl.concat(all_data)

    def _reduce_memory(self, 
                       df:pl.DataFrame, 
                       verbose:bool=False):
        before = round(df.estimated_size('gb'), 4)
        Numeric_Int_types = [pl.Int8,pl.Int16,pl.Int32,pl.Int64]
        Numeric_Float_types = [pl.Float32,pl.Float64]    
        for col in df.columns:
            col_type = df[col].dtype
            c_min = df[col].min()
            c_max = df[col].max()
            if col_type in Numeric_Int_types:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df = df.with_columns(df[col].cast(pl.Int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df = df.with_columns(df[col].cast(pl.Int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df = df.with_columns(df[col].cast(pl.Int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df = df.with_columns(df[col].cast(pl.Int64))
            elif col_type in Numeric_Float_types:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df = df.with_columns(df[col].cast(pl.Float32))
                else:
                    pass
            elif col_type == pl.Utf8:
                df = df.with_columns(df[col].cast(pl.Categorical))
            else:
                pass
        df = df.shrink_to_fit()
        if verbose: print(f"Memory usage: {before} GB => {round(df.estimated_size('gb'), 4)} GB")
        return df

    def _split(self, 
                   df, 
                   dateFeature, 
                   granularity, 
                   trainFeatures, 
                   targetFeatures,
                   lagWindow, 
                   lag,
                   ahead,
                   splitRatio,
                   segmentFeatureEle=None):
        trainFeaturesAdded = []
        df = self.FillDate(df=df,
                          dateColumn=dateFeature,
                          granularity=granularity)
        
        for key in segmentFeatureEle.keys():
            # print(segmentFeatureEle[key] is not None)
        # if segmentFeature is not None and segmentFeatureEle is not None: 
            df = df.with_columns(pl.col(key).fill_null(pl.lit(segmentFeatureEle[key])))
        for f in list_uniqifier([*trainFeatures, *targetFeatures]):
            temp = []
            for l in range(1, lagWindow):
                df = df.with_columns(pl.col(f).shift(periods=l).alias(f'{f}_lag_{l}'))
                temp.append(f'{f}_lag_{l}')
            trainFeaturesAdded.append(temp)

        # Get used columns
        trainFeaturesAdded = np.array(trainFeaturesAdded)
        trainColumns = np.transpose(trainFeaturesAdded[:, -lag:]).flatten()
        targetColumns = [*targetFeatures,
                         *[elem_b for elem_b in np.transpose(trainFeaturesAdded[:, :(ahead-1)]).flatten() if any(elem_a in elem_b for elem_a in targetFeatures)]]
        trainColumns = [item for sublist in [trainColumns[i:i + len([*trainFeatures, *targetFeatures])] for i in range(0, len(trainColumns), len([*trainFeatures, *targetFeatures]))][::-1] for item in sublist]
        targetColumns = [item for sublist in [targetColumns[i:i + len(targetFeatures)] for i in range(0, len(targetColumns), len(targetFeatures))][::-1] for item in sublist]
        
        # Drop nulls
        # print(trainColumns)
        # print(targetColumns)
        # exit()
        df = df[list_uniqifier([dateFeature, *trainColumns, *targetColumns])].drop_nulls()

        # Get train, val, test idx for splitting
        num_samples = len(df)
        train_idx = math.ceil(num_samples*splitRatio[0])
        val_idx = int(num_samples*(splitRatio[0] + splitRatio[1]))

        x = df[trainColumns].to_numpy()
        x = x.reshape(x.shape[0], int(x.shape[1]/len(trainFeaturesAdded)), len(trainFeaturesAdded))
        y = df[targetColumns].to_numpy()
        if y.shape[0]!=1 or y.shape[1]==1: y = np.squeeze(y)
        if y.shape == (): y = np.reshape(y, (1,))

        return x, y, train_idx, val_idx, trainFeaturesAdded, [df[:train_idx][dateFeature].max(),
                                                              df[train_idx:val_idx][dateFeature].max(),
                                                              df[train_idx:val_idx][dateFeature].min(),
                                                              df[val_idx:][dateFeature].min()]
    
    def FillDate(self, 
                 df:pl.DataFrame,  
                 dateColumn:str, 
                 low=None, 
                 high=None, 
                 granularity:int=5,
                 storeValues:bool=False): 
        if not low: low=df[dateColumn].min()
        if not high: high=df[dateColumn].max()

        df = df.join(other=pl.date_range(low=low,
                           high=high,
                           interval=timedelta(minutes=granularity),
                           closed='both',
                           name=dateColumn,
                           eager=True).to_frame(), 
                     on=dateColumn, 
                     how='outer')
        if storeValues: self.df = df
        else: return df

    def SplittingData(self, 
                      df,
                      splitRatio, 
                      lag, 
                      ahead, 
                      offset,
                      trainFeatures:list,
                      targetFeatures:list,
                      segmentFeature:str,
                      dateFeature:str, 
                      lowMemory:bool=False, 
                      saveDir:Path=None, 
                      storeValues:bool=True,
                      granularity:int=1440,
                      saveRaw:bool=False,
                      subname:str=''):

        lag_window = lag + offset

        xtrain = []
        ytrain = []
        xval = []
        yval = []
        xtest = []
        ytest = []

        if saveRaw:
            trainraw = []
            valraw = []
            testraw = []
            header_written = False

        save_file = {
            'invalid': {
                'path': saveDir / 'invalid.txt',
                'writer': open(saveDir / 'invalid.txt', 'a')
                }
        }

        if lowMemory:
            save_file['xtrain'] = {
                'path': saveDir / f'xtrain{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'xtrain{subname}.npy', delete_if_exists=True)
            }
            save_file['ytrain'] = {
                'path': saveDir / f'ytrain{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'ytrain{subname}.npy', delete_if_exists=True)
            }
            save_file['xval'] = {
                'path': saveDir / f'xval{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'xval{subname}.npy', delete_if_exists=True)
            }
            save_file['yval'] = {    
                'path': saveDir / f'yval{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'yval{subname}.npy', delete_if_exists=True)
            }
            save_file['xtest'] = {
                'path': saveDir / f'xtest{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'xtest{subname}.npy', delete_if_exists=True)
            }
            save_file['ytest'] = {
                'path': saveDir / f'ytest{subname}.npy',
                'writer': NpyFileAppend(filename=saveDir / f'ytest{subname}.npy', delete_if_exists=True)
            }
            if saveRaw:
                save_file['trainraw'] = {
                    'path': saveDir / 'trainraw.csv',
                    'writer': None
                }
                save_file['valraw'] = {
                    'path': saveDir / 'valraw.csv',
                    'writer': None
                }
                save_file['testraw'] = {
                    'path': saveDir / 'testraw.csv',
                    'writer': None
                }

        if segmentFeature is not None:
            # print(df)
            with progress_bar() as progress:
                for ele in progress.track(df.unique(subset=self.keys)[self.keys].to_dicts(), description='Splitting data'):
                    c = df.clone()
                    for key in self.keys:
                        c = c.filter((pl.col(key) == ele[key]))
                    d = c.sort(dateFeature)
                    # print(d)
                    # print(ele)
                    # exit()

                    x, y, train_idx, val_idx, trainFeaturesAdded, thething = self._split(df=d, 
                                                                                   dateFeature=dateFeature, 
                                                                                   granularity=granularity,
                                                                                   trainFeatures=trainFeatures, 
                                                                                   targetFeatures=targetFeatures,
                                                                                   lagWindow=lag_window, 
                                                                                   lag=lag,
                                                                                   ahead=ahead,
                                                                                   splitRatio=splitRatio,
                                                                                #    segmentFeature=segmentFeature,
                                                                                   segmentFeatureEle=ele)


                    if any([x[:train_idx].size == 0, 
                            y[:train_idx].size == 0, 
                            (x[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                            (y[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                            (x[val_idx:].size == 0) & (splitRatio[2] != 0), 
                            (y[val_idx:].size == 0) & (splitRatio[2] != 0)]):
                        save_file['invalid']['writer'].write(f'{ele}\n')

                        continue
                    if saveRaw: 
                        train = c.filter(pl.col(dateFeature) <= thething[0])
                        val = c.filter((pl.col(dateFeature) <= thething[1]) & 
                                       (pl.col(dateFeature) >= thething[2]))
                        test = c.filter(pl.col(dateFeature) >= thething[3])

                        if lowMemory:
                            train.to_pandas().to_csv(path_or_buf=save_file['trainraw']['path'], mode='a', index=False, header=not header_written)
                            val.to_pandas().to_csv(path_or_buf=save_file['valraw']['path'], mode='a', index=False, header=not header_written)
                            test.to_pandas().to_csv(path_or_buf=save_file['testraw']['path'], mode='a', index=False, header=not header_written)
                            if not header_written:
                                header_written = True
                        else:
                            trainraw.append(train)
                            valraw.append(val)
                            testraw.append(test)
                    if lowMemory:
                        if ahead > 1: fortran_order=False 
                        else: fortran_order=None 
                        save_file['xtrain']['writer'].append(x[:train_idx], fortran_order=fortran_order) 
                        save_file['ytrain']['writer'].append(y[:train_idx], fortran_order=fortran_order) 
                        save_file['xval']['writer'].append(x[train_idx:val_idx], fortran_order=fortran_order) 
                        save_file['yval']['writer'].append(y[train_idx:val_idx], fortran_order=fortran_order) 
                        save_file['xtest']['writer'].append(x[val_idx:], fortran_order=fortran_order) 
                        save_file['ytest']['writer'].append(y[val_idx:], fortran_order=fortran_order) 
                    else:
                        xtrain.extend(x[:train_idx]) 
                        ytrain.extend(y[:train_idx]) 
                        xval.extend(x[train_idx:val_idx]) 
                        yval.extend(y[train_idx:val_idx]) 
                        xtest.extend(x[val_idx:]) 
                        ytest.extend(y[val_idx:]) 
        else:
            d=df.sort(dateFeature)
            x, y, train_idx, val_idx, trainFeaturesAdded, _ = self._split(df=d, 
                                                                           dateFeature=dateFeature, 
                                                                           granularity=granularity,
                                                                           trainFeatures=trainFeatures, 
                                                                           targetFeatures=targetFeatures,
                                                                           lagWindow=lag_window, 
                                                                           lag=lag,
                                                                           ahead=ahead,
                                                                           splitRatio=splitRatio,
                                                                        #    segmentFeature=None,
                                                                           segmentFeatureEle=None)

            if any([x[:train_idx].size == 0, 
                    y[:train_idx].size == 0, 
                    (x[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                    (y[train_idx:val_idx].size == 0) & (splitRatio[1] != 0), 
                    (x[val_idx:].size == 0) & (splitRatio[2] != 0), 
                    (y[val_idx:].size == 0) & (splitRatio[2] != 0)]):
                print('No instance after splitting')
                exit()
            if saveRaw: 
                if lowMemory:
                    d[:train_idx].to_pandas().to_csv(path_or_buf=save_file['trainraw']['path'], mode='a', index=False, header=not header_written)
                    d[train_idx:val_idx].to_pandas().to_csv(path_or_buf=save_file['valraw']['path'], mode='a', index=False, header=not header_written)
                    d[val_idx:].to_pandas().to_csv(path_or_buf=save_file['testraw']['path'], mode='a', index=False, header=not header_written)
                    if not header_written:
                        header_written = True
                else:
                    trainraw = d[:train_idx]
                    valraw = d[train_idx:val_idx]
                    testraw = d[val_idx:]
            if lowMemory:
                np.save(file=save_file['xtrain']['path'], arr=x[:train_idx])
                np.save(file=save_file['ytrain']['path'], arr=y[:train_idx])
                np.save(file=save_file['xval']['path'], arr=x[train_idx:val_idx])
                np.save(file=save_file['yval']['path'], arr=y[train_idx:val_idx])
                np.save(file=save_file['xtest']['path'], arr=x[val_idx:])
                np.save(file=save_file['ytest']['path'], arr=y[val_idx:])
            else:
                xtrain = x[:train_idx]
                ytrain = y[:train_idx]
                xval = x[train_idx:val_idx]
                yval = y[train_idx:val_idx]
                xtest = x[val_idx:]
                ytest = y[val_idx:]
        
        for key in save_file.keys():
            if save_file[key]['writer'] is not None: save_file[key]['writer'].close()
        if lowMemory:
            xtrain = np.load(file=save_file['xtrain']['path'], mmap_mode='r+')
            ytrain = np.load(file=save_file['ytrain']['path'], mmap_mode='r+')
            xval = np.load(file=save_file['xval']['path'], mmap_mode='r')
            yval = np.load(file=save_file['yval']['path'], mmap_mode='r')
            xtest = np.load(file=save_file['xtest']['path'], mmap_mode='r')
            ytest = np.load(file=save_file['ytest']['path'], mmap_mode='r')

        if lowMemory and saveRaw:
            trainraw = pl.read_csv(source=save_file['trainraw']['path'], 
                                   has_header=self.has_header, 
                                   try_parse_dates=self.try_parse_dates, 
                                   low_memory=self.low_memory, 
                                   infer_schema_length=self.infer_schema_length)
            valraw = pl.read_csv(source=save_file['valraw']['path'], 
                                   has_header=self.has_header, 
                                   try_parse_dates=self.try_parse_dates, 
                                   low_memory=self.low_memory, 
                                   infer_schema_length=self.infer_schema_length)
            testraw = pl.read_csv(source=save_file['testraw']['path'], 
                                   has_header=self.has_header, 
                                   try_parse_dates=self.try_parse_dates, 
                                   low_memory=self.low_memory, 
                                   infer_schema_length=self.infer_schema_length)
        if not lowMemory and saveRaw and segmentFeature is not None:
            trainraw = pl.concat(trainraw)
            valraw = pl.concat(valraw)
            testraw = pl.concat(testraw)

        xtrain = np.array(xtrain)
        ytrain = np.array(ytrain)
        xval = np.array(xval)
        yval = np.array(yval)
        xtest = np.array(xtest)
        ytest = np.array(ytest)

        if storeValues:     
            self.xtrain = xtrain
            self.ytrain = ytrain
            self.xval = xval
            self.yval = yval
            self.xtest = xtest
            self.ytest = ytest

            if saveRaw:
                self.trainraw = trainraw
                self.valraw = valraw
                self.testraw = testraw
        else:
            return [xtrain, ytrain, xval, yval, xtest, ytest]
    
    def preprocess(self):
        pass

    def __call__(self, save_dir, split_ratio, lag, ahead, offset):
        save_dir = Path(save_dir) / 'values'
        save_dir.mkdir(parents=True, exist_ok=True) 

        self.preprocess()
        path = self._get_paths()
        df = self._read_csvs(csvs=path)
        df = self._reduce_memory(df=df, verbose=True)

        used_cols = [self.date_feature, *self.keys, *self.train_features, *self.target_features]
        used_cols = [item for item in used_cols if item is not None]
        df = df[list_uniqifier(used_cols)]

        if self.keys:
            if self.date_feature: df = df.sort(by=[*self.keys, self.date_feature])
            else: df = df.sort(by=self.keys)
        
        self.SplittingData(df,
                                   splitRatio=split_ratio,
                                   lag=lag, 
                                   ahead=ahead, 
                                   offset=offset,
                                   trainFeatures=self.train_features,
                                   targetFeatures=self.target_features,
                                   segmentFeature=self.keys,
                                   dateFeature=self.date_feature, 
                                   lowMemory=self.low_memory, 
                                   saveDir=save_dir, 
                                   storeValues=True,
                                   granularity=self.granularity,
                                   saveRaw=self.normalization)  
        
        """ Normalization """
        # if self.normalization: 
        if self.normalization is not None: 
            print('Normalizing Data')
            scaler = {}
            for col in self.trainraw.columns:
                if col == self.date_feature: continue
                scaler[col] = {
                    'min': self.trainraw[col].min(),
                    'max': self.trainraw[col].max(),
                    'mean': self.trainraw[col].mean(),
                    'median': self.trainraw[col].median(),
                    'std': self.trainraw[col].std(),
                    'variance': self.trainraw[col].var(),
                    'quantile_25': self.trainraw[col].quantile(quantile=0.25, interpolation='nearest'),
                    'quantile_50': self.trainraw[col].quantile(quantile=0.5, interpolation='nearest'),
                    'quantile_75': self.trainraw[col].quantile(quantile=0.75, interpolation='nearest'),
                    'iqr': self.trainraw[col].quantile(quantile=0.75, interpolation='nearest') - self.trainraw[col].quantile(quantile=0.25, interpolation='nearest'),
                    'method': self.normalization
                }
                # self.df = self.df.with_columns((pl.col(col) - self.scaler[col]['min']) / (self.scaler[col]['max'] - self.scaler[col]['min']))
                if self.normalization == 'minmax':
                    df = df.with_columns((pl.col(col) - scaler[col]['min']) / (scaler[col]['max'] - scaler[col]['min']))
                elif self.normalization == 'standard':
                    df = df.with_columns((pl.col(col) - scaler[col]['mean']) / scaler[col]['std'])
                elif self.normalization == 'robust':
                    df = df.with_columns((pl.col(col) - scaler[col]['median']) / (scaler[col]['iqr'] + 1e-12))
                    
            # self.scaler = self.scaler[self.targetFeatures[0]]
            
            with open(save_dir / 'scaler.json', "w") as json_file:
                # print('='*123)
                json.dump(scaler, json_file)

            self.SplittingData(df,
                                       splitRatio=split_ratio,
                                       lag=lag, 
                                       ahead=ahead, 
                                       offset=offset,
                                       trainFeatures=self.train_features,
                                       targetFeatures=self.target_features,
                                       segmentFeature=self.keys,
                                       dateFeature=self.date_feature, 
                                       lowMemory=self.low_memory, 
                                       saveDir=save_dir, 
                                       storeValues=True,
                                       granularity=self.granularity,
                                       saveRaw=False) 
 
        return self.xtrain, self.ytrain, self.xval, self.yval, self.xtest, self.ytest