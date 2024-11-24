import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class Nasadataset(Dataset):
    def __init__(self, mode='train', dataset=None, rul_result=None, window_size=30, max_rul=150,dropped_cols=None):
        # Load dataset using pandas
        self.dropped = None
        col_names = ["unit_no","time","os1","os2","os3"] + [f'sm{i}' for i in range(1,22)] 
        df = pd.read_csv(dataset,sep=" ",header=None,names=col_names,index_col=False)
        if mode == "train":
            to_drop = []
            for col in df.columns:
                if(abs(df[col].std()) < 0.02):
                    to_drop += [col]
            df.drop(columns=to_drop,inplace=True)
            self.dropped = to_drop
            mask = np.triu(np.ones(df.shape).astype(bool))
            df_corr = df.corr()
            mask = np.tril(np.ones(df_corr.shape),k = -1).astype(bool)
            df_corr = df_corr.where(mask)
            mask = df_corr.where( abs(df_corr) > 0.95 ).isna()
            high_corr = []
            for col in df_corr.columns:
                for row in df_corr.index:
                    if abs(df_corr.loc[col , row]) > 0.95 :
                        high_corr.append((row , col))
            to_drop = to_drop = [pair[0] for pair in high_corr]
            df.drop(columns = to_drop , inplace = True)
            self.dropped += to_drop
        # print(df.columns)
        if mode == "test":
            df.drop(columns=dropped_cols,inplace=True)
        self.window_size = window_size
        self.max_rul = max_rul
        self.mode = mode
        # print(df.shape)
        # Calculate RUL for training
        if mode == 'train':
            # Reverse groupby to compute RUL per engine
            df['RUL'] = df.groupby('unit_no')['time'].transform("max") - df['time']
            df['RUL'] = df['RUL'].clip(upper=max_rul)

        # For test mode, load RUL results
        elif mode == 'test' and rul_result is not None:
            rul_df = pd.read_csv(rul_result, header=None, names=['RUL'])
            rul_df['RUL'] = rul_df['RUL']

        # Prepare sliding windows
        X, y = [], []
        self.x, self.y = [],[]
        if mode == "train":
            for unit_no, group in df.groupby('unit_no'):
                group = group.reset_index(drop=True)
                
                # Create sliding windows using pandas
                for start in range(len(group) - self.window_size + 1):
                    window = group.iloc[start:start + self.window_size]
                    y.append(window['RUL'].iloc[-1])  # Use RUL of the last row in the window
                    # print(window.columns)
                    # window = window.drop(columns = "RUL",inplace=True)
                    # print(window.columns)
                    X.append(window.iloc[:, 2:-1].to_numpy())  # Exclude unit_no and time
        if self.mode == 'test':
            for unit_no, group in df.groupby('unit_no'):
                group = group.reset_index(drop=True)

                # Check if the sequence is shorter than the window size
                if len(group) < self.window_size:
                    # Perform interpolation
                    interpolated = pd.DataFrame(
                        index=np.arange(self.window_size),
                        columns=group.columns
                    )
                    for col in group.columns:
                        x_old = np.arange(len(group))
                        y_old = group[col].to_numpy()
                        interpolated[col] = np.interp(
                            np.linspace(0, len(group) - 1, self.window_size),
                            x_old,
                            y_old
                        )
                    interpolated['unit_no'] = unit_no
                    interpolated['time'] = np.linspace(
                        group['time'].iloc[0],
                        group['time'].iloc[-1],
                        self.window_size
                    )
                    group = interpolated
                
                # Extract the last `window_size` rows for the sliding window
                window = group.iloc[-self.window_size:, 2:].to_numpy()  # Exclude unit_no and time
                # print(window.shape)
                self.x.append(window)

                # Retrieve the RUL value from the rul_result
                rul = rul_df.loc[unit_no - 1, 'RUL'] if rul_result is not None else 0
                self.y.append(min(rul, self.max_rul))  # Clip RUL to max_rul
        if mode == "train":
            self.x = np.array(X, dtype=np.float32)
            self.y = np.array(y, dtype=np.float32) / max_rul  # Normalize RUL
        else:
            self.x = np.array(self.x, dtype=np.float32)
            self.y = np.array(self.y, dtype=np.float32) / max_rul  # Normalize RUL
        self.num_cols = df.shape[1]-3
        # print(df.columns)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x_tensor = torch.from_numpy(self.x[index]).to(torch.float32)
        y_tensor = torch.Tensor([self.y[index]]).to(torch.float32)
        return x_tensor, y_tensor
