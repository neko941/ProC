from torch.utils.data import DataLoader

from dataloaders.loaders import GeneralDataset

def provider(args, flag):
    print("Loading dataset...")

    dataset = GeneralDataset(
        root_path='./datasets/weather',
        data_path='weather.csv',
        # root_path=args.root_path,
        # data_path=args.data_path,
        flag=flag,
        seq_len=args.sequence_length,
        label_len=args.label_length,
        pred_len=args.prediction_length,
        date=args.date,
        target=args.target,
    )

    print(flag, len(dataset))
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    return train_loader