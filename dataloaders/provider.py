from torch.utils.data import DataLoader

from dataloaders.loaders import GeneralDataset
    
def provider(args, flag):
    print("Loading dataset...")
    
    dataset = GeneralDataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
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