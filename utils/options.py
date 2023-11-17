import argparse

def parse_opt(ROOT, known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10_000_000, help='total training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--batch_size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--sequence_length', type=int, default=5, help='')
    parser.add_argument('--prediction_length', type=int, default=1, help='')
    parser.add_argument('--offset', type=int, default=1, help='')
    parser.add_argument('--train_size', type=float, default=0.7, help='')
    parser.add_argument('--val_size', type=float, default=0.1, help='')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', default=ROOT / 'runs', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'Nadam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Ftrl'], default='Adam', help='optimizer')
    parser.add_argument('--scheduler', type=str, choices=['StepLR'], default='StepLR', help='optimizer')
    parser.add_argument('--loss', type=str, choices=['MSE'], default='MSE', help='losses')
    parser.add_argument('--seed', type=int, default=941, help='Global training seed')
    parser.add_argument('--normalization', type=str, default=None, choices=[None, 'minmax', 'standard', 'robust'], help='')
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--low_memory', action='store_true', help='Ultilize disk')
    parser.add_argument('--models', 
                        action='append', 
                        choices=['VanillaLSTM', 'VanillaRNN'], 
                        required=True)

    return parser.parse_known_args()[0] if known else parser.parse_args()