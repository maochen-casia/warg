from .trainer import Trainer
from .evaluator import Evaluator
from .checkpoint import Checkpoint
from .logger import Logger
from .dataset import LocDataset, read_data
from torch.utils.data import DataLoader

def build_checkpoint_logger(config, need_logger=True):
    checkpoint = Checkpoint(config)
    logger = Logger(checkpoint.save_dir) if need_logger else None
    print('Checkpoint and logger built successfully.')
    return checkpoint, logger

def build_trainer_evaluator(config, model, data_loaders):

    trainer = Trainer(config.trainer,
                      model,
                      data_loaders['train'])
    
    val_evaluator = Evaluator(config.evaluator,
                              model,
                              data_loaders['val'])

    test_evaluators = {}
    for key in data_loaders.keys():
        if key != 'train' and key != 'val':
            test_evaluators[key] = Evaluator(config.evaluator,
                                        model,
                                        data_loaders[key])
    
    print('Trainer and evaluators built successfully.')

    return trainer, val_evaluator, test_evaluators

def build_data_loaders(config, seed):

    dataset_name = config.dataset
    batch_size = config.batch_size
    left_image_size = config.left_image_size
    sat_image_size = config.sat_image_size
    aug = config.aug
    max_train_init_offset = config.max_train_init_offset
    max_aug_offset = config.max_aug_offset
    max_aug_rotate = config.max_aug_rotate
    max_test_init_offset = config.max_test_init_offset

    data_dict = read_data(data_dir='./data/',
                          dataset_name=dataset_name, keys=config.data_keys)

    num_workers = 16
    data_loaders = {}
    for key in data_dict:
        cur_aug = aug if key=='train' else False
        cur_max_init_offset = max_train_init_offset if key=='train' else max_test_init_offset
        
        cur_dataset = LocDataset(data_dict[key],
                                 left_image_size,
                                 sat_image_size,
                                 aug=cur_aug,
                                 max_init_offset=cur_max_init_offset,
                                 max_aug_offset=max_aug_offset,
                                 max_aug_rotate=max_aug_rotate,
                                 seed=seed)
        cur_data_loader = DataLoader(cur_dataset, batch_size, shuffle=(key=='train'), num_workers=num_workers, 
                                    collate_fn=cur_dataset.collate_fn, pin_memory=True)
        data_loaders[key] = cur_data_loader
    
    print('Data loaders built successfully.')

    return data_loaders