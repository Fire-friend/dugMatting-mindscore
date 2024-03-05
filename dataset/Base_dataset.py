from typing import Tuple


class Base_Dataset:
    """


    """

    def __init__(self,
                 args,
                 mode='train'):
        assert mode in ['train', 'val', 'show'], 'dataset mode is error'
        self.mode = mode

    def get_train_data(self, item: int):

        pass

    def get_val_data(self, item: int):
        pass

    def get_show_data(self, item: int):
        pass

    def __getitem__(self, item):
        if self.mode == 'train':
            return self.get_train_data(item)
        elif self.mode == 'val':
            return self.get_val_data(item)
        elif self.mode == 'show':
            return self.get_show_data(item)

    def __len__(self):
        pass
