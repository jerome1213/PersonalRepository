class TrainConfig:
    def __init__(self):
        self.image_size = 64
        self.img_shape = (self.image_size, self.image_size, 3)
        self.batch_size = 8
        self.max_epochs = 100
        self.print_steps = 50
        self.save_epochs = 20
        self.train_dir = 'train/exp1'