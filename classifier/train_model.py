""" Adaptation from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/main.py) """

from classifier.tools.arg_parser import get_config
from config import data_root, project_root
import os

from classifier.tools.trainer import Trainer

from datetime import date, datetime
from classifier.tools.utils import set_seed

if __name__ == "__main__":
    config, unparsed = get_config()
    assert config.mode in ['basetrain', 'advtrain']
    set_seed(333)
    run_time = date.today().strftime("%y%m%d") + datetime.now().strftime("%H%M")
    save_dir = os.path.join(project_root, 'training', run_time)
    # store configuration
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        f.write("Parameters for " + config.mode + ":\n\n")
        for arg in vars(config):
            argname = arg
            contents = str(getattr(config, arg))
            f.write(argname + " = " + contents + "\n")

    trainer = Trainer(config, save_dir)
    trainer.train(config.mode)