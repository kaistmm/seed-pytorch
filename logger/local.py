import os
import time

import torch
import shutil
from threading import Thread

from .interface import ExperimentLogger

class LocalLogger(ExperimentLogger):
    """Save experiment logs to local storage.
    Note that local files are synchronized every 10 seconds
    (IO buffer is flushed every 10 sec).
    """
    def __init__(self, path, name, project, tags, description=None, backup=None):
        # set path
        path = os.path.join("/", path, project, "/".join(tags), name)
        self.paths = {
            'metric': f'{path}/metric',
            'text'  : f'{path}/text',
            'image' : f'{path}/image',
            'model' : f'{path}/model',
            'backup': f'{path}/backup',
        }
        
        # make directory
        if os.path.exists(path):
            shutil.rmtree(path)
        for dir in self.paths.values():
            os.makedirs(dir)

        # save description
        if description is not None:
            self.log_text('description', description)

        # script backup
        if backup is not None:
            shutil.copytree(backup, f"{self.paths['backup']}/scripts")
            
            for root, dirs, _ in os.walk(self.paths['backup']):
                for dir in dirs:
                    if dir == '__pycache__':
                        shutil.rmtree(f'{root}/{dir}')

        # synchronize thread
        self._metrics = {}
        thread_synchronize = Thread(target=self._sync)
        thread_synchronize.setDaemon(True)
        thread_synchronize.start()

    def _sync(self):
        """Flush files
        """
        while(True):
            for txt in self._metrics.values():
                txt.flush()
            time.sleep(10)

    def log_text(self, name, text):
        path = f'{self.paths["text"]}/{name}.txt'
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        mode = 'a' if os.path.exists(path) else 'w'
        file = open(path, mode, encoding='utf-8')
        file.write(text)
        file.close()

    def log_parameter(self, dictionary):
        for k, v in dictionary.items():
            self.log_text('parameters', f'{k}: {v}\n')

    def log_metric(self, name, value, step=None):
        if name not in self._metrics.keys():
            path = f'{self.paths["metric"]}/{name}.txt'
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
            self._metrics[name] = open(path, 'w', encoding='utf-8').close()
            self._metrics[name] = open(path, 'a', encoding='utf-8')

        if step is None:
            self._metrics[name].write(f'{name}: {value}\n')
        else:
            self._metrics[name].write(f'[{step}] {name}: {value}\n')
    
    def log_image(self, name, image):
        path = f'{self.paths["image"]}/{name}.png'
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        image.save(path, 'PNG')
    
    def save_model(self, name, state_dict):
        path = f'{self.paths["model"]}/{name}.pt'
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(state_dict, path)
    
    def finish(self):
        pass