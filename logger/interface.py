from abc import ABCMeta, abstractmethod

class ExperimentLogger(metaclass=ABCMeta):
    @abstractmethod
    def log_metric(self, name, value, step=None):
        pass

    @abstractmethod
    def log_text(self, name, text):
        pass

    @abstractmethod
    def log_parameter(self, dictionary):
        pass

    @abstractmethod    
    def log_image(self, name, image):
        pass

    @abstractmethod
    def save_model(self, name, state_dict):
        pass

    @abstractmethod
    def finish(self):
        pass