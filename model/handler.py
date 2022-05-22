from ts.torch_handler.base_handler import BaseHandler
import torch

class CustHandler(BaseHandler):

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        print(data)
        print(data.shape)
        return torch.as_tensor(data, device=self.device)