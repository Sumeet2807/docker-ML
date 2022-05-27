from matplotlib.pyplot import axis
from ts.torch_handler.base_handler import BaseHandler
import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2

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


        image_url = data[0]['body']['url']

        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        print(np.array(img).shape)
        # print(data.shape)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        data = torch.ones((16,64))
        return torch.as_tensor(data, device=self.device)

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """
        
        data = torch.argmax(data,axis=1)
        print(data.tolist())
        return [data.tolist()]