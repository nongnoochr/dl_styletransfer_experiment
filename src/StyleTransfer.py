import datetime
import logging

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import models

from .internal import im_convert, load_image

class StyleTransfer:
    
    __final_loss = -1

    def __init__(self, content_image='', style_image=''):
        
        # --- Default weights
        # weights for each style layer 
        # weighting earlier layers more will result in *larger* style artifacts
        # notice we are excluding `conv4_2` our content representation
        self.layer_style_weights = {
            'conv1_1': 1.,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
        }

        # you may choose to leave these as is
        self.content_weight = 1  # alpha
        self.style_weight   = 1e6  # beta
        
        # --- Get the current device (gpu or cpu)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.__device = "cuda"
        else:
            self.__device = "cpu"
        
        
        
        # --- Map the layer index to the vgg19's layer name that matches the paper
        self.__layer_mapping = {
            '0':    'conv1_1',
            '5':    'conv2_1',
            '10':   'conv3_1',
            '19':   'conv4_1',
            '21':   'conv4_2',
            '28':   'conv5_1'
        }
        
        
        # --- Load image as a tensor data
        self.__content_image = content_image
        self.__content  = self.__loadImage(content_image)
        
        self.__style_image = style_image
        self.__style    = self.__loadImage(style_image, shape=self.__content.shape[-2:])

        # create a third "target" image and prep it for change
        # it is a good idea to start of with the target as a copy of our *content* image
        # then iteratively change its style
        self.__target = self.__content.clone().requires_grad_(True).to(self.__device)
        
        # Load features data of the the pre-trained network (vgg19)
        self.__vgg = self.__loadNetworkFeatures()
        
        # --- Get feature data
        
        # get content and style features only once before forming the target image
        self.__content_features = self.__get_features(self.__content)
        self.__style_features   = self.__get_features(self.__style)
        

        # calculate the gram matrices for each layer of our style representation
        self.__style_grams = {layer: self.__gram_matrix(self.__style_features[layer]) for layer in self.__style_features}
        
        
        
    def showInputImages(self):
        
        self.__showImageComparison(self.__content, self.__style,
                                   title_base='Content Image', title_target='Style Image')
        
    def showTargetImage(self):
        self.__showImageComparison(self.__content, self.__target,
                                   title_base='Content Image', title_target='Target Image')
        
    def export_target_image(self, output_path='output.png'):
        
        target_data = im_convert(self.__target)
        plt.imsave(output_path, target_data)
        
        print('*** Target image has been saved to "{}"'.format(output_path))
        
    def generate_target_image(self, steps = 5000, lr=0.003):
        
        # steps = 2000  # decide how many iterations to update your image (5000)
        
        print('**************************')
        print('***** Start generating the Target image *****')
        print('Learning Rate: {}'.format(lr))
        print('Number of learning iterations: {}'.format(steps))
        print('**************************')
        
        # for displaying the target image, intermittently
        show_every = 400
        
        # iteration hyperparameters
        optimizer = optim.Adam([self.__target], lr=lr)

        for ii in range(1, steps+1):
            
            ## TODO: get the features from your target image    
            ## Then calculate the content loss
            target_features = self.__get_features(self.__target)

            content_loss = torch.mean((target_features['conv4_2'] - self.__content_features['conv4_2'])**2)
            
            # the style loss
            # initialize the style loss to 0
            style_loss = 0
            # iterate through each style layer and add to the style loss
            for layer in self.layer_style_weights:
                # get the "target" style representation for the layer
                target_feature = target_features[layer]
                _, d, h, w = target_feature.shape
                
                ## Calculate the target gram matrix
                target_gram = self.__gram_matrix(target_feature)        

                ## get the "style" style representation
                style_gram = self.__style_grams[layer]
                
                ## Calculate the style loss for one layer, weighted appropriately
                layer_weight = self.layer_style_weights[layer]
                layer_style_loss = layer_weight * torch.mean((target_gram - style_gram)**2 )
                
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)
                
                
            ## calculate the *total* loss

            total_loss = (self.content_weight * content_loss) + (self.style_weight * style_loss)
            
            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            self.__final_loss = total_loss.item()
            
            # display intermediate images and print the loss
            if  ii % show_every == 0:
                print('*** [{}] Steps#{} - Total loss: {}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ii, self.__final_loss))
                plt.imshow(im_convert(self.__target))
                plt.show()
        

    # ----------------
    # Read-only attributes
    # ----------------
    @property
    def vgg(self):
        return self.__vgg
    
    @property
    def device(self):
        return self.__device
    
    @property
    def content_image(self):
        return self.__content_image
    
    @property
    def layer_mapping(self):
        return self.__layer_mapping
    
    @property
    def style_image(self):
        return self.__style_image
    
    @property
    def final_loss(self):
        return self.__final_loss
    
    @property
    def data(self):
        return {
            'tensor_content':   self.__content,
            'tensor_style':     self.__style,
            'tensor_target':    self.__target,
            
            'features_content': self.__content_features,
            'features_style':   self.__style_features,
            
            'style_grams':      self.__style_grams
        }
        
    @property
    def weights(self):
        
        return {
            'content_weight':       self.content_weight,
            'style_weight':         self.style_weight,
            
            'layer_style_weights':  self.layer_style_weights,
        }

        
    # ----------------
    # Private methods
    # ----------------
    def __loadNetworkFeatures(self):
        '''Load the pre-trained network vgg19 and '''
        
        # get the "features" portion of VGG19 (we will not need the "classifier" portion)
        vgg = models.vgg19(pretrained=True).features

        # freeze all VGG parameters since we're only optimizing the target image
        for param in vgg.parameters():
            param.requires_grad_(False)
            
        vgg.to(self.__device)
            
        return vgg
    
    def __loadImage(self, image_path, shape=None):
        
        
        try:
            tensor_image = load_image(image_path, shape=shape).to(self.__device)
        except Exception as error:
            logging.exception('Cannot load the specified image: "{}"'.format(image_path))
            raise ValueError

        
        return tensor_image
    
    def __showImageComparison(self, tensor_image_base, tensor_image_target, title_base='', title_target=''):
        
        # display the images
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        # content and style ims side-by-side
        ax1.imshow(im_convert(tensor_image_base))
        ax1.title.set_text(title_base)
        ax1.set_axis_off()
        
        ax2.imshow(im_convert(tensor_image_target))
        ax2.title.set_text(title_target)
        ax2.set_axis_off()
        
    def __get_features(self, image):
        """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """
        
            
        ## -- do not need to change the code below this line -- ##
        features = {}
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.__layer_mapping:
                features[self.__layer_mapping[name]] = x
                
        return features
    
    def __gram_matrix(self, tensor):
        """ Calculate the Gram Matrix of a given tensor 
            Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
        """
        
        ## get the batch_size, depth, height, and width of the Tensor
        ## reshape it, so we're multiplying the features for each channel
        ## calculate the gram matrix
        
        tensor = tensor.squeeze()
        d, h, w = tensor.size()
        
        tensor_vectorized = tensor.view(-1, h*w)
        gram = torch.mm(tensor_vectorized, tensor_vectorized.transpose(0, 1))
        
        return gram 