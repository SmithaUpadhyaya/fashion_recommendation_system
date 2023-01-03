from tensorflow.keras.metrics import binary_crossentropy as Binary_Crossentropy
from tensorflow.keras.losses import Loss
import tensorflow as tf

#https://towardsdatascience.com/a-loss-function-suitable-for-class-imbalanced-data-focal-loss-af1702d75d75
class WeightLossBinaryCrossentropy(Loss):

    def __init__(self, param):
    
        super(WeightLossBinaryCrossentropy, self).__init__()        
        self.params = param           

        
    def get_config(self):
        
        config = super(WeightLossBinaryCrossentropy, self).get_config()
        
        config.update({
                        "params": self.params,                           
                        })
        
        return config

    @classmethod
    def from_config(cls, config):
        
        cls.params = config['params']            
        return
        
        
    def call(self, y_true, y_pred):
        
        y_true = tf.squeeze(y_true, axis = 1)      
        y_true = tf.cast(y_true, dtype = tf.float32)
        loss = Binary_Crossentropy(y_true, y_pred, from_logits = self.params['from_logits'])
        
        #https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/losses/focal_loss.py#L25-L81
        alpha_factor = 1.0
        if 'alpha' in self.params:
            
            alpha = tf.cast(self.params['alpha'], dtype = y_true.dtype)
            alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        loss = tf.math.reduce_sum(alpha_factor * loss, axis = -1)
    
            
        return loss
    