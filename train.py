from unet import UNet
from data import Data


D = Data(custom_loss=True)
model = UNet(input_shape=(267, 267, 1), custom_loss=True)()

try:
    model.load_weights('models/weighted_bce_loss.h5')
except FileNotFoundError:
    pass



#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#    try:
#        tf.config.experimental.set_virtual_device_configuration(
#            gpus[0],
#            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=700)])
#    except RuntimeError as e:
#        # Virtual devices must be set before GPUs have been initialized
#        print(e)

model.fit_generator(generator=D.generator(), steps_per_epoch=len(D.labels)*144, epochs=10)

model.save_weights('weighted_bce_loss.h5')
