import json
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, History

from data import Data
from unet import make_unet


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""""""""""" CONFIG """"""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

LOAD_PREVIOUS = None
USE_CUSTOM_LOSS = False
MODEL_SAVE_PATH = "models/test/bce_loss.h5"
HISTORY_PATH = "models/history/experience_history.json"
MODEL_CHECKPOINT = 5
MODEL_CHECKPOINT_PATH = "models/test/checkpoints/experience_model_{epoch}.h5"
EPOCHS = 2
STEPS_PER_EPOCH = None

data_kwargs = {
    'custom_loss': USE_CUSTOM_LOSS
}

model_kwargs = {
    'input_shape': (267, 267, 1),
    'dropout': 0.3,
    'batch_normalisation': False,
    'lr': 0.001,
    'beta_1': 0.9,
    'beta_2': 0.999,
    'custom_loss': USE_CUSTOM_LOSS,
}

callbacks = []
callbacks.append(ModelCheckpoint(MODEL_CHECKPOINT_PATH, period=MODEL_CHECKPOINT))
callbacks.append(TensorBoard(
    log_dir='models/logs/',
    histogram_freq=1,
    write_grads=True,
    write_images=True,
    update_freq=30
))
callbacks.append(History())

fit_kwargs = {
    'epochs': EPOCHS,
    'callbacks': callbacks
}


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""""" PREPARE TRAINING """"""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

D = Data(**data_kwargs)
model = make_unet(**model_kwargs)

fit_kwargs['generator'] = D.generator()
fit_kwargs['steps_per_epoch'] = STEPS_PER_EPOCH if STEPS_PER_EPOCH else len(D.labels)*144
fit_kwargs['validation_data'] = D.generator(validation=True)
fit_kwargs['validation_steps'] = 10


if LOAD_PREVIOUS is not None:
    model.load_weights(LOAD_PREVIOUS)


# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# """"""""""""""""""""""""" TRAINING """""""""""""""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

history = model.fit_generator(**fit_kwargs)

model.save_weights(MODEL_SAVE_PATH)

with open(HISTORY_PATH, 'w') as f:
    json.dump(history.history, f)
