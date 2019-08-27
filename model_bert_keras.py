
# Import
import ktrain
from ktrain import text
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datadir")
args = parser.parse_args()

# Classes
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']

# Load training and validation data from a folder
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(
    args.datadir,
    maxlen=512,
    preprocess_mode='bert',
    classes=classes
)

# Print text classifiers
text.print_text_classifiers()

# Load BERT
learner = ktrain.get_learner(
    text.text_classifier('bert', (x_train, y_train)),
    train_data=(x_train, y_train),
    val_data=(x_test, y_test),
    batch_size=6
)

# Train the model
learner.fit_onecycle(2e-5, 1)
