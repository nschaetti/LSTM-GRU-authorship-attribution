
# Import
import ktrain
from ktrain import text
import numpy as np
import os
import codecs
import argparse

# Classes
classes = ['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7',
           'class8', 'class9', 'class10', 'class11', 'class12', 'class13', 'class14']

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datadir")
parser.add_argument("--k", default=10)
args = parser.parse_args()

# Average accuracy
average_accuracy = np.zeros(args.k)

# For each fold
for k in range(args.k):
    # Validation directory
    fold_dir = os.path.join(args.datadir, "k{}".format(k))
    fold_val_dir = os.path.join(fold_dir, "val")

    # Load training and validation data from a folder
    (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(
        fold_dir,
        maxlen=512,
        preprocess_mode='bert',
        classes=classes
    )

    # Load BERT
    learner = ktrain.get_learner(
        text.text_classifier('bert', (x_train, y_train)),
        train_data=(x_train, y_train),
        val_data=(x_test, y_test),
        batch_size=16
    )

    # Get good learning rate
    learner.lr_find()

    # Plot
    learner.lr_plot()

    # Train the model
    learner.fit(2e-5, 20, early_stopping=5)
    # learner.fit_onecycle(2e-5, 1)

    # Get the predictor
    predictor = ktrain.get_predictor(learner.model, preproc)

    # Counting
    count = 0
    total = 0

    # For each author
    for a in range(15):
        # Author directory
        author_val_dir = os.path.join(fold_val_dir, u"class{}".format(a))

        # Data
        data = list()

        # For each file
        for author_file in os.listdir(author_val_dir):
            # Read file
            data.append(codecs.open(os.path.join(author_val_dir, author_file), "r", encoding="utf-8").read())
        # end for

        # Predict class
        pred = predictor.predict(data)

        # For each prediction
        for p in pred:
            if p == "class{}".format(a):
                count += 1
            # end if
        # end for

        # Total
        total += len(pred)
    # end for

    # Accuracy
    accuracy = count / total

    # Print success rate
    print("ACCURACY TEST : {}".format(accuracy))
    average_accuracy[k] = accuracy
# end for

# Print average
print("Average accuracy : {}".format(np.mean(average_accuracy)))
