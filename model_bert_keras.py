
# Import
import ktrain
from ktrain import text
import numpy as np
from tools import argument_parsing, functions
import os
import codecs

# Parse args
args, use_cuda, param_space, xp = argument_parsing.parser_training()

# Classes
classes = ['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7',
           'class8', 'class9', 'class10', 'class11', 'class12', 'class13', 'class14']

# Last space
last_space = dict()

# Iterate
for space in param_space:
    # Params
    hidden_size, cell_size, feature, lang, dataset_start, window_size, learning_window, embedding_size, rnn_type, num_layers, dropout, output_dropout = functions.get_params(
        space
    )

    # Set experience state
    xp.set_state(space)

    # For each sample
    for n in range(args.n_samples):
        # Set sample
        xp.set_sample_state(n)

        # Average
        average_k_fold = np.array([])

        # OOV
        oov = np.array([])

        # For each fold
        for k in range(args.k):
            # Validation directory
            fold_dir = os.path.join("bert_data/document", "k{}".format(k))
            fold_val_dir = os.path.join(fold_dir, "val")

            # Load training and validation data from a folder
            (x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(
                "bert_data/document/k{}".format(k),
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

            # Train the model
            learner.fit(0.001, 3, cycle_len=1, cycle_mult=2, early_stopping=5)

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
                    data.append(codecs.open(author_file, "r", encoding="utf-8").read())
                # end for

                # Predict class
                pred = predictor.predict(data)

                # For each prediction
                for p in pred:
                    if p == str(a):
                        count += 1
                    # end if
                # end for

                # Total
                total += len(pred)
            # end for

            # Accuracy
            accuracy = count / total

            # Print success rate
            xp.add_result(accuracy)
        # end for
    # end for
# end for
