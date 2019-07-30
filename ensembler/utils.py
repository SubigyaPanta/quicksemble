import pickle


def load_object(object_filepath):
    with open(object_filepath, 'rb') as handle:
        obj = pickle.load(handle)

    return obj


def save_object(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle)