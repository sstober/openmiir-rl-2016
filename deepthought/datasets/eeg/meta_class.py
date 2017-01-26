__author__ = 'sstober'

from deepthought.util.fs_util import load

class DataFileWithMetaClasses(object):
    def __init__(self, filepath, meta_classes=dict()):

        self.filepath = filepath
        tmp = load(filepath)
        if len(tmp) == 2:
            self.data, self.metadata = tmp
            self.targets = None
        elif len(tmp) == 3:
            self.data, self.metadata, self.targets = tmp
        else:
            raise ValueError('got {} objects instead of 2 or 3.'.format(len(tmp)))

        for class_name, classes in meta_classes.iteritems():
            self._create_meta_class(class_name, classes)


    def _create_meta_class(self, class_name, classes):
            class_map = dict()
            max_value = 0
            for meta in self.metadata:
                key = []
                for label in classes:
                    key.append(meta[label])
                key = tuple(key)

                if not key in class_map:
                    class_map[key] = max_value
                    max_value += 1

                value = class_map[key]

                meta[class_name] = value
                # print key, value, meta


# target_processor implementation
class MetaClassCreator(object):

    def __init__(self, name, labels):
        self.name = name
        self.labels = labels

        self.class_map = dict()
        self.max_value = 0

    def process(self, target, meta):
        key = []
        for label in self.labels:
            key.append(meta[label])
        key = tuple(key)

        if not key in self.class_map:
            self.class_map[key] = self.max_value
            self.max_value += 1

        target = self.class_map[key]

        print key, target, meta
        # meta[self.name] = value
