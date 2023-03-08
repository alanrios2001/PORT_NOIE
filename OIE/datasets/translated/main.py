import translate

BATCH_SIZE = 64
TRANSLATED = False
test_size = 0.0
dev_size = 0.0
#paths = ["lsoie/train", "lsoie/dev", "lsoie/test"]
#files = ["ls_train.conll", "ls_dev.conll", "la_test.conll"]

paths = ["lsoie/dev"]
files = ["dev.conll"]
for path, file in zip(paths, files):
    translate.run(BATCH_SIZE, path, file, test_size, dev_size, TRANSLATED)
