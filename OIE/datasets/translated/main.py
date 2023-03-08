import translate

paths = ["lsoie/dev", "lsoie/test", "lsoie/train"]
files = ["ls_dev.conll", "ls_test.conll", "ls_train.conll"]

BATCH_SIZE = 64
TRANSLATED = False
use_google = True
test_size = 0.0
dev_size = 0.0

for path, file in zip(paths, files):
    translate.run(BATCH_SIZE, path, file, test_size, dev_size, TRANSLATED, use_google=use_google, debug=False)
