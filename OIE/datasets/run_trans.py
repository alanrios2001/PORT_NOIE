import translate

paths = ["translated/lsoie/train", "translated/lsoie/test", "translated/lsoie/dev", "translated/lsoie/dev"]
files = ["ls_train.conll", "ls_test.conll", "ls_dev.conll", "dev.conll"]

cache = "translated/cache"
BATCH_SIZE = 64
TRANSLATED = True
use_google = True
test_size = 0.0
dev_size = 0.0

for path, file in zip(paths, files):
    translate.run(BATCH_SIZE,
                  path,
                  file,
                  test_size,
                  dev_size,
                  TRANSLATED,
                  use_google=use_google,
                  debug=False,
                  cache_dir=cache,
                  sequential=True
                  )
