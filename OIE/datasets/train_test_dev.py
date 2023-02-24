import random

def train_dev_test(test_slice: float, dev_slice: float, output_name: str, in_path: str, out_path: str):
    with open(f"{in_path}/{output_name}_corpus.txt", "r", encoding="utf-8") as file:
        lines = file.read().split("\n\n")
        random.shuffle(lines)
        test_size = int(len(lines) * test_slice)
        dev_size = int(len(lines) * dev_slice)
        train_size = len(lines) - test_size - dev_size
        train = lines[:train_size]
        dev = lines[train_size:train_size + dev_size]
        test = lines[train_size + dev_size:]
        file.close()

    with open(f"{out_path}/{output_name}_train.txt", "a", encoding="utf-8") as file:
        file.writelines("\n\n".join(train))
        file.close()
        print(f"train file saved in: {out_path}/{output_name}_train.txt")
    if len(dev) > 0:
        with open(f"{out_path}/{output_name}_dev.txt", "a", encoding="utf-8") as file:
            file.writelines("\n\n".join(dev))
            file.close()
            print(f"dev file saved in: {out_path}/{output_name}_dev.txt")
    if len(test) > 0:
        with open(f"{out_path}/{output_name}_test.txt", "a", encoding="utf-8") as file:
            file.writelines("\n\n".join(test))
            file.close()
            print(f"test file saved in: {out_path}/{output_name}_test.txt")

    print("train: ", len(train), "|| dev: ", len(dev), "|| test: ", len(test))