from pathlib import Path

MAX_TRAIN = 420
MAX_VALID = 140
p_train = Path('data/train')
p_valid = Path('data/valid')


def trim_data(data_path, max_files):
    for class_dir in data_path.iterdir():
        i = 1
        for img_file in class_dir.iterdir():
            if i > max_files:
                img_file.unlink()

            i += 1

        if i < max_files:
            print(i)


trim_data(p_train, MAX_TRAIN)
trim_data(p_valid, MAX_VALID)
