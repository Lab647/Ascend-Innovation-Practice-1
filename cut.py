import argparse
import os


def cut(raw_folder, new_folder, each_count=100):
    for path, dirs, fs in os.walk(raw_folder):
        for d in dirs:
            dir_path = os.path.join(path, d)
            dir_name = dir_path.replace(raw_folder + '/', '')
            dir_path = os.path.join(new_folder, dir_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            print(dir_path)
        for f in fs[:each_count]:
            file_path = os.path.join(path, f)
            dir_name, file_name = file_path.replace(raw_folder + '/', '').split('/', maxsplit=1)
            new_file = os.path.join(new_folder, dir_name, file_name)
            # print(file_path, '-->', new_file)
            with open(new_file, 'wb') as f1, open(file_path, 'rb') as f2:
                f1.write(f2.read())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="enter the path of datapath, newpath and count (default=100)")
    parser.add_argument("-dp", "--datapath")
    parser.add_argument('-np', '--newpath')
    parser.add_argument('-c', '--count', type=int, default=100)
    args = parser.parse_args()
    data_path = args.datapath
    new_path = args.newpath
    count = args.count
    cut(data_path, new_path, count)
