from sklearn.model_selection import train_test_split
import os
from shutil import rmtree, copy2


def copy2folder(files_to_copy, folder_to_copy_into):
    for file in files_to_copy:
        filename = file.split('/')[-1]
        copy2(file, folder_to_copy_into + '/' + filename)


def prepare_data(data_location='../data/tl_classifier_exceptsmall/simulator',
                 train_data_path='../data/tl_classifier_exceptsmall/simulator/train/',
                 test_data_path='../data/tl_classifier_exceptsmall/simulator/test/'):
    greens = []
    yellows = []
    reds = []
    noLight = []

    # Check if train/validation folders exist. If they do, wipe 'em out
    for folder in [train_data_path, test_data_path]:
        if os.path.exists(folder):
            rmtree(folder)

    # Get all subfolders
    subfolders_tree = os.walk(data_location)

    paths_to_construct = []
    for folder_branch, children, files in subfolders_tree:
        # print(folder_branch)
        # print(children)
        # print(len(files), files)
        # print()
        if folder_branch == data_location:
            paths_to_construct.extend(children)
        if not children:
            origin_paths = [folder_branch + '/' + file for file in files]
            if 'Green' in folder_branch:
                greens.extend(origin_paths)
            if 'Yellow' in folder_branch:
                yellows.extend(origin_paths)
            if 'Red' in folder_branch:
                reds.extend(origin_paths)
            if 'NoTrafficLight' in folder_branch:
                noLight.extend(origin_paths)

    # Create new train/validation folders with the same directory structure as the input data structure
    for path in [train_data_path, test_data_path]:
        os.mkdir(path)
        for folder in paths_to_construct:
            new_folder = path + '/' + folder
            os.makedirs(new_folder)

    # Shuffle and get X% of data, copy to the corresponding train/validation folder
    green_train, green_test = train_test_split(greens,
                                               shuffle=True,
                                               test_size=0.2
                                               )
    red_train, red_test = train_test_split(reds,
                                           shuffle=True,
                                           test_size=0.2
                                           )
    yellow_train, yellow_test = train_test_split(yellows,
                                                 shuffle=True,
                                                 test_size=0.2
                                                 )
    noLight_train, noLight_test = train_test_split(noLight,
                                                   shuffle=True,
                                                   test_size=0.2
                                                   )

    copy2folder(green_train, train_data_path + '/' + 'Green')
    copy2folder(green_test, test_data_path + '/' + 'Green')

    copy2folder(red_train, train_data_path + '/' + 'Red')
    copy2folder(red_test, test_data_path + '/' + 'Red')

    copy2folder(yellow_train, train_data_path + '/' + 'Yellow')
    copy2folder(yellow_test, test_data_path + '/' + 'Yellow')

    copy2folder(noLight_train, train_data_path + '/' + 'NoTrafficLight')
    copy2folder(noLight_test, test_data_path + '/' + 'NoTrafficLight')


if __name__ == '__main__':
    prepare_data()
    # Use the new folder structure for training!
