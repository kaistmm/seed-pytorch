import glob, argparse


def main(args):
    target_dirs = args.target_dirs
    output_filename = args.output_filename
    extensions = args.extensions

    total_data_list = []
    for target_dir in target_dirs:
        for ext in extensions:
            if not ext.startswith('.'):
                ext = '.' + ext
            data_list = glob.glob(target_dir + '**/*' + ext, recursive=True)
            total_data_list.extend(data_list)

    # check duplicate data_file
    total_data_list = list(set(total_data_list))

    print("Done collecting wav file paths")

    if len(total_data_list) != len(set(total_data_list)):
        print("Warning: There are duplicate data_files in the total_data_list. Please check the data_list.")
        if input("Do you want to continue? (y/n): ") != 'y':
            exit()

    with open(output_filename, 'w') as f:
        for filename in total_data_list:
            f.write(filename + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a dataset manifest file from data files in specified directories.")
    parser.add_argument('--target_dirs', nargs='+', required=True,
                        default=[
                            '/path/to/datasetA/',
                            '/path/to/datasetB/',
                            '/path/to/datasetC/',
                        ],
                        help='List of target directories to search for wav files.')
    parser.add_argument('--output_filename', type=str, required=True,
                        help='Name of the output manifest file.')
    parser.add_argument('--extensions', nargs='+', default=['.wav'],
                        help='List of file extensions to search for (e.g., .wav .flac .mp3). Default is .wav')

    args = parser.parse_args()

    """
    Example:
    python make_seed_dataset.py \
    --target_dirs /path/to/libritts_R_16k/train-clean-100/ \
                  /path/to/libritts_R_16k/train-clean-360/ \
                  /path/to/librilight_16k/small/ \
    --output_filename ./manifests/train_libritts+light_1000h.txt \
    --extensions .wav 
    """

    print("Before running this script, please check the following:")
    print("1. Check your data format like 16k sampling rate audio, mono channel, etc.")
    print("2. Check your data list is correct.")
    if input("Do you want to continue? (y/n): ") != 'y':
        exit()
    main(args)




