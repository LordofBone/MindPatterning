# From https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
# Thanks to them
import os
import glob
import pandas as pd


def combine_csvs(file_match, output_file, test_mode=False):
    # Work from the current directory
    os.chdir(os.getcwd())

    # Grab all file names that match the type of file, in this case relaxed/non-relaxed data
    all_filenames = [i for i in glob.glob('{}*.csv'.format(file_match))]

    # Print the file names found
    print(all_filenames)

    # If test mode is set then the model will simply display the files found and not combine them
    if not test_mode:
        #combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
        #export to csv
        # combined_csv.to_csv(output_file, index=False, encoding='utf-8-sig')
        combined_csv.to_csv(output_file, index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find and combine csv files with names containing text')

    parser.add_argument('-match', action="store", dest="file_name", type=str,
                        help='File name to match csv files on')
    parser.add_argument('-fname', action="store", dest="output_name", type=str,
                        help='Output file name for combined csv files')

    args = parser.parse_args()
    file_name = args.file_name
    output_name = args.output_name

    combine_csvs(file_name, output_name, True)
