import os
from tqdm import tqdm

try:
    import pandas as pd
except ImportError:
    err_msg = (
        "The optional dependency pandas must be installed to run this recipe.\n"
    )
    err_msg += "Install using `pip install pandas`.\n"
    raise ImportError(err_msg)

WORDS = {'unknown':-1,
        '小蓝小蓝':0,
        '管家管家':1}


def gen_scp(csv_file, output_path):
    """[summary]

    Parameters
    ----------
    csv_file : [string]
        [csv absolute path]
    output_path : [string]
        [folde to save scp and tex]
    """

    abs_dir = os.path.abspath(output_path)
    scp_path = os.path.join(abs_dir, 'wav.scp')
    text_path = os.path.join(abs_dir, 'text.txt')
    with open(scp_path, 'w', encoding='utf-8') as f_wav, \
         open(text_path, 'w', encoding='utf-8') as f_text:
        csv_file = pd.read_csv(csv_file)
        for index, row in tqdm(csv_file.iterrows()):
            ID, wav, label = row['ID'], row['wav'], row['command']
            f_wav.write('{} {}\n'.format(ID, wav))
            f_text.write('{} {}\n'.format(ID, WORDS[label]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generate scp and text from csv')

    parser.add_argument("csv_path", type=str,
                        help="path to read csv")
    parser.add_argument("scp_path", type=str,
                        help="path to save scp")
    args = parser.parse_args()

    gen_scp(args.csv_path, args.scp_path)
