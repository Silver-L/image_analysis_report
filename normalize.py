import pandas as pd
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='py, csv_path, outdir')
    parser.add_argument('--csv_path', '-i1', default="D:\M1_lecture\lecture\image_analysis\data.xlsx", help= 'data path')
    parser.add_argument('--outdir', '-i2', default="D:/M1_lecture/lecture/image_analysis", help='out dir')
    args = parser.parse_args()

    # check folder
    if not(os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    df = pd.read_excel(args.csv_path, header=0)
    df_matrix = df.as_matrix()

    # extract data
    df_matrix = df_matrix[:,1:]
    status = np.copy(df_matrix[:,16])
    df_matrix[:, 16] = df_matrix[:, 22]
    df_matrix[:, 22] = status
    # print(df_matrix.shape)

    # normalize
    for i in range(df_matrix.shape[1]-1):
        mean = np.mean(df_matrix[:, i])
        var = np.var(df_matrix[:, i])
        df_matrix[:, i] = (df_matrix[:, i] - mean) / np.sqrt(var)

    np.savetxt(os.path.join(args.outdir, 'normalized_data.csv'), df_matrix, delimiter=',')


if __name__ == '__main__':
    main()