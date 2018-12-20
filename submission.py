import fire
import h5py
import pandas as pd

DEV_DATA_LIST = ['../dev.chunk.01']


def write_prediction_result(pred_path, out_path):
    print('Reorder\t\t: ' + pred_path + ' ...')
    pid_order = []
    for data_path in DEV_DATA_LIST:
        h = h5py.File(data_path, 'r')['dev']
        pid_order.extend(h['pid'][::])
    pid_order = [x.decode('utf-8') for x in pid_order]
    pred_df = pd.read_csv(pred_path, sep='\t', index_col=0, header=None)
    pred_ordered = pred_df.loc[pred_df.index.intersection(pid_order)].reindex(pid_order)
    pred_ordered = pred_ordered.fillna(-1)
    pred_ordered = pred_ordered.astype(int)
    pred_ordered.to_csv(out_path, sep='\t', header=None)
    print('Done! \t\t: ' + out_path)


if __name__ == '__main__':
    fire.Fire({'write_result': write_prediction_result})
