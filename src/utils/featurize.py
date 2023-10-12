import os
import numpy as np
import logging
import joblib
import scipy.sparse as sparse


def save_matrix(df, matrix, out_path):
    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).toarray
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).toarray

    result = sparse.hstack([id_matrix, label_matrix, matrix], format="csr")

    joblib.dump(result, out_path)
    msg = f"The output matrix saved at: {out_path} of the size: {result.shape} and data type: {result.dtype}"
    logging.info(msg)

    