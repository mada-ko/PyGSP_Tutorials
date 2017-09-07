import numpy as np


def img_neighbor_indices(img, neighbor, flag_out_of_index=-1):
    """Return the indices of 4 or 8 neigbor in each pixel."""
    img_row, img_col = img.shape
    idx = np.arange(img_row*img_col)  # Convert to vertical vector
    up_idx = np.arange(start=0, stop=img_row*img_col, step=img_row)  # the indices of image upper side
    down_idx = np.arange(start=img_row-1, stop=img_row*img_col, step=img_row)  # the indices of image down side

    if neighbor == 8:
        # 8-neighbor indices of input image
        adj_idx = np.tile(idx, (neighbor, 1))
        adj_idx[0, :] += -img_row-1
        adj_idx[1, :] += -img_row
        adj_idx[2, :] += -img_row+1
        adj_idx[3, :] += -1
        adj_idx[4, :] += 1
        adj_idx[5, :] += img_row-1
        adj_idx[6, :] += img_row
        adj_idx[7, :] += img_row+1

        # Exception handling on image edge
        adj_idx[np.ix_([0, 3, 5], up_idx)] = flag_out_of_index
        adj_idx[np.ix_([2, 4, 7], down_idx)] = flag_out_of_index
        adj_idx[adj_idx < 0] = flag_out_of_index
        adj_idx[adj_idx > img_row*img_col-1] = flag_out_of_index

    elif neighbor == 4:
        # 4-neighbor indices of input image
        adj_idx = np.tile(idx, (neighbor, 1))
        adj_idx[0, :] += -img_row
        adj_idx[1, :] += -1
        adj_idx[2, :] += 1
        adj_idx[3, :] += img_row

        # Exception handling on image edge
        adj_idx[np.ix_([1], up_idx)] = flag_out_of_index
        adj_idx[np.ix_([2], down_idx)] = flag_out_of_index
        adj_idx[adj_idx < 0] = flag_out_of_index
        adj_idx[adj_idx > img_row*img_col-1] = flag_out_of_index

    else:
        raise ValueError

    return adj_idx


def img_to_wmat(img, neighbor=8):
    """Image convert to weight adjacency matrix."""
    img_row, img_col = img.shape
    img_vec = img.flatten()
    xx, yy = np.meshgrid(img_vec, img_vec)
    diff = xx - yy
    diff = diff.flatten()
    diff = np.fabs(diff)

    adj_idx = img_neighbor_indices(img, neighbor, -(img_row**2)*(img_col**2))
    idx_add = np.arange(start=0, stop=(img_row**2)*(img_col**2), step=img_row*img_col)
    add_mat = np.tile(idx_add, (8, 1))
    adj_idx = adj_idx + add_mat

    idx = adj_idx[adj_idx > 0]
    W_mat = diff.copy()
    diff[idx] = -1
    W_mat = W_mat * (diff == -1)
    W_mat = np.reshape(W_mat, (img_row*img_col, img_row*img_col))

    return(W_mat)








if __name__ == '__main__':
    img_row = 3
    img_col = 2
    img = np.random.randn(img_row, img_col)
    w_mat = img_neighbor_indices(img, 4)
    print(w_mat)
