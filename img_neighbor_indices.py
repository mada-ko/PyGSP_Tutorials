import numpy as np


def img_neighbor_indices(img, neighbor, flag_out_of_index=-1):
    """Return the indices of 4 or 8 neigbor in each pixel."""
    img_row, img_col = img.shape
    idx = np.c_[np.arange(img_row*img_col)]  # Convert to vertical vector
    up_idx = np.arange(start=0, stop=img_row*img_col, step=img_row)  # the indices of image upper side
    down_idx = np.arange(start=img_row-1, stop=img_row*img_col, step=img_row)  # the indices of image down side

    if neighbor == 8:
        # 8-neighbor indices of input image
        adj_idx = np.tile(idx, (1, neighbor))
        adj_idx[:, 0] += -img_row-1
        adj_idx[:, 1] += -img_row
        adj_idx[:, 2] += -img_row+1
        adj_idx[:, 3] += -1
        adj_idx[:, 4] += 1
        adj_idx[:, 5] += img_row-1
        adj_idx[:, 6] += img_row
        adj_idx[:, 7] += img_row+1

        # Exception handling on image edge
        adj_idx[np.ix_(up_idx, [0, 3, 5])] = flag_out_of_index
        adj_idx[np.ix_(down_idx, [2, 4, 7])] = flag_out_of_index
        adj_idx[adj_idx < 0] = flag_out_of_index
        adj_idx[adj_idx > img_row*img_col-1] = flag_out_of_index

    elif neighbor == 4:
        # 4-neighbor indices of input image
        adj_idx = np.tile(idx, (1, neighbor))
        adj_idx[:, 0] += -img_row
        adj_idx[:, 1] += -1
        adj_idx[:, 2] += 1
        adj_idx[:, 3] += img_row

        # Exception handling on image edge
        adj_idx[np.ix_(up_idx, [1])] = flag_out_of_index
        adj_idx[np.ix_(down_idx, [2])] = flag_out_of_index
        adj_idx[adj_idx < 0] = flag_out_of_index
        adj_idx[adj_idx > img_row*img_col-1] = flag_out_of_index

    else:
        raise ValueError

    return adj_idx


if __name__ == '__main__':
    img_row = 4
    img_col = 5
    img = np.random.randn(img_row, img_col)
    idx = img_neighbor_indices(img, 8)
    print(idx)
