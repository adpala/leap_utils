import numpy as np
from leap_utils.io import LArray, print_table
import logging
import warnings


warnings.simplefilter('default')

body_parts = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR',
              'tail']
positions = np.random.randint(0, 100, (20, len(body_parts), 2))
row_index = {'filename': np.array(['rec1'] * 10 + ['rec2'] * 10), 'frame': np.array(list(range(10)) * 2),
             'fly': np.array([0, 1] * 5 + [0, 1, 2, 3, 4] * 2)}

la = LArray(positions, col_names=body_parts, row_index=row_index)
# add row index
row_sex = {'sex': np.array(['male', 'female'] * 5 + ['male', 'male', 'male', 'male', 'female'] * 2)}
la.row_index.update(row_sex)


def test_norow():
    test_get( LArray(positions, col_names=body_parts))


def test_la_nocol():
    test_get(LArray(positions, row_index=row_index))


def test_la_nonothing():
    test_get(LArray(positions))


def test_tuplify():
    print('bool:', la._tuplify(True))
    print('int:', la._tuplify(1))
    print('float', la._tuplify(1.0))
    print('range', la._tuplify(range(3)))
    print('tuple', la._tuplify(tuple(range(3))))
    print('nparray', la._tuplify(np.arange(3)))


def test_get(la=la):
    print('la[:, 11, :] -> (20, 2) == ', la[:, 11, :].shape)
    print('la[:, "tail", :] -> (20, 2) == ', la[:, 'tail', :].shape)
    print('la[:, "WRONG", :] -> (20, 0, 2) == ', la[:, 'WRONG', :].shape)
    print('la[:, [4, 6], :] -> (20, 2, 2) == ', la[:, [4, 6], :].shape)
    print('la[:, [4, 6]] -> (20, 2, 2) == ', la[:, [4, 6]].shape)

    print("la[::2, ('tail', 'head')] -> (10, 2, 2)", la[::2, ('tail', 'head')].shape)
    print("la[::2, ['tail', 'head']] -> (10, 2, 2)", la[::2, ['tail', 'head']].shape)

    print('----------- ROWS ----------------')
    print("la.rows('rec2') -> (10, 12, 2) ==", la.rows('rec2').shape)
    print("la.rows(fly=1) -> (7, 12, 2) ==", la.rows(fly=1).shape)
    print("la.rows('rec2', fly=1) -> (2, 12, 2) ==", la.rows('rec2', fly=1).shape)
    print("la.rows('rec1', fly=1) -> (5, 12, 2) ==", la.rows('rec1', fly=1).shape)
    print("la.rows('rec1', fly=[0, 1]) -> (10,12,2) ==", la.rows('rec1', fly=[0, 1]).shape)
    print("la.rows(('rec1', 'rec2'), fly=[0, 1]) -> (14,12,2) ==", la.rows(('rec1', 'rec2'), fly=[0, 1]).shape)
    print("la.rows(['rec1', 'rec2'], fly=[0, 1]) -> (14,12,2) ==", la.rows(['rec1', 'rec2'], fly=[0, 1]).shape)
    print("la.rows(frame=range(10), fly=[0, 1]) -> (14, 12, 2) ==", la.rows(frame=range(10), fly=[0, 1]).shape)

    print('----------- ELLIPSES -------------')
    print("la[:2, ...] -> (2,12,1) ==", la[:2, ...].shape)
    print("la[::2, ...] -> (10,12,1) ==", la[::2, ...].shape)
    print("la[2:8:2, ...] -> (3,12,1) ==", la[2:8:2, ...].shape)
    print("la[:, 'tail', ...] -> (20,2) ==", la[:, 'tail', ...].shape)
    print("la[:, 'backR':'tail', ...] -> (20,4,2) ==", la[:, 'backR':'tail', ...].shape)

    mask = np.zeros((20,)).astype(np.bool)
    mask[[1, 4, 5]] = True
    print("la[mask, 'tail', ...] -> (3, 2) ==", la[mask, 'tail', ...].shape)

    print('----------- FANCY ROW MATCHING ---')
    print(la[{'filename': ('rec1', 'rec2'), 'frame': (0, 1, 3)}].shape)
    print(la.rows(('rec1', 'rec2'), (0, 1, 3)).shape)
    print(la.rows(('rec1', 'rec2'), frame=(0, 1, 3)).shape)
    print(la[{'filename': ('rec1', 'rec2'), 'frame': (0, 1, 3)}, ...].shape)
    # print(la['rec1'].shape)
    # print(la['rec1', ...].shape)
    # print(la[('rec1',), ...].shape)
    # print(la[(('rec1',), [1, 2]), ...].shape)
    # print(la[('rec1', [1, 2]), ...].shape)
    # print(la[[('rec1', 'rec2'), 1, [0, 1, 2]], ...].shape)
    # print(la[(('rec1', 'rec2'), [1, 2], [0, 1, 2]), ...].shape)
    # print(la[(('rec1', 'rec2'), [1, 2], [0, 1, 2]), ...].shape)
    print(la[{'filename': ('rec1', 'rec2'), 'frame': (0, 1, 2, 3)}, ...].shape)


def test_set():
    la[0, ...] = np.zeros((12, 2))


def test_print():
    col_headers = la.col_names
    col_values = la
    row_headers = list(la.row_index.keys())
    row_values = np.array(list(la.row_index.values())).T
    print(print_table(col_values, col_headers, row_values, row_headers, last_row=5))
    print(print_table(col_values, col_headers, row_values, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers, row_values=None, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=None, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=None, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=row_values, row_headers=None, last_row=5))
    print(print_table(col_values, col_headers=None, row_values=row_values, row_headers=row_headers, last_row=5))

    la.print()
    la.print_head()
    la.print_tail()
    print(la)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # la['rec1']
    # la['rec1',...]
    # la[('rec1',)]
    # la[('rec1',),...]
    # la[('rec1', [0, 1])]
    # la[('rec1',[0,1]),...]
    # test_get(la)
