from leap_utils.io import DataLoader

def test_DataLoader():
    dl = DataLoader('scripts/io.yaml', root='ROOT_OVERRIDE')
    print(dl._config['root'])
    print(dl._config)
    print(dl.types)
    print(dl.loaders)

    print(dl._idfile('test.h5'))
    print(dl._idfile('test.npy'))
    print(dl._idfile('test.mp4'))

def test_LArray():
    pass

# try:
#     dl.get('poposes', '')
# except KeyError as e:
#     print(e)
#     print('successfully raised KeyError')
#
# print(dl.path('tracks', 'localhost-20180720_182837'))
# tracks = dl.get('tracks', 'localhost-20180720_182837')
# print(tracks.keys())
#
# poses = dl.get('poses', 'localhost-20180720_182837')
# print(dl.path('poses', 'localhost-20180720_182837'))
# print(poses.keys())
#
# vr = dl.get('video', 'localhost-20180720_182837')
# print(dl.path('video', 'localhost-20180720_182837'))
# print(vr)
#
# from videoreader import VideoReader
# vr = dl.get('video', 'localhost-20180720_182837', loader_fun=VideoReader)
# print(vr)
