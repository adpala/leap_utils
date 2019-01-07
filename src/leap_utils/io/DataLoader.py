"""Load data conveniently."""
import yaml
import importlib
import os
import logging
from typing import Callable
from itertools import product


class DataLoader:
    """DataLoader.

    More documentation:
    - config format
    - types with required and optional fields
    '''
        poses:
          folder: res
          suffix: _poses.h5
          loader: hdf5
    '''
    - loader system with required and optional fields (by type, by extension, by loader_fun)
    '''
        csv:
            extensions: [csv]
            module: pandas
            function: from_csv
            # args: [arguments]
            # kwargs: {Delimiter: \t}
            '''
    - add fixed files: e.g. network models?
    """

    def __init__(self, config_filename: str, root: str = None) -> None:
        """Initialize DataLoader with data from config.

        Args:
            config_filename: str = 'io.yaml' (TODO: or Dict)
            root: str = None
        """
        self._config_filename = config_filename
        self._config = self._load_config()
        self.types = self._config['types']
        self.loaders = self._config['loaders']
        if root is not None:
            self._config['root'] = root

    def get(self, typ: str, identifier: str,
            loader_fun: Callable = None, loader_fun_args: list = [], loader_fun_kwargs: dict = {}):
        """Load specific type of data from a dataset identifier.

        Args:
            typ: see self.types
            identifier: datename
            loader_fun: function handle (overrides config and file extension)
            loader_fun_args: = [] (overrides config and file extension)
            loader_fun_kwargs = {} (overrides config and file extension)
        """
        if typ not in self.types.keys():
            raise KeyError(f'Unknown file type "{typ}". Needs to be one of {self.types.keys()}.')

        filepath = self._filename(typ, identifier)
        logging.debug(f'FILEPATH assembled as "{filepath}".')

        if loader_fun is None:
            if 'loader' in self.types[typ]:
                loader_name = self.types[typ]['loader']
                logging.debug(f'LOADER defined as {loader_name} description of file type "{typ}".')
            else:
                loader_name = self._idfile(filepath)
                logging.debug(f'LOADER inferred as {loader_name} from file path "{filepath}".')

            try:
                loader = self.loaders[loader_name]
            except KeyError:
                raise KeyError(f'Unknown file loader "{loader_name}". Needs to be one of {self.loaders.keys()}.')

            module = self._import(loader['module'])
            try:
                loader_fun_strg = f"module.{loader['function']}"
            except KeyError:
                raise KeyError(f"need to specify 'function' field in configuration for {'typ'}")

            loader_fun = eval(loader_fun_strg)

            if not loader_fun_args and 'args' in loader:
                loader_fun_args = loader['args']

            if not loader_fun_kwargs and 'kwargs' in loader:
                loader_fun_kwargs = loader['kwargs']
        else:
            logging.debug(f'LOADER {loader_fun} supplied via argument.')

        logging.debug(f"LOADER COMMAND is {loader_fun}.")
        return loader_fun(filepath, *loader_fun_args, **loader_fun_kwargs)

    def path(self, typ, datename):
        return self._filename(typ, datename)

    def _load_config(self):
        with open(self._config_filename, 'r') as f:
            return yaml.load(f)

    def _filename(self, typ, datename):
        if not isinstance(self.types[typ]['folder'], (list, tuple)):
            self.types[typ]['folder'] = [self.types[typ]['folder']]
        if not isinstance(self.types[typ]['suffix'], (list, tuple)):
            self.types[typ]['suffix'] = [self.types[typ]['suffix']]

        filepath = None
        filepaths_tried = []
        for folder, suffix in product(self.types[typ]['folder'], self.types[typ]['suffix']):
            _filepath = os.path.join(self._config['root'],
                                     folder,
                                     datename,
                                     datename + suffix)
            filepaths_tried.append(_filepath)
            if os.path.exists(_filepath):
                filepath = _filepath
                break

        if filepath is None:
            raise FileNotFoundError(f"Could not find {typ} file for {datename}, tried {filepaths_tried}.")
        return filepath

    def _loader(self):
        pass

    def _import(self, module_name):
        return importlib.import_module(module_name)

    def _idfile(self, filepath):
        for name, loader in self.loaders.items():
            for ext in loader['extensions']:
                if filepath.endswith(ext):
                    return name
