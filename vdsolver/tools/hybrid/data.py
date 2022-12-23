import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class VTKGroup:
    def __init__(self, root):
        self.root = root

    def __getattr__(self, name):
        result = self.root.findall(name)

        if len(result) == 0:
            try:
                return getattr(self.root, name)
            except Exception:
                raise AttributeError(f'{self.filepath} has no "{name}".')

        elif len(result) == 1:
            return VTKGroup(result[0])

        else:
            return [VTKGroup(r) for r in result]

    def __str__(self):
        return str(self.root)

    def __repr__(self) -> str:
        return repr(self.root)

    def __iter__(self):
        return iter(self.root)


class VTKFile(VTKGroup):
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.tree = ET.parse(filepath)
        super().__init__(self.tree.getroot())


class HybridData:
    """ 
    Data manager for hybrid code.

    Notes
    -----
    [Sample Data Format]

    <?xml version="1.0" ?>\n
    <VTKFile type="StructuredGrid">\n
    <StructuredGrid WholeExtent="0 319 0 319 0 319">\n
      <Piece Extent="0 319 0 319 0 319">\n
        <PointData Scalars="BY">\n
          <DataArray Name="BY" type="Float64" format="ascii">\n
            <<BY-DATA>>\n
          </DataArray Name="BY" type="Float64" format="ascii">\n
        </PointData>\n
        <Points>\n
          <DataArray NumberOfComponents="3" type="Float64" Name="points" format="ascii">\n
            <<POINT-DATA>>\n
          </DataArray>\n
        </Points>\n
      </Piece>\n
    </StructuredGrid>\n
    </VTKFile type="StructuredGrid">
    """

    def __init__(self, filepath: Path, require_points: bool = False, load=True):
        """Create data manager for hybrid code.

        Parameters
        ----------
        filepath : Path
            File path.
        require_points : bool, optional
            True if loading coordinate data, by default False
        load : bool, optional
            True if data is to be loaded immediately, by default True.

            hybrid_data3d.load() needs to be called later if False.
        """
        self.filepath: Path = filepath
        self.require_points: bool = require_points
        if load:
            self.load()

    def load(self):
        """Load VTK data file.
        """
        for event, elem in ET.iterparse(self.filepath, events=['start', 'end']):
            if event == 'start' and elem.tag == 'StructuredGrid':
                extent_str = elem.attrib['WholeExtent']
                self.lim: np.ndarray = np.array(
                    list(map(int, extent_str.split()))).reshape(3, 2)
                self._shape: np.ndarray = self.lim[:, 1] - self.lim[:, 0] + 1

            elif event == 'start' and elem.tag == 'PointData':
                self._dataname: str = elem.attrib['Scalars']

            elif event == 'end' and elem.tag == 'DataArray' and 'Name' in elem.attrib and elem.attrib['Name'] != 'points':
                values = np.zeros(np.product(self.shape))

                text: str = elem.text
                lines = text.strip().splitlines()
                for i, line in enumerate(lines):
                    values[i] = float(line.strip())
                self._values = values.reshape(self.shape)

                if not self.require_points:
                    return

            elif self.require_points \
                    and event == 'end' and elem.tag == 'DataArray'\
                    and 'Name' in elem.attrib and elem.attrib['Name'] == 'points':
                points = np.zeros((np.product(self.shape), 3))
                for i, line in enumerate(elem.itertext()):
                    points[i, :] = list(map(float, line.split()))
                self._points = points.reshape((*self.shape, 3))

            if event == 'end':
                elem.clear()

    @property
    def dataname(self) -> str:
        return self._dataname

    @property
    def shape(self) -> np.ndarray:
        return self._shape

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def points(self) -> np.ndarray:
        return self._points

    # def load(self):
    #     """Load VTK data file.
    #     """
    #     print(0, psutil.virtual_memory().percent)
    #     vtk = VTKFile(self.filepath)
    #     print(1, psutil.virtual_memory().percent)

    #     extent_str = vtk.StructuredGrid.attrib['WholeExtent']
    #     self.lim: np.ndarray = np.array(
    #         list(map(int, extent_str.split()))).reshape(3, 2)
    #     self.shape: np.ndarray = self.lim[:, 1] - self.lim[:, 0] + 1

    #     self.dataname: str = vtk.StructuredGrid.Piece.PointData.attrib['Scalars']

    #     if self.require_points:
    #         points_str = vtk.StructuredGrid.Piece.Points.DataArray.text.strip()
    #         points_splited = points_str.replace('\n', ' ').split()
    #         self.points: np.ndarray = \
    #             np.array(list(map(float, points_splited))) \
    #             .reshape((*self.shape, 3))
    #     print(2, psutil.virtual_memory().percent)

    #     values_str = vtk.StructuredGrid.Piece.PointData.DataArray.text.strip()
    #     values_str_io = io.StringIO(values_str)
    #     print(3, psutil.virtual_memory().percent)

    #     self.values = np.zeros(np.product(self.shape))

    #     # values_splited = values_str.split('\n')

    #     print(4, psutil.virtual_memory().percent)
    #     for i, line in enumerate(values_str_io):
    #         self.values[i] = float(line)
    #     self.values = self.values.reshape(self.shape)

    #     # self.values: np.ndarray = np.array(
    #     #     list(map(float, values_splited))).reshape(self.shape)
    #     print(5, psutil.virtual_memory().percent)
