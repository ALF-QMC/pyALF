"""Implamemts bravais lattice object."""
# pylint: disable=invalid-name

import os
import platform
import sys
from ctypes import CDLL, POINTER, byref, c_double, c_int

import numpy as np
from numba import jit

_cache = {}
_use_fortran_init = True


class UnitCell:
    """Lattice unit cell.

    Parameters
    ----------
    latt_grp : h5py.Group
        HDF5 group containing the lattice information.

    """

    # pylint: disable=too-few-public-methods
    def __init__(self, latt_grp):
        self.N_coord = latt_grp.attrs["N_coord"]
        self.Ndim = len(latt_grp.attrs["Orbital1"])
        orbital_positions = []
        i = 0
        while True:
            i += 1
            try:
                orbital_positions.append(latt_grp.attrs[f"Orbital{i}"])
            except KeyError:
                break
        self.orbital_positions = np.array(orbital_positions)
        self.Norb = len(self.orbital_positions)

        if self.Norb != latt_grp.attrs["Norb"]:
            if self.Norb != latt_grp.attrs['N_coord']:
                raise RuntimeError("Non-recoverable mixup with Norb/N_coord")
            print("Fixing mixup between Norb and N_coord")
            print(f"Wrong values Norb={latt_grp.attrs['Norb']}, "
                  f"N_coord={latt_grp.attrs['N_coord']}")
            print(f"Correct values Norb={latt_grp.attrs['N_coord']}, "
                  f"N_coord={latt_grp.attrs['Norb']}")
            self.N_coord = latt_grp.attrs["Norb"]
            # latt_grp.attrs["N_coord"] = latt_grp.attrs["Norb"]
            # latt_grp.attrs["Norb"] = self.Norb


class Lattice:
    """Finite size Bravais lattice object, mirroring ALF's Lattice type.

    Parameters
    ----------
    *args : dict, tuple, or list

        if dict: {'L1': L1, 'L2': L2, 'a1': a1, 'a2': a2}.

        if tuple or list: [L1, L2, a1, a2].

        L1, L2: 2d vectors defining periodic boundary conditions.

        a1, a2: 2d primitive vectors.

    force_python_init : bool, default=False
        Force the usage of Python version of the initialization.
        Default behavior is to first try compiled Fortran and fall back to
        Python if that fails.

    Attributes
    ----------
    L1, L2 : arrays of floats
        2d vectors defining periodic boundary conditions in real space.
    a1, a2 : arrays of floats
        2d primitive vectors in real space.
    BZ1, BZ2 : arrays of floats
        2d vectors defining periodic boundary conditions in k space.
    b1, b2 : arrays of floats
        2d primitive vectors in k space.
    N : int
        Number of unit cells
    r : array of floats, shape=(N, 2)
        Real-space coordinates.
    k : array of floats, shape=(N, 2)
        K-space coordinates.
    b1_perp : array, shape=(2,)
    b2_perp : array, shape=(2,)
    L : int
    listr : array of ints, shape=(N, 2)
        `r[i] = listr[i, 0]*a1 + listr[i, 1]*a2`
    listk : array of ints, shape=(N, 2)
        `k[i] = listk[i, 0]*a1 + listk[i, 1]*b2`
    invlistr : array of ints, shape=(2*L+1, 2*L+1)
    invlistk : array of ints, shape=(2*L+1, 2*L+1)
    nnlistr : array of ints, shape=(N, 3, 3)
    nnlistk : array of ints, shape=(N, 3, 3)
    imj : array of ints, shape=(N, N)

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, *args, force_python_init=False):
        if len(args) == 1:
            self.L1 = np.array(args[0]["L1"], dtype=float)
            self.L2 = np.array(args[0]["L2"], dtype=float)
            self.a1 = np.array(args[0]["a1"], dtype=float)
            self.a2 = np.array(args[0]["a2"], dtype=float)
        elif len(args) == 4:
            self.L1 = np.array(args[0], dtype=float)
            self.L2 = np.array(args[1], dtype=float)
            self.a1 = np.array(args[2], dtype=float)
            self.a2 = np.array(args[3], dtype=float)
        else:
            raise TypeError(f'Wrong number of arguments {len(args)}')

        s = f'L1={self.L1}L2={self.L2}a1={self.a1}a2={self.a2}'
        if s in _cache:
            (self.BZ1, self.BZ2, self.b1, self.b2,
             self.b1_perp, self.b2_perp, self.L, self.N,
             self.listr, self.invlistr, self.nnlistr,
             self.listk, self.invlistk, self.nnlistk,
             self.imj, self.r, self.k) = _cache[s]
        else:
            global _use_fortran_init  # pylint: disable=global-statement
            if _use_fortran_init and (not force_python_init):
                try:
                    init = _init0(self.L1, self.L2, self.a1, self.a2)
                except OSError:
                    print("Compiled Fortran lattice initialisazion does not work, "
                          "falling back to Python version", file=sys.stderr)
                    _use_fortran_init = False
                    init = _init1(self.L1, self.L2, self.a1, self.a2)
                except NotImplementedError:
                    print('No lattices binary for this system architecture: '
                          f'"{platform.machine()}".'
                          'Falling back to Python version', file=sys.stderr)
                    _use_fortran_init = False
                    init = _init1(self.L1, self.L2, self.a1, self.a2)
            else:
                init = _init1(self.L1, self.L2, self.a1, self.a2)

            (self.BZ1, self.BZ2, self.b1, self.b2,
             self.b1_perp, self.b2_perp, self.L, self.N,
             self.listr, self.invlistr, self.nnlistr,
             self.listk, self.invlistk, self.nnlistk,
             self.imj) = init

            self.r = np.empty((self.N, 2))
            self.k = np.empty((self.N, 2))
            for n in range(self.N):
                self.r[n] = self.listr[n, 0]*self.a1 + self.listr[n, 1]*self.a2
                self.k[n] = self.listk[n, 0]*self.b1 + self.listk[n, 1]*self.b2

            _cache[s] = (
                self.BZ1, self.BZ2, self.b1, self.b2,
                self.b1_perp, self.b2_perp, self.L, self.N,
                self.listr, self.invlistr, self.nnlistr,
                self.listk, self.invlistk, self.nnlistk,
                self.imj, self.r, self.k)

    def __repr__(self):
        class_name = type(self).__name__
        params = {'L1': self.L1, 'L2': self.L2, 'a1': self.a1, 'a2': self.a2}
        return f"{class_name}({params!r})"

    def periodic_boundary_k(self, k):
        """Apply periodic boundary conditions on vector in k space."""
        return _periodic_boundary(np.array(k, dtype=float), self.BZ1, self.BZ2)

    def periodic_boundary_r(self, r):
        """Apply periodic boundary conditions on vector in r space."""
        return _periodic_boundary(np.array(r, dtype=float), self.L1, self.L2)

    def r_to_n(self, r):
        """Map vector in r space to integer running over all lattice points."""
        r1 = self.periodic_boundary_r(r)

        n1 = int(round(np.dot(self.BZ1, r1) / (2*np.pi)))
        n2 = int(round(np.dot(self.BZ2, r1) / (2*np.pi)))
        n = self.invlistr[n1, n2]

        if not np.allclose(r1, self.r[n]):
            raise RuntimeError(f'r not found {r} {r1} {n1} {n2} {n} {self.r[n]}')
        return n

    def k_to_n(self, k):
        """Map vector in k space to integer running over all lattice points."""
        k1 = self.periodic_boundary_k(np.array(k))

        n1 = int(round(np.dot(self.b1_perp, k1)))
        n2 = int(round(np.dot(self.b2_perp, k1)))
        n = self.invlistk[n1, n2]

        if not np.allclose(k1, self.k[n]):
            raise RuntimeError(f'k not found {k} {k1} {n1} {n2} {n} {self.k[n]}')
        return n

    def fourier_K_to_R(self, X):
        """Fourier transform from k to r space.

        Last index of input has to run over all lattice points in k space.

        Last index of output runs over all lattice points in r space.
        """
        if X.shape[-1] != self.N:
            raise TypeError("Last index of X has wrong number of elements")
        Y = np.zeros(X.shape, dtype=X.dtype)
        for i in range(self.N):
            for j in range(self.N):
                Y[..., i] += X[..., j]*np.exp(1j*np.dot(self.r[i], self.k[j]))
        Y = Y/self.N
        return Y

    def fourier_R_to_K(self, X):
        """Fourier transform from r to k space.

        Last index of input has to run over all lattice points in r space.

        Last index of output runs over all lattice points in k space.
        """
        if X.shape[-1] != self.N:
            raise TypeError("Last index of X has wrong number of elements")
        Y = np.zeros(X.shape, dtype=X.dtype)
        for i in range(self.N):
            for j in range(self.N):
                Y[..., i] += X[..., j]*np.exp(-1j*np.dot(self.k[i], self.r[j]))
        Y = Y/self.N
        return Y

    def rotate(self, n, theta):
        """Rotate vector in k space.

        Parameters
        ----------
        n : int
            Index corresponding to input vector.
        theta : float
            Angle of rotation.

        Returns
        -------
        int
            Index corresponding to output vector.

        """
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        return self.k_to_n(np.matmul(R, self.k[n]))

    def plot_r(self, data):
        """Plot data in r space.

        Parameters
        ----------
        data : iterable
            Index corresponds to coordinates.

        """
        # pylint: disable=import-outside-toplevel
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        cmap = mpl.cm.ScalarMappable(
            mpl.colors.Normalize(
                vmin=data.min(), vmax=data.max()),
            mpl.cm.Greys)  # pylint: disable=no-member

        _plot_2d(self.r, self.a1, self.a2, ax, data, cmap)
        fig.colorbar(cmap, ax=ax, shrink=0.6)
        ax.set_xlabel(r'$r_x$')
        ax.set_ylabel(r'$r_y$')

    def plot_k(self, data):
        """Plot data in k space.

        Parameters
        ----------
        data : iterable
            Index corresponds to coordinates.

        """
        # pylint: disable=import-outside-toplevel
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        cmap = mpl.cm.ScalarMappable(
            mpl.colors.Normalize(
                vmin=data.min(), vmax=data.max()),
            mpl.cm.Greys)  # pylint: disable=no-member

        _plot_2d(self.k, self.b1, self.b2, ax, data, cmap)
        fig.colorbar(cmap, ax=ax, shrink=0.6)
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')


def _plot_2d(coords, vec1, vec2, ax, data, cmap):  # pylint: disable=too-many-arguments
    # pylint: disable=import-outside-toplevel
    # pylint: disable=too-many-locals
    import matplotlib as mpl
    from matplotlib.path import Path

    verts0 = _calc_patch(vec1, vec2)
    codes = [Path.MOVETO] + (len(verts0)-2)*[Path.LINETO] \
        + [Path.CLOSEPOLY]

    for coord, dat in zip(coords, data):
        verts = verts0 + coord
        ax.add_patch(mpl.patches.PathPatch(
            Path(verts, codes), facecolor=cmap.to_rgba(dat), lw=0))

    ax.set_aspect('equal')
    xmin = coords[:, 0].min() - abs(vec1[0]) - abs(vec2[0])
    xmax = coords[:, 0].max() + abs(vec1[0]) + abs(vec2[0])
    ymin = coords[:, 1].min() - abs(vec1[1]) - abs(vec2[1])
    ymax = coords[:, 1].max() + abs(vec1[1]) + abs(vec2[1])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


@jit(nopython=True, cache=True)
def _periodic_boundary(r, L1, L2):
    for L in [L2, L1, L2-L1, L2+L1]:
        x = np.dot(r, L) / np.sum(L**2)
        if x > 0.5+1e-08:
            r = _periodic_boundary(r-L, L1, L2)
        if x < -0.5+1e-08:
            r = _periodic_boundary(r+L, L1, L2)
    return r


def _find_cross(x, d, a):
    """Solve x+r*d = a/2 + g*a_perp."""
    mat = np.array([[d[0], d[1]], [-a[1], a[0]]])
    if np.allclose(np.linalg.det(mat), 0):
        return -1, -1
    mat2 = np.linalg.inv(mat)

    r, g = np.matmul(a/2-x, mat2)

    return r, g


def _calc_patch(a1, a2):
    """Calculate the corners of the Wiger-Seitz cell defined by a1, a2."""
    NNs = [d[0]*a1 + d[1]*a2 for d in
           [[1, 0], [0, 1], [-1, 0], [0, -1],
            [1, 1], [1, -1], [-1, -1], [-1, 1]]]

    verts = []
    x = 0
    a0 = NNs.pop(0)
    x += a0/2
    d = np.array([-a0[1], a0[0]])
    while NNs:
        i_min = -1
        r_min = np.inf
        for i, a in enumerate(NNs):
            r, g = _find_cross(x, d, a)
            if 1e-8 < r < r_min and abs(g) > 1e-8:
                i_min = i
                r_min = r
        r, g = _find_cross(x, d, a0)
        if 1e-8 < r < r_min and abs(g) > 1e-8:
            verts.append(x+r*d)
            verts.append(np.array([0, 0]))
            return np.array(verts)
        x += r_min*d
        verts.append(np.copy(x))
        a = NNs.pop(i_min)
        d = np.array([-a[1], a[0]])
    verts.append([0, 0])
    return np.array(verts)


def _init0(L1, L2, a1, a2):
    # pylint: disable=too-many-locals
    if platform.machine() == 'x86_64':
        lattices_bin = 'lattices_x86-64.so'
    elif platform.machine() == 'arm64':
        lattices_bin = 'lattices_armv8.4-a.so'
    else:
        raise NotImplementedError(
            'No lattices binary for this system architecture: '
            f'"{platform.machine()}". Please contact pyALF maintainers.')

    lattices_fortran = CDLL(
        os.path.join(
            os.path.dirname(__file__),
            lattices_bin
        )
    )
    ndim = c_int(len(L1))
    lattices_fortran.make_lattice_c(
        ndim,
        L1.ctypes.data_as(POINTER(c_double)),
        L2.ctypes.data_as(POINTER(c_double)),
        a1.ctypes.data_as(POINTER(c_double)),
        a2.ctypes.data_as(POINTER(c_double))
        )
    ndim = ndim.value

    L = c_int()
    lattices_fortran.get_l(byref(L))
    L = L.value

    N = c_int()
    lattices_fortran.get_n(byref(N))
    N = N.value

    BZ1 = np.empty(ndim, dtype=np.double)
    BZ2 = np.empty(ndim, dtype=np.double)
    b1 = np.empty(ndim, dtype=np.double)
    b2 = np.empty(ndim, dtype=np.double)
    b1_perp = np.empty(ndim, dtype=np.double)
    b2_perp = np.empty(ndim, dtype=np.double)
    listr = np.empty((N, ndim), dtype=np.int32, order='F')
    invlistr = np.empty((2*L+1, 2*L+1), dtype=np.int32, order='F')
    nnlistr = np.empty((N, 3, 3), dtype=np.int32, order='F')
    listk = np.empty((N, ndim), dtype=np.int32, order='F')
    invlistk = np.empty((2*L+1, 2*L+1), dtype=np.int32, order='F')
    nnlistk = np.empty((N, 3, 3), dtype=np.int32, order='F')
    imj = np.empty((N, N), dtype=np.int32, order='F')

    lattices_fortran.get_arrays(
        c_int(ndim),
        c_int(L),
        c_int(N),
        BZ1.ctypes.data_as(POINTER(c_int)),
        BZ2.ctypes.data_as(POINTER(c_int)),
        b1.ctypes.data_as(POINTER(c_int)),
        b2.ctypes.data_as(POINTER(c_int)),
        b1_perp.ctypes.data_as(POINTER(c_int)),
        b2_perp.ctypes.data_as(POINTER(c_int)),
        listr.ctypes.data_as(POINTER(c_int)),
        invlistr.ctypes.data_as(POINTER(c_int)),
        nnlistr.ctypes.data_as(POINTER(c_int)),
        listk.ctypes.data_as(POINTER(c_int)),
        invlistk.ctypes.data_as(POINTER(c_int)),
        nnlistk.ctypes.data_as(POINTER(c_int)),
        imj.ctypes.data_as(POINTER(c_int)),
    )

    lattices_fortran.clear_lattice_c()

    return(BZ1, BZ2, b1, b2,
           b1_perp, b2_perp, L, N,
           listr, invlistr, nnlistr,
           listk, invlistk, nnlistk,
           imj)


@jit(nopython=True, cache=True)
def _init1(L1, L2, a1, a2):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    ndim = len(L1)

    # Compute the Reciprocal Lattice vectors.
    mat = np.array([[a1[0], a1[1]], [a2[0], a2[1]]])
    mat2 = 2. * np.pi * np.linalg.inv(mat)
    BZ1 = np.copy(mat2[:,0])
    BZ2 = np.copy(mat2[:,1])

    # K-space Quantization  from periodicity in L1_p and L2_p
    X = 2. * np.pi / (np.dot(BZ1, L1) * np.dot(BZ2, L2)
                      - np.dot(BZ2, L1) * np.dot(BZ1, L2))
    b1 = X * (np.dot(BZ2, L2)*BZ1 - np.dot(BZ1, L2)*BZ2)
    b2 = X * (np.dot(BZ1, L1)*BZ2 - np.dot(BZ2, L1)*BZ1)

    # Compute b1_perp, b2_perp
    mat = np.array([[b1[0], b1[1]], [b2[0], b2[1]]])
    mat2 = np.linalg.inv(mat)
    b1_perp = mat2[:,0]
    b2_perp = mat2[:,1]

    # Count the number of Lattice points and setup list, invlist
    N1 = abs(int(round(np.dot(BZ1, L1) / (2.*np.pi))))
    N2 = abs(int(round(np.dot(BZ2, L1) / (2.*np.pi))))
    N3 = abs(int(round(np.dot(BZ1, L2) / (2.*np.pi))))
    N4 = abs(int(round(np.dot(BZ2, L2) / (2.*np.pi))))
    L = np.array([N1, N2, N3, N4]).max()

    invlistr = np.full((2*L+1, 2*L+1), fill_value=-1, dtype=np.int32)
    N = 0
    for i1 in range(-L, L+1):
        for i2 in range(-L, L+1):
            x = i1*a1 + i2*a2
            in_latt = True
            for a in [L2, L1, L2-L1, L2+L1]:
                if((np.dot(x, a) > np.dot(a, a)/2. + 1e-5)
                   or (np.dot(x, a) < -np.dot(a, a)/2. + 1e-5)):
                    in_latt = False
            if in_latt:
                invlistr[i1, i2] = N
                N += 1

    listr = np.empty((N, ndim), dtype=np.int32)
    nc = 0
    for i1 in range(-L, L+1):
        for i2 in range(-L, L+1):
            x = i1*a1 + i2*a2
            in_latt = True
            for a in [L2, L1, L2-L1, L2+L1]:
                if((np.dot(x, a) > np.dot(a, a)/2. + 1e-5)
                   or (np.dot(x, a) < -np.dot(a, a)/2. + 1e-5)):
                    in_latt = False
            if in_latt:
                listr[nc] = [i1, i2]
                nc += 1
    if nc != N:
        print(L, nc, N)
        raise RuntimeError('Error in initialsation of Lattice')

    nc = 0
    listk = np.empty((N, ndim), dtype=np.int32)
    invlistk = np.full((2*L+1, 2*L+1), fill_value=-1, dtype=np.int32)
    for i1 in range(-L, L+1):
        for i2 in range(-L, L+1):
            x = i1*b1 + i2*b2
            in_latt = True
            for b in [BZ2, BZ1, BZ2-BZ1, BZ2+BZ1]:
                if((np.dot(x, b) > (np.dot(b, b)/2. + 1e-7))
                   or (np.dot(x, b) < (-np.dot(b, b)/2. + 1e-7))):
                    in_latt = False
            if in_latt:
                listk[nc] = [i1, i2]
                invlistk[i1, i2] = nc
                nc += 1
    if nc != N:
        print(L, nc, N)
        raise RuntimeError('Error in initialsation of Lattice')

    # Setup lists of nearest neighbors
    nnlistr = np.zeros((N, 3, 3), dtype=np.int32)
    nnlistk = np.zeros((N, 3, 3), dtype=np.int32)
    for n in range(N):
        for nd1 in [-1, 0, 1]:
            for nd2 in [-1, 0, 1]:
                d = nd1*a1 + nd2*a2
                x = listr[n, 0]*a1 + listr[n, 1]*a2 + d
                x = _periodic_boundary(x, L1, L2)
                n1 = int(round(np.dot(BZ1, x) / (2.*np.pi)))
                n2 = int(round(np.dot(BZ2, x) / (2.*np.pi)))
                nn = invlistr[(n1, n2)]
                nnlistr[n, nd1, nd2] = nn
                # if not np.allclose(x, listr[nn, 0]*a1 + listr[nn, 1]*a2,
                #                    rtol=1e-05, atol=1e-08, equal_nan=False):
                #     raise RuntimeError(
                #       'Error in initialsation of Lattice, setting of nnlistr'
                #     )

                d = nd1*b1 + nd2*b2
                x = listk[n, 0]*b1 + listk[n, 1]*b2 + d
                x = _periodic_boundary(x, BZ1, BZ2)
                n1 = int(round(np.dot(x, b1_perp)))
                n2 = int(round(np.dot(x, b2_perp)))
                nn = invlistk[(n1, n2)]
                nnlistk[n, nd1, nd2] = nn
                # if not np.allclose(x, listk[nn, 0]*b1 + listk[nn, 1]*b2,
                #                    rtol=1e-05, atol=1e-08, equal_nan=False):
                #     print(x, listk[nn, 0]*b1 + listk[nn, 1]*b2)
                #     raise RuntimeError(
                #       'Error in initialsation of Lattice, setting of nnlistk'
                #     )

    # setup imj
    imj = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        x_i = listr[i, 0]*a1 + listr[i, 1]*a2
        for j in range(N):
            x_j = listr[j, 0]*a1 + listr[j, 1]*a2
            d = _periodic_boundary(x_i - x_j, L1, L2)
            n1 = int(round(np.dot(BZ1, d) / (2.*np.pi)))
            n2 = int(round(np.dot(BZ2, d) / (2.*np.pi)))
            imj_temp = invlistr[n1, n2]
            imj[i, j] = imj_temp

    return(BZ1, BZ2, b1, b2,
           b1_perp, b2_perp, L, N,
           listr, invlistr, nnlistr,
           listk, invlistk, nnlistk,
           imj)
