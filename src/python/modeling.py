"""
Modeling tools for PWL convex optimization.

Routines for specifying and solving convex optimization problems with
piecewise-linear objective and constraint functions.
"""

# Copyright 2012-2023 M. Andersen and L. Vandenberghe.
# Copyright 2010-2011 L. Vandenberghe.
# Copyright 2004-2009 J. Dahl and L. Vandenberghe.
# 
# This file is part of CVXOPT.
#
# CVXOPT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# CVXOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from cvxopt.base import matrix, spmatrix
from cvxopt import blas, solvers 
import sys
if sys.version_info[0] < 3: 
    import __builtin__ as builtins
else:
    import builtins

__all__ = ["variable", "constraint", "op", "min", "max", "sum", "dot"]
 
class variable(object):

    """
    Vector valued optimization variable.

    variable(size=1, name='') creates a variable of length size.


    Arguments:

    size      length of the variable (positive integer)
    name      name of the variable (string)


    Attributes:

    name      the name of the variable
    value     None or a 'd' matrix of size (len(self),1)
    _size     the length of the variable  
    """

    def __init__(self, size=1, name=''):

        self.name = name   
        self.value = None
        if type(size) is int and size > 0: 
            self._size = size
        else: 
            raise TypeError("size must be a positive integer")


    def __len__(self):
         
        return self._size


    def __repr__(self):

        return "variable(%d,'%s')" %(len(self),self.name)  


    def __str__(self):

        s = self.name
        if self.name:  s += ': '
        s += 'variable of length %d\nvalue: ' %len(self)
        if self.value is None: 
            s += 'None'
        else: 
            s += '\n' + str(self.value)
        return s


    def __setattr__(self,name,value):

        if name == 'value':
            if value is None or (_isdmatrix(value) and 
                value.size == (len(self),1)):
                object.__setattr__(self, name, value)

            elif type(value) is int or type(value) is float:
                object.__setattr__(self, name, 
                    matrix(value, (len(self),1), tc='d'))

            else:
                raise AttributeError("invalid type or size for "\
                    "attribute 'value'")

        elif name == 'name':
            if type(value) is str:
                object.__setattr__(self,name,value)

            else:
                raise AttributeError("invalid type for attribute "\
                    "'name'")

        elif name == '_size':
            object.__setattr__(self,name,value)

        else:
            raise AttributeError("'variable' object has no attribute "\
                "'%s'" %name)


    def __pos__(self):

        f = _function()
        f._linear._coeff[self] = matrix(1.0)
        return f


    def __neg__(self):

        f = _function()
        f._linear._coeff[self] = matrix(-1.0)
        return f


    def __abs__(self):

        return max(self,-self)
        

    def __add__(self,other):

        return (+self).__add__(other)


    def __radd__(self,other):

        return (+self).__radd__(other)


    def __iadd__(self,other):

        raise NotImplementedError("in-place addition not implemented"\
            " for 'variable' objects")


    def __sub__(self,other):

        return (+self).__sub__(other)


    def __rsub__(self,other):

        return (+self).__rsub__(other)


    def __isub__(self,other):

        raise NotImplementedError("in-place subtraction not "\
            "implemented for 'variable' objects")


    def __mul__(self,other):

        return (+self).__mul__(other)


    def __rmul__(self,other):

        return (+self).__rmul__(other)


    def __imul__(self,other):

        raise NotImplementedError("in-place multiplication not "\
            "implemented for 'variable' objects")


    if sys.version_info[0] < 3: 

        def __div__(self,other):

            return (+self).__div__(other)


        def __idiv__(self,other):

            raise NotImplementedError("in-place division not implemented "\
                "for 'variable' objects")

    else:
    
        def __truediv__(self,other):

            return (+self).__truediv__(other)


        def __itruediv__(self,other):

            raise NotImplementedError("in-place division not implemented "\
                "for 'variable' objects")


    def __eq__(self,other):

        return constraint(self-other, '=')


    def __le__(self,other):

        return constraint(self-other, '<')


    def __ge__(self,other):

        return constraint(other-self, '<')


    def __lt__(self,other):

        raise NotImplementedError


    def __gt__(self,other):

        raise NotImplementedError


    def __getitem__(self,key):

        return (+self).__getitem__(key)


    if sys.version_info[0] >= 3: 

        def __hash__(self):

            return id(self)


class _function(object):

    """
    Vector valued function.

    General form: 

        f = constant + linear + sum of nonlinear convex terms + 
            sum of nonlinear concave terms 

    The length of f is the maximum of the lengths of the terms in the 
    sum.  Each term must have length 1 or length equal to len(f).

    _function() creates the constant function f=0 with length 1.


    Attributes:

    _constant      constant term as a 1-column dense 'd' matrix of 
                   length 1 or length len(self)
    _linear        linear term as a _lin  object of length 1 or length
                   len(self)
    _cvxterms      nonlinear convex terms as a list [f1,f2,...] with 
                   each fi of type _minmax or _sum_minmax.  Each fi has
                   length 1 or length equal to len(self).
    _ccvterms      nonlinear concave terms as a list [f1,f2,...] with 
                   each fi of type _minmax or _sum_minmax.  Each fi has
                   length 1 or length equal to len(self).


    Methods:

    value()        returns the value of the function: None if one of the
                   variables has value None;  a dense 'd' matrix of size
                   (len(self),1) if all the variables have values
    variables()    returns a (copy of) the list of variables
    _iszero()      True if self is identically zero
    _isconstant()  True if there are no linear/convex/concave terms
    _islinear()    True if there are no constant/convex/concave terms
    _isaffine()    True if there are no nonlinear convex/concave terms 
    _isconvex()    True if there are no nonlinear concave terms
    _isconcave()   True if there are no nonlinear convex terms
    """

    def __init__(self): 

        self._constant = matrix(0.0)
        self._linear = _lin()
        self._cvxterms = []
        self._ccvterms = []


    def __len__(self):

        if len(self._constant) > 1: return len(self._constant)

        lg = len(self._linear)
        if lg > 1: return lg

        for f in self._cvxterms:
            lg = len(f)
            if lg > 1: return lg

        for f in self._ccvterms:
            lg = len(f)
            if lg > 1: return lg

        return 1


    def __repr__(self):

        if self._iszero():
            return '<zero function of length %d>' %len(self)

        elif self._isconstant():
            return '<constant function of length %d>' %len(self)

        elif self._islinear():
            return '<linear function of length %d>' %len(self)

        elif self._isaffine():
            return '<affine function of length %d>' %len(self)

        elif self._isconvex():
            return '<convex function of length %d>' %len(self)

        elif self._isconcave():
            return '<concave function of length %d>' %len(self)

        else:
            return '<function of length %d>' %len(self)


    def __str__(self):

        s = repr(self)[1:-1] 

        # print constant term if nonzero
        if not self._iszero() and (len(self._constant) != 1 or 
            self._constant[0]):
            s += '\nconstant term:\n' + str(self._constant)

        else:
            s += '\n'

        # print linear term if nonzero
        if self._linear._coeff:
            s += 'linear term: ' + str(self._linear)

        # print nonlinear convex term if nonzero
        if self._cvxterms: 
            s += '%d nonlinear convex term(s):' %len(self._cvxterms)
            for f in self._cvxterms: s += '\n' + str(f) 

        # print nonlinear concave term if nonzero
        if self._ccvterms: 
            s += '%d nonlinear concave term(s):' %len(self._ccvterms)
            for f in self._ccvterms: s += '\n' + str(f) 

        return s

    
    def value(self):

        val = self._constant

        if self._linear._coeff:
            nval = self._linear.value()     
            if nval is None: return None
            else: val = val + nval 

        for f in self._cvxterms:
            nval = f.value()
            if nval is None: return None
            else: val = val + nval

        for f in self._ccvterms:
            nval = f.value()
            if nval is None: return None
            else: val = val + nval

        return val


    def variables(self):

        l = self._linear.variables()
        for f in self._cvxterms:
            l += [v for v in f.variables() if v not in l]
        for f in self._ccvterms:
            l += [v for v in f.variables() if v not in l]
        return l


    def _iszero(self):

        if not self._linear._coeff and not self._cvxterms and \
            not self._ccvterms and not blas.nrm2(self._constant): 
            return True
        else: return False


    def _isconstant(self):

        if not self._linear._coeff and not self._cvxterms and \
            not self._ccvterms: return True
        else: return False


    def _islinear(self):

        if len(self._constant) == 1 and not self._constant[0] and \
            not self._cvxterms and not self._ccvterms: return True
        else: return False


    def _isaffine(self):

        if not self._cvxterms and not self._ccvterms: return True
        else: return False


    def _isconvex(self):

        if not self._ccvterms: return True
        else: return False


    def _isconcave(self):

        if not self._cvxterms: return True
        else: return False


    def __pos__(self):

        f = _function()
        f._constant = +self._constant
        f._linear = +self._linear
        f._cvxterms = [+g for g in self._cvxterms]
        f._ccvterms = [+g for g in self._ccvterms]
        return f


    def __neg__(self):

        f = _function()
        f._constant = -self._constant
        f._linear = -self._linear
        f._ccvterms = [-g for g in self._cvxterms]
        f._cvxterms = [-g for g in self._ccvterms]
        return f


    def __add__(self,other):

        # convert other to matrix (dense 'd' or sparse) or _function
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):  # ie, dense 'd' or sparse
            if other.size[1] != 1:
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented

        if 1 != len(self) != len(other) != 1: 
            raise ValueError('incompatible lengths')

        f = _function()

        if _ismatrix(other):
            # this converts sparse other to dense 'd' 
            f._constant = self._constant + other
            f._linear = +self._linear 
            f._cvxterms = [+fk for fk in self._cvxterms] 
            f._ccvterms = [+fk for fk in self._ccvterms] 

        else:  #type(other) is _function:
            if not (self._isconvex() and other._isconvex()) and \
                not (self._isconcave() and other._isconcave()):
                raise ValueError('operands must be both convex or '\
                    'both concave')

            f._constant = self._constant + other._constant
            f._linear = self._linear + other._linear
            f._cvxterms = [+fk for fk in self._cvxterms] + \
                [+fk for fk in other._cvxterms]
            f._ccvterms = [+fk for fk in self._ccvterms] + \
                [+fk for fk in other._ccvterms]

        return f


    def __radd__(self,other):

        return self.__add__(other)


    def __iadd__(self,other):

        # convert other to matrix (dense 'd' or sparse) or _function
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):
            if other.size[1] != 1: 
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented

        if len(self) != len(other) != 1: 
            raise ValueError('incompatible lengths')

        if _ismatrix(other):
            if 1 == len(self._constant) != len(other): 
                self._constant = self._constant + other
            else:
                self._constant += other

        else:   #type(other) is _function:
            if not (self._isconvex() and other._isconvex()) and \
                not (self._isconcave() and other._isconcave()):
                raise ValueError('operands must be both convex or '\
                    'both concave')

            if 1 == len(self._constant) != len(other._constant): 
                self._constant = self._constant + other._constant
            else:
                self._constant += other._constant

            if 1 == len(self._linear) != len(other._linear): 
                self._linear = self._linear + other._linear
            else:
                self._linear += other._linear

            self._cvxterms += [+fk for fk in other._cvxterms] 
            self._ccvterms += [+fk for fk in other._ccvterms] 

        return self


    def __sub__(self,other):

        # convert other to matrix (dense 'd' or sparse) or _function
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):
            if other.size[1] != 1: 
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented

        if 1 != len(self) != len(other) != 1: 
            raise ValueError('incompatible lengths')

        f = _function()

        if _ismatrix(other):
            f._constant = self._constant - other
            f._linear = +self._linear 
            f._cvxterms = [+fk for fk in self._cvxterms]  
            f._ccvterms = [+fk for fk in self._ccvterms] 

        else:   #type(other) is _function:
            if not (self._isconvex() and other._isconcave()) and \
                not (self._isconcave() and other._isconvex()):
                raise ValueError('operands must be convex and '\
                    'concave or concave and convex')

            f._constant = self._constant - other._constant
            f._linear = self._linear - other._linear
            f._cvxterms = [+fk for fk in self._cvxterms] + \
                [-fk for fk in other._ccvterms]
            f._ccvterms = [+fk for fk in self._ccvterms] + \
                [-fk for fk in other._cvxterms]

        return f


    def __rsub__(self,other):

        # convert other to matrix (dense 'd' or sparse) or _function
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _isdmatrix(other):
            if other.size[1] != 1: 
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented

        if 1 != len(self) != len(other) != 1: 
            raise ValueError('incompatible lengths')

        f = _function()

        if _ismatrix(other):
            f._constant = other - self._constant 
            f._linear = -self._linear 
            f._cvxterms = [-fk for fk in self._ccvterms] 
            f._ccvterms = [-fk for fk in self._cvxterms] 

        else:   # type(other) is _function:
            if not (self._isconvex() and other._isconcave()) and \
                not (self._isconcave() and other._isconvex()):
                raise ValueError('operands must be convex and '\
                    'concave or concave and convex')

            f._constant = other._constant - self._constant 
            f._linear = other._linear - self._linear 
            f._cvxterms = [-fk for fk in self._ccvterms] + \
                [fk for fk in other._cvxterms]
            f._ccvterms = [-fk for fk in self._cvxterms] + \
                [fk for fk in other._ccvterms]

        return f


    def __isub__(self,other):

        # convert other to matrix (dense or sparse 'd') or _function
        if type(other) is int or type(other) is float:
            other = matrix(other, tc='d')
        elif _ismatrix(other):
            if other.size[1] != 1: 
                raise ValueError('incompatible dimensions')
        elif type(other) is variable:
            other = +other
        elif type(other) is not _function:
            return NotImplemented

        if len(self) != len(other) != 1: 
            raise ValueError('incompatible lengths')

        if _ismatrix(other):
            if 1 == len(self._constant) != len(other): 
                self._constant = self._constant - other
            else:
                self._constant -= other

        else:   #type(other) is _function:
            if not (self._isconvex() and other._isconcave()) and \
                not (self._isconcave() and other._isconvex()):
                raise ValueError('operands must be convex and '\
                    'concave or concave and convex')

            if 1 == len(self._constant) != len(other._constant): 
                self._constant = self._constant - other._constant
            else:
                self._constant -= other._constant

            if 1 == len(self._linear) != len(other._linear): 
                self._linear = self._linear - other._linear
            else:
                self._linear -= other._linear

            self._cvxterms += [-fk for fk in other._ccvterms] 
            self._ccvterms += [-fk for fk in other._cvxterms] 

        return self


    def __mul__(self,other):

        if type(other) is int or type(other) is float: 
            other = matrix(other, tc='d')

        if (_ismatrix(other) and other.size == (1,1)) or \
            (_isdmatrix(other) and other.size[1] == 1 == len(self)):

            f = _function()

            if other.size == (1,1) and other[0] == 0.0:
                # if other is zero, return constant zero function
                # of length len(self)
                f._constant = matrix(0.0, (len(self),1))
                return f

            if len(self._constant) != 1 or self._constant[0]:
                # skip if self._constant is zero
                f._constant = self._constant*other

            if self._linear._coeff: 
                # skip if self._linear is zero
                f._linear = self._linear*other

            if not self._isaffine():  
                # allow only multiplication with scalar
                if other.size == (1,1):
                    if other[0] > 0.0:
                        f._cvxterms = \
                            [fk*other[0] for fk in self._cvxterms]
                        f._ccvterms = \
                            [fk*other[0] for fk in self._ccvterms]

                    elif other[0] < 0.0: 
                        f._cvxterms = \
                            [fk*other[0] for fk in self._ccvterms]
                        f._ccvterms = \
                            [fk*other[0] for fk in self._cvxterms]

                    else: # already dealt with above
                        pass

                else: 
                    raise ValueError('can only multiply with scalar')

        else: 
            raise TypeError('incompatible dimensions or types')

        return f


    def __rmul__(self,other):

        if type(other) is int or type(other) is float: 
            other = matrix(other, tc='d')

        lg = len(self)

        if (_ismatrix(other) and other.size[1] == lg) or \
            (_isdmatrix(other) and other.size == (1,1)):

            f = _function()

            if other.size == (1,1) and other[0] == 0.0:
                # if other is zero, return constant zero function
                # of length len(self)
                f._constant = matrix(0.0, (len(self),1))
                return f

            if len(self._constant) != 1 or self._constant[0]:
                if 1 == len(self._constant) != lg and \
                    not _isscalar(other):
                    f._constant = other * self._constant[lg*[0]]
                else:
                    f._constant = other * self._constant
            
            if self._linear._coeff:
                if 1 == len(self._linear) != lg and \
                    not _isscalar(other):
                    f._linear = other * self._linear[lg*[0]]
                else:
                    f._linear = other * self._linear

            if not self._isaffine():
                # allow only scalar multiplication
                if other.size == (1,1):
                    if other[0] > 0.0:
                        f._cvxterms = [other[0] * fk for 
                            fk in self._cvxterms]  
                        f._ccvterms = [other[0] * fk for 
                            fk in self._ccvterms]  

                    elif other[0] < 0.0:
                        f._cvxterms = [other[0] * fk for 
                            fk in self._ccvterms]  
                        f._ccvterms = [other[0] * fk for 
                            fk in self._cvxterms]  

                    else: pass

                else: 
                    raise ValueError('can only multiply with scalar')

        else: 
            raise TypeError('incompatible dimensions or types')

        return f
                

    def __imul__(self,other):

        if type(other) is int or type(other) is float: 
            other = matrix(other, tc='d')

        if _isdmatrix(other) and other.size == (1,1):

            if other[0] == 0.0: 
                self._constant = matrix(0.0, (len(self),1))
                return self

            if len(self._constant) != 1 or self._constant[0]:
                self._constant *= other[0]

            if self._linear._coeff:  self._linear *= other[0]

            if not self._isaffine():
                if other[0] > 0.0:
                    for f in self._cvxterms: f *= other[0]
                    for f in self._ccvterms: f *= other[0]

                elif other[0] < 0.0:
                    cvxterms = [f*other[0] for f in self._ccvterms]
                    self._ccvterms = [f*other[0] for f in 
                        self._cvxterms]
                    self._cvxterms = cvxterms

                else:
                    pass

            return self

        else: 
            raise TypeError('incompatible dimensions or types')

    if sys.version_info[0] < 3: 

        def __div__(self,other):

            if type(other) is int or type(other) is float:
                return self.__mul__(1.0/other)

            elif _isdmatrix(other) and other.size == (1,1):
                return self.__mul__(1.0/other[0])

            else:
                return NotImplemented

    else:

        def __truediv__(self,other):

            if type(other) is int or type(other) is float:
                return self.__mul__(1.0/other)

            elif _isdmatrix(other) and other.size == (1,1):
                return self.__mul__(1.0/other[0])

            else:
                return NotImplemented



    def __rdiv__(self,other):

        return NotImplemented


    if sys.version_info[0] < 3: 

        def __idiv__(self,other):

            if type(other) is int or type(other) is float:
                return self.__imul__(1.0/other)

            elif _isdmatrix(other) and other.size == (1,1):
                return self.__imul__(1.0/other[0])

            else:
                return NotImplemented

    else:

        def __itruediv__(self,other):

            if type(other) is int or type(other) is float:
                return self.__imul__(1.0/other)

            elif _isdmatrix(other) and other.size == (1,1):
                return self.__imul__(1.0/other[0])

            else:
                return NotImplemented
        

    def __abs__(self):

        return max(self,-self)


    def __eq__(self,other):

        return constraint(self-other, '=')


    def __le__(self,other):

        return constraint(self-other, '<')


    def __ge__(self,other):

        return constraint(other-self, '<')


    def __lt__(self,other):

        raise NotImplementedError


    def __gt__(self,other):

        raise NotImplementedError


    def __getitem__(self,key):

        lg = len(self)
        l = _keytolist(key,lg)
        if not l: raise ValueError('empty index set')

        f = _function()

        if len(self._constant) != 1 or self._constant[0]:
            if 1 == len(self._constant) != lg: 
                f._constant = +self._constant
            else: 
                f._constant = self._constant[l]

        if self._linear:
            if 1 == len(self._linear) != lg: 
                f._linear = +self._linear
            else: 
                f._linear = self._linear[l]

        for fk in self._cvxterms:
            if 1 == len(fk) != lg: 
                f._cvxterms += [+fk]

            elif type(fk) is _minmax: 
                f._cvxterms += [fk[l]]

            else:  # type(fk) is _sum_minmax 
                # fk is defined as fk = sum(fmax)
                fmax = _minmax('max', *fk._flist)
                # take f += sum_j fk[j][l] = sum_{gk in fmax} gk[l] 
                f._cvxterms += [gk[l] for gk in fmax]

        for fk in self._ccvterms:
            if 1 == len(fk) != lg: 
                f._ccvterms += [+fk]

            elif type(fk) is _minmax:
                f._ccvterms += [fk[l]]

            else:  # type(fk) is _sum_minmax
                # fk is defined as fk = sum(fmin)
                fmin = _minmax('min',*fk._flist)
                # take f += sum_j fk[j][l] = sum_{gk in fmin} gk[l] 
                f._ccvterms += [gk[l] for gk in fmin]

        return f



def sum(s):

    """
    Built-in sum redefined to improve efficiency when s is a vector
    function.  
    
    If type(s) is not _function object, this is the built-in sum 
    without start argument.
    """

    if type(s) is _function:
        lg = len(s)
        f = _function()

        if 1 == len(s._constant) != lg:
            f._constant = lg * s._constant
        else:
            f._constant = matrix(sum(s._constant), tc='d')

        if 1 == len(s._linear) != lg:
            f._linear = lg * s._linear
        else:
            f._linear = sum(s._linear)

        for c in s._cvxterms:
            if len(c) == 1:
                f._cvxterms += [lg*c]
            else:  #type(c) must be _minmax if len(c) > 1
                f._cvxterms += [_sum_minmax('max', *c._flist)]

        for c in s._ccvterms:
            if len(c) == 1:
                f._ccvterms += [lg*c]
            else:  #type(c) must be _minmax if len(c) > 1
                f._ccvterms += [_sum_minmax('min', *c._flist)]

        return f

    else:
        return builtins.sum(s)



class _lin(object):

    """
    Vector valued linear function.


    Attributes:

    _coeff       dictionary {variable: coefficient}.  The coefficients
                 are dense or sparse matrices.  Scalar coefficients are
                 stored as 1x1 'd' matrices.


    Methods:

    value()        returns the value of the function: None if one of the
                   variables has value None;  a dense 'd' matrix of size
                   (len(self),1) if all the variables have values
    variables()    returns a (copy of) the list of variables
    _addterm()     adds a linear term  a*v
    _mul()         in-place multiplication
    _rmul()        in-place right multiplication
    """


    def __init__(self):
     
        self._coeff = {}


    def __len__(self):
    
        for v,c in iter(self._coeff.items()):
            if c.size[0] > 1: 
                return c.size[0]
            elif _isscalar(c) and len(v) > 1: 
                return len(v)
        return 1


    def __repr__(self):

        return '<linear function of length %d>' %len(self)


    def __str__(self):

        s = repr(self)[1:-1] + '\n'
        for v,c in iter(self._coeff.items()): 
            s += 'coefficient of ' + repr(v) + ':\n'  + str(c) 
        return s


    def value(self):

        value = matrix(0.0, (len(self),1))
        for v,c in iter(self._coeff.items()):
            if v.value is None: 
                return None
            else: 
                value += c*v.value   
        return value


    def variables(self):

        return varlist(self._coeff.keys())
   

    def _addterm(self,a,v):   

        """ 
        self += a*v  with v variable and a int, float, 1x1 dense 'd' 
        matrix, or sparse or dense 'd' matrix with len(v) columns.
        """

        lg = len(self)

        if v in self._coeff:

            # self := self + a*v with v a variable of self
            #
            # Valid types/sizes:
            #
            # 1. a is a matrix (sparse or dense) with a.size[0]>1,
            #    a.size[1]=len(v), and either lg=1 or lg=a.size[0].
            #
            # 2. a is a matrix (sparse or dense), a.size = (1,len(v)),
            #    lg arbitrary.
            #
            # 3. a is int or float or 1x1 dense matrix, and len(v)>1
            #    and either lg=1 or lg=len(v)
            #
            # 4. a is int or float or 1x1 dense matrix, and len(v)=1

            c = self._coeff[v]
            if _ismatrix(a) and a.size[0] > 1 and a.size[1] == len(v)\
                and (lg == 1 or lg == a.size[0]):
                newlg = a.size[0]
                if c.size == a.size:
                    self._coeff[v] = c + a 
                elif c.size == (1,len(v)):
                    self._coeff[v] = c[newlg*[0],:] + a
                elif _isdmatrix(c) and c.size == (1,1):
                    m = +a
                    m[::newlg+1] += c[0]
                    self._coeff[v] = m
                else:
                    raise TypeError('incompatible dimensions')
                    
            elif _ismatrix(a) and a.size == (1,len(v)):
                if c.size == (lg,len(v)):
                    self._coeff[v] = c + a[lg*[0],:]
                elif c.size == (1,len(v)):
                    self._coeff[v] = c + a
                elif _isdmatrix(c) and c.size == (1,1):
                    m = a[lg*[0],:]
                    m[::lg+1] += c[0]
                    self._coeff[v] = m
                else:
                    raise TypeError('incompatible dimensions')

            elif _isscalar(a) and len(v) > 1 and (lg == 1 or 
                lg == len(v)):
                newlg = len(v)
                if c.size == (newlg,len(v)):
                    self._coeff[v][::newlg+1] = c[::newlg+1] + a
                elif c.size == (1,len(v)):
                    self._coeff[v] = c[newlg*[0],:]
                    self._coeff[v][::newlg+1] = c[::newlg+1] + a 
                elif _isscalar(c):
                    self._coeff[v] = c + a
                else:
                    raise TypeError('incompatible dimensions')

            elif _isscalar(a) and len(v) == 1:
                self._coeff[v] = c + a    # add a to every elt of c

            else:
                raise TypeError('coefficient has invalid type or '\
                    'incompatible dimensions ')

        elif type(v) is variable:

            # self := self + a*v with v not a variable of self
            #
            # 1. if a is a scalar and len(v)=lg or lg=1 or len(v)=1:
            #    convert a to dense 1x1 matrix and add v:a pair to 
            #    dictionary
            #
            # 2. If a is a matrix (dense or sparse) and a.size[1]=len(v)
            #    and a.size[0]=lg or lg=1 or a.size[0]=1: 
            #    make a copy of a and add v:a pair to dictionary

            if _isscalar(a) and (lg == 1 or len(v) == 1 or 
                len(v) == lg):
                self._coeff[v] = matrix(a, tc='d')

            elif _ismatrix(a) and a.size[1] == len(v) and \
                (lg == 1 or a.size[0] == 1 or a.size[0] == lg):
                self._coeff[v] = +a

            else:
                raise TypeError('coefficient has invalid type or '\
                    'incompatible dimensions ')
        
        else: 
            raise TypeError('second argument must be a variable')


    def _mul(self,a):

        ''' 
        self := self*a where a is scalar or matrix 
        '''

        if type(a) is int or type(a) is float:
            for v in iter(self._coeff.keys()): self._coeff[v] *= a

        elif _ismatrix(a) and a.size == (1,1):
            for v in iter(self._coeff.keys()): self._coeff[v] *= a[0]
        
        elif len(self) == 1 and _isdmatrix(a) and a.size[1] == 1:
            for v,c in iter(self._coeff.items()): self._coeff[v] = a*c

        else: 
            raise TypeError('incompatible dimensions')


    def _rmul(self,a):

        ''' 
        self := a*self where a is scalar or matrix 
        '''

        lg = len(self)
        if _isscalar(a):
            for v in iter(self._coeff.keys()): self._coeff[v] *= a
        
        elif lg == 1 and _ismatrix(a) and a.size[1] == 1:
            for v,c in iter(self._coeff.items()): self._coeff[v] = a*c

        elif _ismatrix(a) and a.size[1] == lg: 
            for v,c in iter(self._coeff.items()):
                if c.size == (1,len(v)):
                    self._coeff[v] = a*c[lg*[0],:]
                else: 
                    self._coeff[v] = a*c
             
        else: 
            raise TypeError('incompatible dimensions')


    def __pos__(self):

        f = _lin()
        for v,c in iter(self._coeff.items()): f._coeff[v] = +c
        return f


    def __neg__(self):

        f = _lin()
        for v,c in iter(self._coeff.items()): f._coeff[v] = -c
        return f


    def __add__(self,other):

        # self + other with other variable or _lin

        f = +self

        if type(other) is int or type(other) is float and not other:
            # Needed to make sum(f) work, because it defaults to
            # 0 + f[0] + ... + f[len(f)-1].
            return f

        if type(other) is variable: 
            f._addterm(1.0, other)

        elif type(other) is _lin:
            for v,c in iter(other._coeff.items()): f._addterm(c,v)

        else: return NotImplemented

        return f


    def __radd__(self,other):

        return self.__add__(other)

    
    def __iadd__(self,other):

        '''
        self += other  
        
        Only allowed if it does not change the length of self.
        '''

        lg = len(self)

        if type(other) is variable and (len(other) == 1 or 
            len(other) == lg):
            self._addterm(1.0,other)

        elif type(other) is _lin and (len(other) == 1 or 
            len(other) == lg): 
            for v,c in iter(other._coeff.items()): self._addterm(c,v)

        else: 
            raise NotImplementedError('in-place addition must result '\
                'in a function of the same length')
        
        return self


    def __sub__(self,other):

        f = +self

        if type(other) is variable:
            f._addterm(-1.0, other)
        elif type(other) is _lin:
            for v,c in iter(other._coeff.items()): f._addterm(-c,v)
        else: 
            return NotImplemented
        
        return f


    def __rsub__(self,other):

        f = -self

        if type(other) is variable:
            f._addterm(1.0, other)
        elif type(other) is _lin:
            for v,c in iter(other._coeff.items()): f._addterm(c,v)
        else: 
            return NotImplemented
        
        return f


    def __isub__(self,other):

        '''
        self -= other  
        
        Only allowed if it does not change the length of self.
        '''

        lg = len(self)

        if type(other) is variable and (len(other) == 1 or 
            len(other) == lg):
            self._addterm(-1.0, other)

        elif type(other) is _lin and (len(other) == 1 or 
            len(other) == lg):
            for v,c in iter(other._coeff.items()): self._addterm(-c,v)

        else: 
            raise NotImplementedError('in-place subtraction must '\
                'result in a function of the same length')
        
        return self


    def __mul__(self,other):

        if _isscalar(other) or _ismatrix(other):
            f = +self
            f._mul(other)

        else: 
            return NotImplemented

        return f


    def __rmul__(self,other):

        if _isscalar(other) or _ismatrix(other):
            f = +self
            f._rmul(other)

        else:
            return NotImplemented

        return f
        

    def __imul__(self,other):    

        '''
        self *= other  
        
        Only allowed for scalar multiplication with a constant (int, 
        float, 1x1 'd' matrix).
        '''

        if _isscalar(other): 
            self._mul(other)
        else: 
            raise NotImplementedError('in-place multiplication '  \
                'only defined for scalar multiplication')
        return self


    def __getitem__(self,key):

        l = _keytolist(key,len(self))
        if not l: raise ValueError('empty index set')

        f = _lin()
        for v,c in iter(self._coeff.items()):
            if c.size == (len(self), len(v)):  
                f._coeff[v] = c[l,:]

            elif _isscalar(c) and len(v) == 1:  
                f._coeff[v] = matrix(c, tc='d')

            elif c.size == (1,1) and len(v) > 1:
                # create a sparse matrix with c[0] element in 
                # position (k,l[k]) for k in range(len(l)) 
                f._coeff[v] = spmatrix(c[0], range(len(l)), l, (len(l),len(v)), 'd')

            else:  # c is 1 by len(v)
                f._coeff[v] = c[len(l)*[0],:]

        return f



class _minmax(object):

    """
    Componentwise maximum or minimum of functions.  

    A function of the form f = max(f1,f2,...,fm) or f = max(f1) or
    f = min(f1,f2,...,fm) or f = min(f1) with each fi an object of 
    type _function.  

    If m>1, then len(f) = max(len(fi)) and f is the componentwise 
    maximum/minimum of f1,f2,...,fm.  Each fi has length 1 or length 
    equal to len(f).

    If m=1, then len(f) = 1 and f is the maximum/minimum of the 
    components of f1: f = max(f1[0],f1[1],...) or 
    f = min(f1[0],f1[1],...).
   

    Attributes:

    _flist       [f1,f2,...,fm]
    _ismax       True for 'max', False for 'min'


    Methods:

    value()      returns the value of the function
    variables()  returns a copy of the list of variables
    """

    def __init__(self,op,*s):  

        self._flist = []

        if op == 'max':
            self._ismax = True
        else: 
            self._ismax = False

        if len(s) == 1: 

            if type(s[0]) is variable or (type(s[0]) is _function and 
                (s[0]._isconvex() and self._ismax) or 
                (s[0]._isconcave() and not self._ismax)):
                self._flist += [+s[0]]
            else:
                raise TypeError('unsupported argument type')

        else:
            # cnst will be max/min of the constant arguments
            cnst = None  

            lg = 1
            for f in s:
                if type(f) is int or type(f) is float: 
                    f = matrix(f, tc='d')

                if _isdmatrix(f) and f.size[1] == 1:
                    if cnst is None: 
                        cnst = +f
                    elif self._ismax: 
                        cnst = _vecmax(cnst,f)
                    else:
                        cnst = _vecmin(cnst,f)

                elif type(f) is variable or type(f) is _function:
                    self._flist += [+f]

                else:
                    raise TypeError('unsupported argument type')

                lgf = len(f)
                if 1 != lg != lgf != 1:
                    raise ValueError('incompatible dimensions')
                elif 1 == lg != lgf: 
                    lg = lgf

            if cnst is not None: self._flist += [_function()+cnst]


    def __len__(self):

        if len(self._flist) == 1: return 1
        for f in self._flist:
            lg = len(f)
            if len(f) > 1: return lg
        return 1
        

    def __repr__(self):

        if self._ismax: s = 'maximum'
        else: s = 'minimum'

        if len(self._flist) == 1:
            return '<' + s + ' component of a function of length %d>'\
                %len(self._flist[0])
        else:
            return "<componentwise " + s + " of %d functions of "\
                "length %d>" %(len(self._flist),len(self))


    def __str__(self):

        s = repr(self)[1:-1] + ':'
        if len(self._flist) == 1:
            s += '\n' + repr(self._flist[0])[1:-1]
        else:
            for k in range(len(self._flist)):
                s += "\nfunction %d: " %k + repr(self._flist[k])[1:-1]
        return s


    def value(self):
 
        if self._ismax:
            return _vecmax(*[f.value() for f in self._flist])
        else:
            return _vecmin(*[f.value() for f in self._flist])


    def variables(self):

        l = varlist()
        for f in self._flist:
            l += [v for v in f.variables() if v not in l]
        return l


    def __pos__(self):
         
        if self._ismax:
            f = _minmax('max', *[+fk for fk in self._flist])
        else:
            f = _minmax('min', *[+fk for fk in self._flist])
        
        return f


    def __neg__(self):

        if self._ismax:
            f = _minmax('min', *[-fk for fk in self._flist])
        else:
            f = _minmax('max', *[-fk for fk in self._flist])
        return f


    def __mul__(self,other):

        if type(other) is int or type(other) is float or \
            (_ismatrix(other) and other.size == (1,1)):
            if _ismatrix(other): other = other[0]

            if other >= 0.0: 
                if self._ismax:
                    f = _minmax('max', *[other*fk for fk in 
                        self._flist])
                else:
                    f = _minmax('min', *[other*fk for fk in
                        self._flist])

            else: 
                if self._ismax:
                    f = _minmax('min', *[other*fk for fk in 
                        self._flist])
                else:
                    f = _minmax('max', *[other*fk for fk in 
                        self._flist])
                
            return f 

        else:
            return NotImplemented

                
    def __rmul__(self,other):

        return self.__mul__(other)


    def __imul__(self,other):

        if _isscalar(other):
            if type(other) is matrix: other = other[0]
            for f in self._flist:  f *= other
            if other < 0.0: self._ismax = not self._ismax
            return self

        raise NotImplementedError('in-place multiplication is only '\
            'defined for scalars')


    def __getitem__(self,key):

        lg = len(self)
        l = _keytolist(key,lg)
        if not l: raise ValueError('empty index set')

        if len(self._flist) == 1: fl = list(self._flist[0])    
        else: fl = self._flist

        if self._ismax: f = _minmax('max')
        else: f = _minmax('min')

        for fk in fl:
            if 1 == len(fk) != lg:  f._flist += [+fk]
            else:  f._flist += [fk[l]]

        return f



def max(*s):

    """
    Identical to the built-in max except when some of the arguments are 
    variables or functions.

    f = max(s1,s2,...) returns the componentwise maximum of s1,s2,..,
    as a convex function with len(f) = max(len(si)).
    The arguments si can be scalars, 1-column dense 'd' matrices, 
    variables, or functions.  At least one argument must be a function 
    or a variable.  The arguments can be scalars or vectors with length 
    equal to len(f).

    f = max(s) with len(s) > 1 returns the maximum component of s as 
    a function with len(f) = 1.  The argument can be a variable or a 
    function.

    f = max(s) with len(s) = 1 and s[0] a function returns s[0].

    f = max(s) with s a list or tuple of variables, functions, 
    constants, returns f = max(*s).

    Does not work with generators (Python 2.4).
    """

    try: return builtins.max(*s)
    except NotImplementedError:
        f = _function()
        try: 
            f._cvxterms = [_minmax('max',*s)]
            return f
        except: 
            # maybe s[0] is a list or tuple of variables, functions
            # and constants
            try: return max(*s[0])
            except: raise NotImplementedError



def min(*s):

    """
    Identical to the built-in min except when some of the arguments are 
    variables or functions.

    f = min(s1,s2,...) returns the componentwise minimum of s1,s2,..,
    as function with len(f) = max(len(si)).
    The arguments si can be scalars, 1-column dense 'd' matrices, 
    variables, or functions.  At least one argument must be a function 
    or a variable.  The arguments can be scalars or vectors with length 
    equal to len(f).

    f = min(s) with len(s) > 1 returns the minimum component of s as 
    a function with len(f) = 1.  The argument can be a variable or a 
    function.

    f = min(s) with len(s) = 1 returns s[0].

    f = min(s) with s a list or tuple of variables, functions, 
    constants, returns f = min(*s).

    Does not work with generators (Python 2.4).
    """

    try: return builtins.min(*s)
    except NotImplementedError:
        f = _function()
        try: 
            f._ccvterms = [_minmax('min',*s)]
            return f
        except:
            # maybe s[0] is a list or tuple of variables, functions
            # and constants
            try: return min(*s[0])
            except: raise NotImplementedError



class _sum_minmax(_minmax):

    """
    Sum of componentwise maximum or minimum of functions.  

    A function of the form f = sum(max(f1,f2,...,fm)) or 
    f = sum(min(f1,f2,...,fm)) with each fi an object of 
    type _function.  

    m must be greater than 1.  len(f) = 1.
    Each fi has length 1 or length equal to max_i len(fi)).


    Attributes:

    _flist       [f1,f2,...,fm]
    _ismax       True for 'max', False for 'min'


    Methods:

    value()      returns the value of the function
    variables()  returns a copy of the list of variables
    _length()    number of terms in the sum
    """

    def __init__(self,op,*s):  

        _minmax.__init__(self,op,*s)
        if len(self._flist) == 1: 
            raise TypeError('expected more than 1 argument')


    def __len__(self):

        return 1


    def _length(self):

        for f in self._flist:
            lg = len(f)
            if len(f) > 1: return lg
        return 1


    def __repr__(self):

        if self._ismax: s = 'maximum'
        else: s = 'minimum'
        return "<sum of componentwise " + s + " of %d functions of "\
            "length %d>" %(len(self._flist),len(self))


    def __str__(self):

        s = repr(self)[1:-1] 
        for k in range(len(self._flist)):
            s += "\nfunction %d: " %k + repr(self._flist[k])[1:-1]
        return s


    def value(self):
 
        if self._ismax:
            return matrix(sum(_vecmax(*[f.value() for f in 
                self._flist])), tc='d')
        else:
            return matrix(sum(_vecmin(*[f.value() for f in 
                self._flist])), tc='d')


    def __pos__(self):
         
        if self._ismax:
            f = _sum_minmax('max', *[+fk for fk in self._flist])
        else:
            f = _sum_minmax('min', *[+fk for fk in self._flist])
        
        return f


    def __neg__(self):

        if self._ismax:
            f = _sum_minmax('min', *[-fk for fk in self._flist])
        else:
            f = _sum_minmax('max', *[-fk for fk in self._flist])
        return f


    def __mul__(self,other):

        if type(other) is int or type(other) is float or \
            (_ismatrix(other) and other.size == (1,1)):

            if _ismatrix(other): other = other[0]

            if other >= 0.0: 
                if self._ismax:
                    f = _sum_minmax('max', *[other*fk for fk in 
                        self._flist])
                else:
                    f = _sum_minmax('min', *[other*fk for fk in
                        self._flist])

            else: 
                if self._ismax:
                    f = _sum_minmax('min', *[other*fk for fk in 
                        self._flist])
                else:
                    f = _sum_minmax('max', *[other*fk for fk in 
                        self._flist])
                
            return f 

        else:
            return NotImplemented

                
    def __rmul__(self,other):

        return self.__mul__(other)


    def __getitem__(self,key):

        l = _keytolist(key,1)
        if not l: raise ValueError('empty index set')

        # expand sum and convert to a  _function
        if self._ismax: f = sum(_minmax('max',*self._flist))
        else: f = sum(_minmax('min',*self._flist))

        return f[l]



class constraint(object):

    """
    Equality or inequality constraint.

    constraint(f, ctype='=', name='') constructs a constraint
    f=0 (if ctype is '=') or f<=0 (if ctype is '<').


    Arguments:

    f                convex function if '<', affine function if '='
    ctype            '=' or '<'
    name             string with the constraint name


    Attributes:

    multiplier       a variable of length len(f).  multiplier.name is 
                     the constraint name with '_mul' appended.
    name             constraint name.  Writing to .name also modifies 
                     the name of .multiplier.  
    _f               the constraint function (borrowed reference)
    _type            '=' or '<'


    Methods:

    value()          returns the value of the constraint function
    variables()      returns the variables of the constraint function
    type()           returns ._type 
    _aslinearineq()  converts convex piecewise-linear inequality into 
                     an equivalent set of linear inequalities
    """

    def __init__(self, f, ctype='=', name=''):

        if ctype == '=' or ctype  == '<':
            self._type = ctype
        else:
            raise TypeError("'ctype' argument must be '<' or '='")

        if type(f) is not _function:
            raise TypeError("'f' argument must be a function")
        
        if ctype == '=':
            if f._isaffine(): self._f = f
            else:
                raise TypeError("constraint function must be affine")
        
        else:
            if f._isconvex(): self._f = f
            else:
                raise TypeError("constraint function must be convex")
 
        self.name = name
        self.multiplier = variable(len(self), name + '_mul')
        

    def __len__(self):
    
        return len(self._f)


    def __repr__(self):

        lg = len(self)

        if self._type == '=': s = 'equality'
        else: s = 'inequality'

        if lg == 1: t = "<scalar %s" %s
        else: t = "<%s in R^%d" %(s,lg)

        if self.name != '': return t + ", '" + self.name + "'>"
        else: return t + ">"


    def __str__(self):

        return repr(self)[1:-1] + '\nconstraint function:\n' + \
            str(self._f)


    def __setattr__(self,name,value):

        if name == 'name':
            if type(value) is str:
                object.__setattr__(self,name,value)
                if hasattr(self,'multiplier'): 
                    self.multiplier.name = value + '_mul'
            else:
                raise TypeError("invalid type for attribute 'name'")

        elif name == 'multiplier' or name == '_type' or name == '_f': 
            object.__setattr__(self,name,value)

        else:
            raise AttributeError("'constraint' object has no "\
                "attribute '%s'" %name)


    def type(self):

        """ Returns '=' for equality constraints, '<' for inequality."""
        
        return self._type


    def value(self):

        """ Returns value of the constraint function."""

        return self._f.value()


    def variables(self):

        """ Returns a list of variables of the constraint function."""

        return self._f.variables()


    def _aslinearineq(self):

        """ 
        Converts a convex PWL inequality into an equivalent set of 
        linear inequalities. 

        Returns a tuple (ineqs, aux_ineqs, aux_vars).  

        If self is a linear inequailty, then ineqs = [self], 
        aux_ineqs = [], aux_vars = [].

        If self is PWL then ineqs and aux_ineqs are two lists of 
        linear inequalities that together are equivalent to self.
        They are separated in two sets so that the multiplier for self 
        depends only on the multipliers of the constraints in ineqs:
        - if len(self) == max(len(ineqs[k])), then the multiplier of 
          self is sum_k ineqs[k].multiplier
        - if len(self) == max(len(ineqs[k])), then the multiplier of 
          self is sum(sum_k ineqs[k].multiplier)

        aux_vars is a varlist with new auxiliary variables.
        """

        if self.type() != '<': 
            raise TypeError('constraint must be an inequality')

        ineqs, aux_ineqs, aux_vars = [], [], varlist()

        # faff._constant and faff._linear are references to the 
        # affine part of the constraint function
        faff = _function()
        faff._constant = self._f._constant 
        faff._linear = self._f._linear

        cvxterms = self._f._cvxterms
        if not cvxterms:  # inequality is linear
            ineqs += [self]

        elif len(cvxterms) == 1 and type(cvxterms[0]) is _minmax:
            # constraint is of the form f = faff + max() <= 0

            if len(cvxterms[0]._flist) == 1:
                # constraint of the form f = faff + max(f0) <= 0 
                f0 = cvxterms[0]._flist[0]

                if len(faff) == 1:
                    # write as scalar + f0 <= 0 with f0 possibly a 
                    # vector
                    c = faff + f0 <= 0
                    c.name = self.name
                    c, caux, newvars = c._aslinearineq()
                    ineqs += c
                    aux_ineqs += caux
                    aux_vars += newvars

                else:
                    # write as vector + f0[k] <= 0 for all k
                    for k in range(len(f0)):
                        c = faff + f0[k] <= 0
                        c.name = self.name + '(%d)' %k 
                        c, caux, newvars = c._aslinearineq()
                        ineqs += c
                        aux_ineqs += caux
                        aux_vars += newvars

            else:
                # constraint of the form f = faff + max(f0,f1,...) <= 0
                for k in range(len(cvxterms[0]._flist)):
                    c = faff + cvxterms[0]._flist[k] <= 0
                    c.name = self.name + '(%d)' %k 
                    c, caux, newvars = c._aslinearineq()
                    ineqs += c
                    aux_ineqs += caux
                    aux_vars += newvars

        else:
            # constraint is of the form f = faff + g1 + g2 .... <= 0 
            # with gi = max() or sum max() and the number of gi's can
            # be one.

            sumt = _function()

            for k in range(len(cvxterms)):
                if type(cvxterms[k]) is _minmax:
                    # gk is max(f0,f1,...)

                    tk = variable(len(cvxterms[k]), 
                        self.name + '_x' + str(k))
                    aux_vars += [tk]
                    sumt = sumt + tk

                    if len(cvxterms[k]._flist) == 1:
                        # add constraint gk = max(f0) <= tk

                        f0 = cvxterms[k]._flist[0]
                        c = f0 <= tk
                        c.name = self.name + '[%d]' %k
                        c, caux, newvars = c._aslinearineq()
                        aux_ineqs += c + caux
                        aux_vars += newvars

                    else:
                        # add constraint gk = max(f0,f1, ... ) <= tk

                        for j in range(len(cvxterms[k]._flist)):
                            fj = cvxterms[k]._flist[j]
                            c = fj <= tk
                            c.name = self.name + '[%d](%d)' %(k,j)
                            c, caux, newvars = c._aslinearineq()
                            aux_ineqs += c + caux
                            aux_vars += newvars

                else:
                    # gk is sum(max(f0,f1,...)

                    tk = variable(cvxterms[k]._length(), self.name + 
                        '_x' + str(k))
                    aux_vars += [tk]
                    sumt = sumt + sum(tk)

                    # add contraint max(f0,f1, ... ) <= tk
                    for j in range(len(cvxterms[k]._flist)):
                        fj = cvxterms[k]._flist[j]
                        c = fj <= tk
                        c.name = self.name + '[%d](%d)' %(k,j)
                        c, caux, newvars = c._aslinearineq()
                        aux_ineqs += c + caux
                        aux_vars += newvars

            c = faff + sumt <= 0 
            c.name = self.name
            ineqs += [c] 
                   
        return (ineqs, aux_ineqs, aux_vars)



class op(object):

    """
    An optimization problem.

    op(objective=0.0, constraints=None, name '') constructs an 
    optimization problem.


    Arguments:

    objective       scalar (int, float, 1x1 dense 'd' matrix), scalar 
                    variable, scalar affine function or scalar convex
                    piecewise-linear function.  Scalars and variables 
                    are converted to affine functions.  
    constraints     None, a single constraint, or a list of constraints
                    None means the same as an empty list.  A single 
                    constraint means the same as a singleton list.
    name            string with the name of the LP


    Attributes:

    objective       the objective function (borrowed reference to the 
                    function passed as argument).  
    name            the name of the optimization problem
    status          initially None.  After solving the problem, 
                    summarizes the outcome.
    _inequalities   list of inequality constraints 
    _equalities     list of equality constraints 
    _variables      a dictionary {v: dictionary with keys 'o','i','e'}
                    The keys v are the variables in the problem.
                    'o': True/False depending on whether v appears in 
                    the objective or not;
                    'i': list of inequality constraints v appears in;
                    'e': list of equality constraints v appears in.
               

    Methods:

    variables()     returns a list of variables.  The list is a varlist
                    (defined below), ie, a subclass of 'list'.
    constraints()   returns a list of constraints 
    inequalities()  returns a list of inequality constraints
    equalities()    returns a list of equality constraints
    delconstraint() deletes a constraint 
    addconstraint() adds a constraint
    _inmatrixform() returns an equivalent LP in matrix form
    solve()         solves the problem
    tofile()        if the problem is an LP, writes it to an MPS file
    fromfile()      reads an LP from an MPS file
    """
     
    def __init__(self, objective=0.0, constraints=None, name=''):

        self._variables = dict()

        self.objective = objective   
        for v in self.objective.variables():
            self._variables[v] = {'o': True, 'i': [], 'e': []}

        self._inequalities, self._equalities = [], []
        if constraints is None: 
            pass
        elif type(constraints) is constraint:
            if constraints.type() == '<':
                self._inequalities += [constraints]
            else:
                self._equalities += [constraints]
        elif type(constraints) == list and not [c for c in constraints
            if type(c) is not constraint]:
            for c in constraints:
                if c.type() == '<':
                    self._inequalities += [c]
                else:
                    self._equalities += [c]
        else: 
            raise TypeError('invalid argument for constraints')

        for c in self._inequalities:
            for v in c.variables():
                if v in self._variables:
                    self._variables[v]['i'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [c], 'e': []}

        for c in self._equalities:
            for v in c.variables():
                if v in self._variables:
                    self._variables[v]['e'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [], 'e': [c]}

        self.name = name
        self.status = None


    def __repr__(self):

        n = sum(map(len,self._variables))
        m = sum(map(len,self._inequalities))
        p = sum(map(len,self._equalities))
        return "<optimization problem with %d variables, %d inequality"\
            " and %d equality constraint(s)>" %(n,m,p)


    def __setattr__(self,name,value):

        if name == 'objective':

            if _isscalar(value):
                value = _function() + value
            elif type(value) is variable and len(value) == 1:
                value = +value
            elif type(value) is _function and value._isconvex() and \
                len(value) == 1:
                pass
            else:
                raise TypeError("attribute 'objective' must be a "\
                    "scalar affine or convex PWL function")

            # remove variables in _variables that only appear in current
            # objective 
            for v in self.variables():
                if not self._variables[v]['i'] and not \
                    self._variables[v]['e']: del self._variables[v]

            object.__setattr__(self,'objective',value)

            # update _variables
            for v in self.objective.variables():
                if v not in self._variables:
                    self._variables[v] = {'o': True, 'i': [], 'e': []}
                else:
                    self._variables[v]['o'] = True

        elif name == 'name':
            if type(value) is str:
                object.__setattr__(self,name,value)
            else:
                raise TypeError("attribute 'name' must be string")

        elif name == '_inequalities' or name == '_equalities' or \
            name == '_variables' or name == 'status':
            object.__setattr__(self,name,value)

        else:
            raise AttributeError("'op' object has no attribute "\
                "'%s'" %name)


    def variables(self):    
    
        """ Returns a list of variables of the LP. """

        return varlist(self._variables.keys())


    def constraints(self):

        """ Returns a list of constraints of the LP."""

        return self._inequalities + self._equalities


    def equalities(self):
    
        """ Returns a list of equality constraints of the LP."""

        return list(self._equalities)


    def inequalities(self):

        """ Returns a list of inequality constraints of the LP."""
        
        return list(self._inequalities)


    def delconstraint(self,c):

        """ 
        Deletes constraint c from the list of constrains  
        """

        if type(c) is not constraint:
            raise TypeError("argument must be of type 'constraint'")

        try: 
            if c.type() == '<': 
                self._inequalities.remove(c)
                for v in c.variables():
                    self._variables[v]['i'].remove(c)
            else: 
                self._equalities.remove(c)
                for v in c.variables():
                    self._variables[v]['e'].remove(c)
            if not self._variables[v]['o'] and \
                not self._variables[v]['i'] and \
                not self._variables[v]['e']:
                del self._variables[v]

        except ValueError:  # c is not a constraint
           pass


    def addconstraint(self,c):

        """ 
        Adds constraint c to the list of constraints. 
        """

        if type(c) is not constraint:
            raise TypeError('argument must be of type constraint')

        if c.type() == '<': self._inequalities += [c]             
        if c.type() == '=': self._equalities += [c]             
        for v in c.variables():
            if c.type() == '<':
                if v in self._variables:
                    self._variables[v]['i'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [c], 'e': []}
            else:
                if v in self._variables:
                    self._variables[v]['e'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [], 'e': [c]}


    def _islp(self):

        """ 
        Returns True if self is an LP; False otherwise.
        """

        if not self.objective._isaffine(): return False
        for c in self._inequalities:
            if not c._f._isaffine(): return False
        for c in self._equalities: 
            if not c._f._isaffine(): return False
        return True


    def _inmatrixform(self, format='dense'):

        """ 
        Converts self to an LP in matrix form 

                minimize    c'*x+d
                subject to  G*x <= h
                            A*x = b.

        c, h, b are dense column matrices; G and A sparse or dense 
        matrices depending on format ('sparse' or 'dense').   

        If self is already an LP in matrix form with the correct matrix
        types, then _inmatrixform() returns None.  Otherwise it returns 
        a tuple (newlp, vmap, mmap).

        newlp is an LP in matrix form with the correct format and 
        matrix types.

        vmap is a dictionary with the variables of self as keys and
        affine functions as values.  For each variable v of self, 
        vmap[v] is a function of the new variable x that can be 
        evaluated to obtain the solution v from the solution x.

        mmap is a dictionary with the constraints of self as keys and
        affine functions as values.  For each constraint c of self, 
        mmap[c] is a function of the multipliers of the new lp that can
        be evaluated to obtain the optimal multiplier for c.
        """

        variables, aux_variables = self.variables(), varlist()   

        # lin_ineqs is a list of linear inequalities in the original
        # problem.  pwl_ineqs is a dictionary {i: [c1,c2,...], ...} 
        # where i is a PWL inequality in the original problem.
        # aux_ineqs are new auxiliary inequalities that together
        # with the ck constraints in pwl_ineqs are equivalent to the
        # original ones.  The sum of the multipliers of the constraints
        # in pwl_ineqs[i] forms the multiplier of i.

        lin_ineqs, pwl_ineqs, aux_ineqs = [], dict(), []
        for i in self._inequalities:
            if i._f._isaffine(): lin_ineqs += [i]
            else: pwl_ineqs[i] = []

        equalities = self._equalities
        objective = +self.objective

        # return None if self is already an LP in the requested form 
        if objective._isaffine() and len(variables) == 1 and \
            not pwl_ineqs and len(lin_ineqs) <= 1 and \
            len(equalities) <= 1:
            v = variables[0]

            if lin_ineqs: G = lin_ineqs[0]._f._linear._coeff[v]
            else: G = None

            if equalities: A = equalities[0]._f._linear._coeff[v]
            else: A = None

            if (format == 'dense' and (G is None or _isdmatrix(G)) and 
                (A is None or _isdmatrix(A))) or \
                (format == 'sparse' and (G is None or _isspmatrix(G)) 
                and (A is None or _isspmatrix(A))):  
                return None


        # convert PWL objective to linear
        if not objective._isaffine():

            # f = affine + sum_k fk with each fk a maximum of convex 
            # functions or a sum of a maximum of convex functions.  
            # If fk is a maximum of convex functions we introduce a
            # new variable tk and replace fk in the objective with tk,
            # with a new constraint fk <= tk.
            # If fk is sum(gk) where gk is a maximum of convex 
            # functions we introduce a new variable tk and replace fk 
            # in the objective with sum(tk) with a new constraint 
            # gk <= tk.

            newobj = _function()
            newobj._constant = +objective._constant
            newobj._linear = +objective._linear

            for k in range(len(objective._cvxterms)):
                fk = objective._cvxterms[k]
                
                if type(fk) is _minmax:
                    tk = variable(1, self.name + '_x' + str(k))
                    newobj += tk
                else:
                    tk = variable(fk._length(), self.name + 
                        '_x' + str(k))
                    newobj += sum(tk)

                aux_variables += [tk]

                for j in range(len(fk._flist)):
                    c = fk._flist[j] <= tk
                    if len(fk._flist) > 1:
                        c.name = self.name + '[%d](%d)' %(k,j)
                    else:
                        c.name = self.name + '[%d]' %k
                    c, caux, newvars = c._aslinearineq()  
                    aux_ineqs += c + caux
                    aux_variables +=  newvars
            objective = newobj


        # convert PWL inequalities to linear
        for i in pwl_ineqs:
            pwl_ineqs[i], caux, newvars = i._aslinearineq()
            aux_ineqs += caux
            aux_variables += newvars


        # n is the length of x, c
        # The variables are available in variables and aux_variables.
        # variable v is stored in x[vslc[v]]
        vslc = dict()
        n = 0
        for v in variables + aux_variables:
            vslc[v] = slice(n, n+len(v))
            n += len(v)
        c = matrix(0.0, (1,n))
        for v,cf in iter(objective._linear._coeff.items()):
            if _isscalar(cf): 
                c[vslc[v]] = cf[0]
            elif _isdmatrix(cf):  
                c[vslc[v]] = cf[:]
            else:  
                c[vslc[v]] = matrix(cf[:], tc='d')
        if n > 0:
            x = variable(n)
            cost = c*x + objective._constant
        else:
            cost = _function() + objective._constant[0]
        vmap = dict()
        for v in variables: vmap[v] = x[vslc[v]]


        # m is the number of rows of G, h
        # The inequalities are available in lin_lineqs, pwl_ineqs,
        # aux_ineqs.
        # inequality i is stored in G[islc[i],:]*x <= h[islc[i]]
        islc = dict()
        for i in lin_ineqs + aux_ineqs:  islc[i] = None
        for c in pwl_ineqs:
            for i in pwl_ineqs[c]: islc[i] = None
        m = 0
        for i in islc:
            islc[i] = slice(m, m+len(i))
            m += len(i)
        if format == 'sparse': 
            G = spmatrix(0.0, [], [], (m,n))   
        else:   
            G = matrix(0.0, (m,n))
        h = matrix(0.0, (m,1))

        for i in islc:
            lg = len(i)
            for v,cf in iter(i._f._linear._coeff.items()):
                if cf.size == (lg, len(v)):
                    if _isspmatrix(cf) and _isdmatrix(G):
                        G[islc[i], vslc[v]] = matrix(cf, tc='d')
                    else:
                        G[islc[i], vslc[v]] = cf
                elif cf.size == (1, len(v)):
                    if _isspmatrix(cf) and _isdmatrix(G):
                        G[islc[i], vslc[v]] = \
                            matrix(cf[lg*[0],:], tc='d')
                    else:
                        G[islc[i], vslc[v]] = cf[lg*[0],:]
                else: #cf.size[0] == (1,1):
                    G[islc[i].start+m*vslc[v].start:
                        islc[i].stop+m*vslc[v].stop:m+1] = cf[0]
            if _isscalar(i._f._constant):
                h[islc[i]] = -i._f._constant[0]
            else:
                h[islc[i]] = -i._f._constant[:]


        # p is the number of rows in A, b
        # equality e is stored A[eslc[e],:]*x == b[eslc[e]]
        eslc = dict()
        p = 0
        for e in equalities:
            eslc[e] = slice(p, p+len(e))
            p += len(e)
        if format == 'sparse':
            A = spmatrix(0.0, [], [], (p,n))  
        else:
            A = matrix(0.0, (p,n))
        b = matrix(0.0, (p,1))

        for e in equalities:
            lg = len(e)
            for v,cf in iter(e._f._linear._coeff.items()):
                if cf.size == (lg,len(v)):
                    if _isspmatrix(cf) and _isdmatrix(A):
                        A[eslc[e], vslc[v]] = matrix(cf, tc='d')
                    else:
                        A[eslc[e], vslc[v]] = cf
                elif cf.size == (1, len(v)):
                    if _isspmatrix(cf) and _isdmatrix(A):
                        A[eslc[e], vslc[v]] = \
                            matrix(cf[lg*[0],:], tc='d')
                    else:
                        A[eslc[e], vslc[v]] = cf[lg*[0],:]
                else: #cf.size[0] == (1,1):
                    A[eslc[e].start+p*vslc[v].start:
                        eslc[e].stop+p*vslc[v].stop:p+1] = cf[0]
            if _isscalar(e._f._constant):
                b[eslc[e]] = -e._f._constant[0]
            else:
                b[eslc[e]] = -e._f._constant[:]

        constraints = []
        if n:
            if m: constraints += [G*x<=h] 
            if p: constraints += [A*x==b]
        else:
            if m: constraints += [_function()-h <= 0] 
            if p: constraints += [_function()-b == 0]
            
        mmap = dict()

        for i in  lin_ineqs:
            mmap[i] = constraints[0].multiplier[islc[i]]

        for i in  pwl_ineqs:
            mmap[i] = _function()
            for c in pwl_ineqs[i]:
                mmap[i] = mmap[i] + constraints[0].multiplier[islc[c]]
            if len(i) == 1 != len(mmap[i]):
                mmap[i] = sum(mmap[i])

        for e in  equalities:
            mmap[e] = constraints[1].multiplier[eslc[e]]
        return (op(cost, constraints), vmap, mmap)


    def solve(self, format='dense', solver = 'default', **kwargs):

        """
        Solves LP using dense or sparse solver.

        format is 'dense' or 'sparse' 

        solver is 'default', 'glpk' or 'mosek'

        solve() sets self.status, and if status is 'optimal', also 
        the value attributes of the variables and the constraint 
        multipliers.  If solver is 'python' then if status is 
        'primal infeasible', the constraint multipliers are set to
        a proof of infeasibility; if status is 'dual infeasible' the
        variables are set to a proof of dual infeasibility.
        """

        t = self._inmatrixform(format)

        if t is None:
            lp1 = self
        else:
            lp1, vmap, mmap = t[0], t[1], t[2]

        variables = lp1.variables()
        if not variables: 
            raise TypeError('lp must have at least one variable')
        x = variables[0]
        c = lp1.objective._linear._coeff[x]
        if _isspmatrix(c): c = matrix(c, tc='d')

        inequalities = lp1._inequalities
        if not inequalities:
            raise TypeError('lp must have at least one inequality')
        G = inequalities[0]._f._linear._coeff[x]
        h = -inequalities[0]._f._constant

        equalities = lp1._equalities
        if equalities:
            A = equalities[0]._f._linear._coeff[x]
            b = -equalities[0]._f._constant
        elif format == 'dense':
            A = matrix(0.0, (0,len(x)))
            b = matrix(0.0, (0,1))
        else:
            A = spmatrix(0.0, [], [],  (0,len(x)))
            b = matrix(0.0, (0,1))

        sol = solvers.lp(c[:], G, h, A, b, solver=solver, **kwargs)

        x.value = sol['x']
        inequalities[0].multiplier.value = sol['z']
        if equalities: equalities[0].multiplier.value = sol['y']

        self.status = sol['status']
        if type(t) is tuple:
            for v,f in iter(vmap.items()): v.value = f.value()
            for c,f in iter(mmap.items()): c.multiplier.value = f.value()
         


    def tofile(self, filename):

        ''' 
        writes LP to file 'filename' in MPS format.
        '''

        if not self._islp(): raise TypeError('problem must be an LP')

        constraints = self.constraints()
        variables = self.variables()
        inequalities = self.inequalities()
        equalities = self.equalities()

        f = open(filename,'w')
        f.write('NAME')
        if self.name: f.write(10*' ' + self.name[:8].rjust(8))
        f.write('\n')

        f.write('ROWS\n') 
        f.write(' N  %8s\n' %'cost')
        for k in range(len(constraints)):
            c = constraints[k]
            for i in range(len(c)):
                if c._type == '<':
                    f.write(' L  ')
                else:
                    f.write(' E  ')
                if c.name:
                    name = c.name 
                else:
                    name = str(k) 
                name = name[:(7-len(str(i)))] + '_' + str(i)
                f.write(name.rjust(8))
                f.write('\n')

        f.write('COLUMNS\n') 
        for k in range(len(variables)):
            v = variables[k]
            for i in range(len(v)):
                if v.name: 
                    varname = v.name
                else:
                    varname = str(k)
                varname = varname[:(7-len(str(i)))] + '_' + str(i)

                if v in self.objective._linear._coeff:
                    cf = self.objective._linear._coeff[v]
                    if cf[i] != 0.0:
                        f.write(4*' ' + varname[:8].rjust(8))
                        f.write(2*' ' + '%8s' %'cost')
                        f.write(2*' ' + '% 7.5E\n' %cf[i])

                for j in range(len(constraints)):
                     c = constraints[j]
                     if c.name:
                         cname = c.name 
                     else:
                         cname = str(j) 
                     if v in c._f._linear._coeff:
                         cf = c._f._linear._coeff[v]
                         if cf.size == (len(c),len(v)):
                             nz = [k for k in range(cf.size[0]) 
                                 if cf[k,i] != 0.0]
                             for l in nz:
                                 conname = cname[:(7-len(str(l)))] \
                                     + '_' + str(l)
                                 f.write(4*' ' + varname[:8].rjust(8))
                                 f.write(2*' ' + conname[:8].rjust(8))
                                 f.write(2*' ' + '% 7.5E\n' %cf[l,i])
                         elif cf.size == (1,len(v)):
                             if cf[0,i] != 0.0:
                                 for l in range(len(c)):
                                     conname = cname[:(7-len(str(l)))] \
                                         + '_' + str(l)
                                     f.write(4*' ' + 
                                         varname[:8].rjust(8))
                                     f.write(2*' ' + 
                                         conname[:8].rjust(8))
                                     f.write(2*' '+'% 7.5E\n' %cf[0,i])
                         elif _isscalar(cf):
                             if cf[0,0] != 0.0:
                                 conname = cname[:(7-len(str(i)))] \
                                     + '_' + str(i)
                                 f.write(4*' ' + varname[:8].rjust(8))
                                 f.write(2*' ' + conname[:8].rjust(8))
                                 f.write(2*' ' + '% 7.5E\n' %cf[0,0])
                        
        f.write('RHS\n') 
        for j in range(len(constraints)):
            c = constraints[j]
            if c.name:
                cname = c.name 
            else:
                cname = str(j) 
            const = -c._f._constant
            for l in range(len(c)):
                 conname = cname[:(7-len(str(l)))] + '_' + str(l)
                 f.write(14*' ' + conname[:8].rjust(8))
                 if const.size[0] == len(c):
                     f.write(2*' ' + '% 7.5E\n' %const[l])
                 else:
                     f.write(2*' ' + '% 7.5E\n' %const[0])

        f.write('RANGES\n') 

        f.write('BOUNDS\n') 
        for k in range(len(variables)):
            v = variables[k]
            for i in range(len(v)):
                if v.name:
                    varname = v.name
                else:
                    varname = str(k)
                varname = varname[:(7-len(str(i)))] + '_' + str(i)
                f.write(' FR ' + 10*' ' + varname[:8].rjust(8) + '\n')

        f.write('ENDATA\n')
        f.close()


    def fromfile(self, filename):

        ''' 
        Reads LP from file 'filename' assuming it is a fixed format 
        ascii MPS file.

        Does not include serious error checking. 

        MPS features that are not allowed: comments preceded by 
        dollar signs, linear combinations of rows, multiple righthand
        sides, ranges columns or bounds columns.
        '''

        self._inequalities = []
        self._equalities = []
        self.objective = _function()
        self.name = ''

        f = open(filename,'r')

        s = f.readline()
        while s[:4] != 'NAME': 
            s = f.readline()
            if not s: 
                raise SyntaxError("EOF reached before 'NAME' section "\
                    "was found")
        self.name = s[14:22].strip()

        s = f.readline()
        while s[:4] != 'ROWS': 
            if not s: 
                raise SyntaxError("EOF reached before 'ROWS' section "\
                    "was found")
            s = f.readline()
        s = f.readline()


        # ROWS section
        functions = dict()   # {MPS row label: affine function}
        rowtypes = dict()    # {MPS row label: 'E', 'G' or 'L'}
        foundobj = False     # first occurrence of 'N' counts
        while s[:7] != 'COLUMNS': 
            if not s: raise SyntaxError("file has no 'COLUMNS' section")
            if len(s.strip()) == 0 or s[0] == '*': 
                pass
            elif s[1:3].strip() in ['E','L','G']:
                rowlabel = s[4:12].strip()
                functions[rowlabel] = _function()
                rowtypes[rowlabel] = s[1:3].strip()
            elif s[1:3].strip() == 'N':
                rowlabel = s[4:12].strip()
                if not foundobj:
                    functions[rowlabel] = self.objective
                    foundobj = True
            else: 
                raise ValueError("unknown row type '%s'" %s[1:3].strip())
            s = f.readline()
        s = f.readline()


        # COLUMNS section
        variables = dict()   # {MPS column label: variable}
        while s[:3] != 'RHS': 
            if not s: 
                raise SyntaxError("EOF reached before 'RHS' section "\
                    "was found")
            if len(s.strip()) == 0 or s[0] == '*': 
                pass
            else:
                if s[4:12].strip(): collabel = s[4:12].strip()
                if collabel not in variables:
                    variables[collabel] = variable(1,collabel)
                v = variables[collabel]
                rowlabel = s[14:22].strip()
                if rowlabel not in functions:
                    raise KeyError("no row label '%s'" %rowlabel)
                functions[rowlabel]._linear._coeff[v] = \
                    matrix(float(s[24:36]), tc='d')
                rowlabel = s[39:47].strip()
                if rowlabel:
                    if rowlabel not in functions:
                        raise KeyError("no row label '%s'" %rowlabel)
                    functions[rowlabel]._linear._coeff[v] =  \
                        matrix(float(s[49:61]), tc='d')
            s = f.readline()
        s = f.readline()


        # RHS section
        # The RHS section may contain multiple right hand sides,
        # identified with different labels in s[4:12].
        # We read in only one rhs, the one with the first rhs label 
        # encountered.
        rhslabel = None
        while s[:6] != 'RANGES' and s[:6] != 'BOUNDS' and \
            s[:6] != 'ENDATA':
            if not s: raise SyntaxError( \
                 "EOF reached before 'ENDATA' was found")
            if len(s.strip()) == 0 or s[0] == '*': 
                pass
            else:
                if None != rhslabel != s[4:12].strip():
                    # skip if rhslabel is different from 1st rhs label
                    # encountered
                    pass  
                else:
                    if rhslabel is None: rhslabel = s[4:12].strip()
                    rowlabel = s[14:22].strip()
                    if rowlabel not in functions:
                        raise KeyError("no row label '%s'" %rowlabel)
                    functions[rowlabel]._constant = \
                        matrix(-float(s[24:36]), tc='d')
                    rowlabel = s[39:47].strip()
                    if rowlabel:
                        if rowlabel not in functions:
                            raise KeyError("no row label '%s'" \
                                %rowlabel)
                        functions[rowlabel]._constant = \
                            matrix(-float(s[49:61]), tc='d')
            s = f.readline()


        # RANGES section
        # The RANGES section may contain multiple range vectors,
        # identified with different labels in s[4:12].
        # We read in only one vector, the one with the first range label
        # encountered.
        ranges = dict()
        for l in iter(rowtypes.keys()): 
            ranges[l] = None   # {rowlabel: range value}
        rangeslabel = None
        if s[:6] == 'RANGES':
            s = f.readline()
            while s[:6] != 'BOUNDS' and s[:6] != 'ENDATA':
                if not s: raise SyntaxError( \
                    "EOF reached before 'ENDATA' was found")
                if len(s.strip()) == 0 or s[0] == '*': 
                    pass
                else:
                    if None != rangeslabel != s[4:12].strip():
                        pass  
                    else:
                        if rangeslabel == None: 
                            rangeslabel = s[4:12].strip()
                        rowlabel = s[14:22].strip()
                        if rowlabel not in rowtypes:
                            raise KeyError("no row label '%s'"%rowlabel)
                        ranges[rowlabel] = float(s[24:36])
                        rowlabel = s[39:47].strip()
                        if rowlabel != '':
                            if rowlabel not in functions:
                                raise KeyError("no row label '%s'" \
                                    %rowlabel)
                            ranges[rowlabel] =  float(s[49:61])
                s = f.readline()


        # BOUNDS section
        # The BOUNDS section may contain bounds vectors, identified 
        # with different labels in s[4:12].
        # We read in only one bounds vector, the one with the first 
        # label encountered.
        boundslabel = None
        bounds = dict()
        for v in iter(variables.keys()):  
            bounds[v] = [0.0, None] #{column label: [l.bound, u. bound]}
        if s[:6] == 'BOUNDS':
            s = f.readline()
            while s[:6] != 'ENDATA':
                if not s: raise SyntaxError( \
                    "EOF reached before 'ENDATA' was found")
                if len(s.strip()) == 0 or s[0] == '*': 
                    pass
                else:
                    if None != boundslabel != s[4:12].strip():
                        pass  
                    else:
                        if boundslabel is None: 
                            boundslabel = s[4:12].strip()
                        collabel = s[14:22].strip()
                        if collabel not in variables:
                            raise ValueError('unknown column label ' \
                                + "'%s'" %collabel)
                        if s[1:3].strip() == 'LO': 
                            if bounds[collabel][0] != 0.0:
                                raise ValueError("repeated lower "\
                                    "bound for variable '%s'" %collabel)
                            bounds[collabel][0] = float(s[24:36])
                        elif s[1:3].strip() == 'UP': 
                            if bounds[collabel][1] != None:
                                raise ValueError("repeated upper "\
                                    "bound for variable '%s'" %collabel)
                            bounds[collabel][1] = float(s[24:36])
                        elif s[1:3].strip() == 'FX': 
                            if bounds[collabel] != [0, None]:
                                raise ValueError("repeated bounds "\
                                    "for variable '%s'" %collabel)
                            bounds[collabel][0] = float(s[24:36])
                            bounds[collabel][1] = float(s[24:36])
                        elif s[1:3].strip() == 'FR': 
                            if bounds[collabel] != [0, None]:
                                raise ValueError("repeated bounds "\
                                    "for variable '%s'" %collabel)
                            bounds[collabel][0] = None
                            bounds[collabel][1] = None
                        elif s[1:3].strip() == 'MI': 
                            if bounds[collabel][0] != 0.0:
                                raise ValueError("repeated lower " \
                                    "bound for variable '%s'" %collabel)
                            bounds[collabel][0] = None
                        elif s[1:3].strip() == 'PL': 
                            if bounds[collabel][1] != None:
                                raise ValueError("repeated upper " \
                                    "bound for variable '%s'" %collabel)
                        else:
                            raise ValueError("unknown bound type '%s'"\
                                %s[1:3].strip())
                s = f.readline()

        for l, type in iter(rowtypes.items()):

            if type == 'L':   
                c = functions[l] <= 0.0
                c.name = l
                self._inequalities += [c]
                if ranges[l] != None:
                    c = functions[l] >= -abs(ranges[l])     
                    c.name = l + '_lb'
                    self._inequalities += [c]
            if type == 'G':   
                c = functions[l] >= 0.0
                c.name = l
                self._inequalities += [c]
                if ranges[l] != None:
                    c = functions[l] <= abs(ranges[l])     
                    c.name = l + '_ub'
                    self._inequalities += [c]
            if type == 'E':   
                if ranges[l] is None or ranges[l] == 0.0:
                    c = functions[l] == 0.0
                    c.name = l
                    self._equalities += [c]
                elif ranges[l] > 0.0:
                    c = functions[l] >= 0.0
                    c.name = l + '_lb'
                    self._inequalities += [c]
                    c = functions[l] <= ranges[l]
                    c.name = l + '_ub'
                    self._inequalities += [c]
                else:
                    c = functions[l] <= 0.0
                    c.name = l + '_ub'
                    self._inequalities += [c]
                    c = functions[l] >= ranges[l]
                    c.name = l + '_lb'
                    self._inequalities += [c]

        for l,bnds in iter(bounds.items()):
            v = variables[l]
            if None != bnds[0] != bnds[1]:
                c = v >= bnds[0]
                self._inequalities += [c]
            if bnds[0] != bnds[1] != None:
                c = v  <= bnds[1]
                self._inequalities += [c]
            if None != bnds[0] == bnds[1]:
                c = v == bnds[0]
                self._equalities += [c]

        # Eliminate constraints with no variables
        for c in self._inequalities + self._equalities:
            if len(c._f._linear._coeff) == 0:
                if c.type() == '=' and c._f._constant[0] != 0.0:
                    raise ValueError("equality constraint '%s' "\
                       "has no variables and a nonzero righthand side"\
                       %c.name)
                elif c.type() == '<' and c._f._constant[0] > 0.0:
                    raise ValueError("inequality constraint '%s' "\
                       "has no variables and a negative righthand side"\
                       %c.name)
                else:
                    print("removing redundant constraint '%s'" %c.name)
                    if c.type() == '<': self._inequalities.remove(c)
                    if c.type() == '=': self._equalities.remove(c)


        self._variables = dict()
        for v in self.objective._linear._coeff.keys():
            self._variables[v] = {'o': True, 'i': [], 'e': []}
        for c in self._inequalities:
            for v in c._f._linear._coeff.keys():
                if v in self._variables:
                    self._variables[v]['i'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [c], 'e': []}
        for c in self._equalities:
            for v in c._f._linear._coeff.keys():
                if v in self._variables:
                    self._variables[v]['e'] += [c]
                else:
                    self._variables[v] = {'o': False, 'i': [], 'e': [c]}
           
        self.status = None

        f.close()



def dot(x,y):

    """
    Inner products of variable or affine function with constant vector.
    """ 
    
    if _isdmatrix(x) and _isdmatrix(y):
        return blas.dot(x,y)

    elif _isdmatrix(x) and (type(y) is variable or 
        (type(y) is _function and y._isaffine()) and  
        x.size == (len(y),1)):
        return x.trans() * y

    elif _isdmatrix(y) and (type(x) is variable or 
        (type(x) is _function and x._isaffine()) and 
        y.size == (len(x),1)):
        return y.trans() * x

    else:
        raise TypeError('invalid argument types or incompatible '\
            'dimensions')



def _isscalar(a):   

    """ True if a is an int or float or 1x1 dense 'd' matrix. """

    if type(a) is int or type(a) is float or (_isdmatrix(a) and
        a.size == (1,1)): return True
    else: return False



def _isdmatrix(a):   

    """ True if a is a nonempty dense 'd' matrix. """

    if type(a) is matrix and a.typecode == 'd' and min(a.size) != 0: 
        return True
    else: 
        return False



def _isspmatrix(a):   

    """ True if a is a nonempty sparse 'd' matrix. """

    if type(a) is spmatrix and a.typecode == 'd' and min(a.size) != 0: 
        return True
    else: 
        return False



def _ismatrix(a):   

    """ True if a is a nonempty 'd' matrix. """

    if type(a) in (matrix, spmatrix) and a.typecode == 'd' and \
        min(a.size) != 0: 
        return True
    else: 
        return False



def _keytolist(key,n):

    """
    Converts indices, index lists, index matrices, and slices of
    a length n sequence into lists of integers.

    key is the index passed to a call to __getitem__().
    """

    if type(key) is int:
        if -n <= key < 0:  
            l = [key+n] 
        elif 0 <= key < n:  
            l = [key] 
        else:  
            raise IndexError('variable index out of range')

    elif (type(key) is list and not [k for k in key if type(k) is not 
        int]) or (type(key) is matrix and key.typecode == 'i'):
        l = [k for k in key if -n <= k < n]
        if len(l) != len(key):
            raise IndexError('variable index out of range')
        for i in range(len(l)): 
            if l[i] < 0: l[i] += n
        
    elif type(key) is slice:
            
        ind = key.indices(n)
        l = list(range(ind[0],ind[1],ind[2]))

    else: 
        raise TypeError('invalid key')

    return l



class varlist(list):
 
    """
    Standard list with __contains__() redefined to use 'is' 
    instead of '=='.
    """

    def __contains__(self,item):
            
        for k in range(len(self)): 
            if self[k] is item: return True
        return False



def _vecmax(*s):

    """
    _vecmax(s1,s2,...) returns the componentwise maximum of s1, s2,... 
    _vecmax(s) returns the maximum component of s.
    
    The arguments can be None, scalars or 1-column dense 'd' vectors 
    with lengths equal to 1 or equal to the maximum len(sk).  

    Returns None if one of the arguments is None.  
    """

    if not s:
        raise TypeError("_vecmax expected at least 1 argument, got 0")

    val = None
    for c in s:

        if c is None: 
            return None

        elif type(c) is int or type(c) is float: 
            c = matrix(c, tc='d')

        elif not _isdmatrix(c) or c.size[1] != 1:
            raise TypeError("incompatible type or size")

        if val is None:
            if len(s) == 1:  return matrix(max(c), tc='d')
            else: val = +c

        elif len(val) == 1 != len(c): 
            val = matrix([max(val[0],x) for x in c], tc='d')

        elif len(val) != 1 == len(c):
            val = matrix([max(c[0],x) for x in val], tc='d')

        elif len(val) == len(c):
            val = matrix( [max(val[k],c[k]) for k in range(len(c))], 
                tc='d' )
    
        else: 
            raise ValueError('incompatible dimensions')

    return val



def _vecmin(*s):

    """
    _vecmin(s1,s2,...) returns the componentwise minimum of s1, s2,... 
    _vecmin(s) returns the minimum component of s.
    
    The arguments can be None, scalars or 1-column dense 'd' vectors 
    with lengths equal  to 1 or equal to the maximum len(sk).  

    Returns None if one of the arguments is None.  
    """

    if not s:
        raise TypeError("_vecmin expected at least 1 argument, got 0")

    val = None
    for c in s:

        if c is None: 
            return None

        elif type(c) is int or type(c) is float: 
            c = matrix(c, tc='d')

        elif not _isdmatrix(c) or c.size[1] != 1:
            raise TypeError("incompatible type or size")

        if val is None:
            if len(s) == 1:  return matrix(min(c), tc='d')
            else: val = +c

        elif len(val) == 1 != len(c): 
            val = matrix( [min(val[0],x) for x in c], tc='d' )

        elif len(val) != 1 == len(c):
            val = matrix( [min(c[0],x) for x in val], tc='d' )

        elif len(val) == len(c):
            val = matrix( [min(val[k],c[k]) for k in range(len(c))], 
                tc='d' )
    
        else: 
            raise ValueError('incompatible dimensions')

    return val
