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

options = {'dformat' : '% .2e',
           'iformat' : '% i',
           'width' : 7,
           'height' : -1}

def matrix_str_default(X):

    from sys import maxsize
    from cvxopt.printing import options

    width, height = options['width'], options['height']
    iformat, dformat = options['iformat'], options['dformat']

    sgn = ['-','+']

    if   X.typecode == 'i': fmt = iformat
    else: fmt = dformat

    s = ""
    m, n   = X.size
    if width < 0: width = maxsize
    if height < 0: height = maxsize

    if width*height == 0: return ""
    if len(X) == 0: return ""

    rlist = range(0,min(m,height))
    clist = range(0,min(n,width))

    if X.typecode == 'z':
        twidth = max([ len(fmt % X[i,j].real + sgn[X[i,j].imag>0] + 'j' + \
                              (fmt % abs(X[i,j].imag)).lstrip() ) \
                          for i in rlist for j in clist ])
    else:
        twidth = max([ len(fmt % X[i,j]) for i in rlist for j in clist ])

    for i in rlist:
        s += '['

        for j in clist:

            if X.typecode == 'z':
                s += format(fmt % X[i,j].real + sgn[X[i,j].imag>0] + 'j' +\
                          (fmt % abs(X[i,j].imag)).lstrip(), '>%i' %twidth)
            else:
                s += format(fmt % X[i,j], '>%i' %twidth)

            s += ' '
        
        if width < n: s += '... ]\n'        
        else: s = s[:-1] + ']\n'
           
    if height < m: 
        s += "[" + min(n,width)*(':'.center(twidth)+' ')

        if width < n: s += '    ]\n'
        else: s = s[:-1] + ']\n'

    return s

def matrix_repr_default(X):
    return "<%ix%i matrix, tc='%c'>" %(X.size[0],X.size[1],X.typecode)

def spmatrix_str_default(X):

    from sys import maxsize
    from cvxopt.printing import options

    width, height = options['width'], options['height']
    iformat, dformat = options['iformat'], options['dformat']

    sgn = ['-','+']

    if   X.typecode == 'i': fmt = iformat
    else: fmt = dformat

    s = ""
    m, n   = X.size
    if width < 0: width = maxsize
    if height < 0: height = maxsize

    if width*height == 0: return ""
 
    rlist = range(0,min(m,height))
    clist = range(0,min(n,width))

    Xr = X[:min(m,height),:min(n,width)]
    Idx = list(zip(*(Xr.I,Xr.J)))
    
    if len(Idx) > 0:
        if X.typecode == 'z':
            twidth = max([ len(fmt % X[i,j].real + sgn[X[i,j].imag>0] + 'j' + \
                                  (fmt % abs(X[i,j].imag)).lstrip() ) \
                              for i in rlist for j in clist ])
        else:
            twidth = max([ len(fmt % X[i,j]) for i in rlist for j in clist ])
    else:
        twidth = 1

    for i in rlist:
        s += '['

        for j in clist:

            if (i,j) in Idx:
                if X.typecode == 'z':
                    s +=  format(fmt % X[i,j].real + sgn[X[i,j].imag>0] + 'j' + \
                                 (fmt % abs(X[i,j].imag)).lstrip(), '>%i' %twidth)
                else:
                    s += format(fmt % X[i,j], '>%i' %twidth)
            else: 
                s += format(0, '^%i' %twidth)
                
            s += ' '
        
        if width < n: s += '... ]\n'        
        else: s = s[:-1] + "]\n"
           
    if height < m: 
        s += "[" + min(n,width)*(format(':', '^%i' %twidth)+' ')

        if width < n: s += '   ]\n'
        else: s = s[:-1] + ']\n'

    return s


def spmatrix_str_triplet(X):

    from cvxopt.printing import options

    iformat, dformat = options['iformat'], options['dformat']

    sgn = ['-','+']

    if   X.typecode == 'i': fmt = iformat
    else: fmt = dformat

    s = ""
    
    if len(X) > 0:
        if X.typecode == 'z':
            twidth = max([ len(fmt % Xk.real + sgn[Xk.imag>0] + 'j' + \
                                   (fmt % abs(Xk.imag)).lstrip() ) \
                               for Xk in X.V ])
        else:
            twidth = max([ len(fmt % Xk) for Xk in X.V ])

        imax = max([ len(str(i)) for i in X.I ])
        jmax = max([ len(str(j)) for j in X.J ])

    else:
        twidth = 0 

    for k in range(len(X)):
        s += "(" 
        s += format(X.I[k], '>%i' %imax)  + "," + \
            format(X.J[k], '>%i' %jmax) 
        s += ") "

        if X.typecode=='z':
            s +=  format(fmt % X.V[k].real + sgn[X.V[k].imag>0] + 'j' + \
                         (fmt % abs(X.V[k].imag)).lstrip(), '>%i' %twidth)
        else:
            s += format(fmt % X.V[k], '>%i' %twidth)
        s += "\n"
                
    return s

def spmatrix_repr_default(X):
    return "<%ix%i sparse matrix, tc='%c', nnz=%i>" \
        %(X.size[0],X.size[1],X.typecode,len(X.V))
