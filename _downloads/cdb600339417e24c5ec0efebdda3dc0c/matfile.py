"""
Functions for reading and writing Matlab mat-files.

Based on the documentation of the mat-file format at
www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf.
"""

from cvxopt import matrix, spmatrix
from struct import pack, unpack, calcsize
from math import ceil
from time import ctime
import zlib

__all__  = ['read', 'write']

def read(f):

    """
    Returns a dictionary with the contents of the mat-file f.

    The elements in the dictionary are Matlab name/variable pairs.
    Matlab variables of a type that is not supported (for example, 
    arrays of dimension higher than two) are replaced by strings.
    """

    def decode_str(s, E):

        TYPES = ['','b','B','h','H','i','i','f','','d']
        CLASS = ['Cell array','Structure','Object']

        def get_tag(s):
            (utype, ulgt) =  unpack("%sii"%E, s[:8])
            if (utype >> 16):
                ulgt, utype = (utype >> 16), (utype & 0xffff)
                offs, inc = 4, 8*((4+ulgt+7)/8)
            else:
                offs, inc = 8, 8*((8+ulgt+7)/8)
            return (utype, ulgt, offs, inc)
                          
        datatype, numbytes = unpack("%sii"%E,s[:8])
        idx = 8
    
        if datatype == 14:

            utype, ulgt, flags =  unpack("%siii"%E,s[idx:idx+12])
            flags, cls = (flags >> 8) & 0xff, flags & 0xff        
            nzmax = unpack("i",s[idx+12:idx+16])[0] # only for sparse
            idx += 16        
                
            # dimension array
            (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8])
            dims = unpack("%s%ii"%(E,ulgt/4), s[idx+offs:idx+offs+ulgt])
            idx += inc
            if len(dims)>2:
                return (name, "array with dimensions %s not implemented" %dims)

            # array name
            (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8])
            name = unpack("%is"%ulgt, s[idx+offs:idx+offs+ulgt])[0]
            idx += inc

            if cls in [0,1,2,3]:
                return (name, CLASS[cls-1]+" not implemented")

            if cls == 4:
                (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8]) 
                return (name,unpack("%is"%ulgt, s[idx+offs:idx+offs+ulgt])[0])
            
            # sparse data
            elif cls == 5: 
                if (flags & 8): return (name, "complex sparse not implemented")
                
                (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8])
                ri = unpack("%ii"%(ulgt/4), s[idx+offs:idx+offs+ulgt])
                idx += inc

                (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8])
                cp = unpack("%ii"%(ulgt/4), s[idx+offs:idx+offs+ulgt])
                idx += inc
                
                # real part
                (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8])
                tmp = "%s%i%s" %(E,ulgt/calcsize(TYPES[utype]),TYPES[utype])
                pr = unpack(tmp, s[idx+offs:idx+offs+ulgt])
                idx += inc

                # convert CCS to IJV
                V = matrix(list(pr))                
                I = matrix(list(ri[:len(V)]),tc='i')
                J = matrix(0, size=(len(I),1), tc='i')

                for i in xrange(1,len(cp)):
                    J[cp[i-1]:cp[i]]=i-1
                
                return (name, spmatrix(V,I,J,size=dims))
                
            # dense matrix 
            # real part
            (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8])
            tmp = "%s%i%s" %(E,ulgt/calcsize(TYPES[utype]),TYPES[utype])
            pr = unpack(tmp, s[idx+offs:idx+offs+ulgt])
            idx += inc
            
            if cls in [6,7]:
                tc = 'd'
            else:
                tc = 'i'
                
            pr = matrix(list(pr), dims, tc)

            # imaginary part
            if (flags & 8):
                (utype, ulgt, offs, inc) = get_tag(s[idx:idx+8])
                tmp = "%s%i%s" %(E,ulgt/calcsize(TYPES[utype]),TYPES[utype])
                pi = unpack(tmp, s[idx+offs:idx+offs+ulgt])
                pi = matrix(list(pi), dims, tc)

                return (name, pr+1J*pi)
            else:
                return (name, pr)                
        
        elif datatype == 15:
            return decode_str(zlib.decompress(s[8:]), E)

        else: raise ValueError, "unexpected datatype"

    #
    # Start of read
    #
    s = open(f,'rb').read()

    E = '<'
    endian = unpack("%s2s"%E,s[126:128])[0]
    if endian == 'MI': E = '>'

    matstr, subsys, version = unpack("%s116s8sH"%E,s[:126])    
    if version != 0x0100: raise ValueError, "not a level 5 MAT file"

    lgt = 128

    D = {}
    while lgt < len(s):
        datatype, numbytes = unpack("%sii"%E,s[lgt:lgt+8])

        (name, var) = decode_str(s[lgt:lgt+8+numbytes],E)
        D[name] = var
    
        lgt += numbytes + 8

    return D



def write(f, var):

    """
    Writes the dictionary var to the mat-file f.  

    var is a dictionary with name/variable pairs.  The name is the name
    of the Matlab variable created in the mat-file.  The type of the
    variable must be matrix, spmatrix, string, int, float or complex. 
    """

    def encode_obj(name, val):

        def encode_str(var, id):
            if len(var) <= 4:
                return pack("HH4s",id,len(var),var)
            else:
                l = int(8*ceil(len(var)/8.0))
                return pack("ii%is"%l,id,len(var),var)
        
        def encode_cvxopt_mtx(val):
            if val.typecode == 'i':
                s = pack("ii",5,len(val)*4)  
                for i in xrange(len(val)):
                    s  += pack("i", val[i])
                
                npad = 8*int(ceil(4*len(val)/8.0))
                s += (npad-len(val)*4)*'\0'
            
            elif val.typecode == 'd':
                s = pack("ii",9,len(val)*8)
                for i in xrange(len(val)):
                    s += pack("d", val[i])
                
            else:
                s = pack("ii",9,len(val)*8)
                for i in xrange(len(val)):
                    s += pack("d", val[i].real)
                s += pack("ii",9,len(val)*8)
                for i in xrange(len(val)):
                    s += pack("d", val[i].imag)

            return s
    

        if isinstance(val, matrix):
            flags = 8*(val.typecode == 'z')
                
            if val.typecode == 'i':
                cls = 12
            else:
                cls = 6
                
            s  = pack("iiixxxx", 6, 8, flags*256+cls)        
            s += pack("iiii", 5, 8, val.size[0], val.size[1])
            s += encode_str(name,1)
            s += encode_cvxopt_mtx(val)
            
        elif isinstance(val, spmatrix):
            flags, cls, nzmax = 0, 5, len(val)

            I, J, V = val.I, val.J, val.V
            
            cp = matrix(0,(val.size[1]+1,1),tc='i')
            for i in xrange(len(J)):
                cp[J[i]+1] += 1

            for i in xrange(val.size[1]):
                cp[i+1] += cp[i]

            s  = pack("iiii", 6, 8, flags*256+cls, nzmax)        
            s += pack("iiii", 5, 8, val.size[0], val.size[1])
            s += encode_str(name,1)
            
            s += encode_cvxopt_mtx(I)
            s += encode_cvxopt_mtx(cp)
            s += encode_cvxopt_mtx(V)

        elif isinstance(val, str):
            flags, cls = 0, 4
            s  = pack("iiixxxx", 6, 8, flags*256+cls)
            s += pack("iiii", 5, 8, 1, len(val))
            s += encode_str(name,1)
            s += encode_str(val,16)            

        elif isinstance(val, int):
            return encode_obj(name, matrix(val, tc='i'))

        elif isinstance(val, float):
            return encode_obj(name, matrix(val, tc='d'))

        elif isinstance(val, complex):
            return encode_obj(name, matrix(val, tc='z'))

        else:
            raise TypeError, "%s not supported" %type(val)

        s = pack("ii",14,len(s)) + s                    
        #s = zlib.compress(s, 9)
        #s = pack("ii",15,len(s)) + s
    
        return s
        
    #
    # start of write
    #
    IDstr = "MATLAB 5.0 MAT-file, Created by CVXOPT on " + ctime()
    endian = 256*ord('M')+ord('I')
    
    s = pack("116s8sHH",IDstr,' ',0x0100,endian)
    
    for name in var.keys():
        s2 = encode_obj(name, var[name])
        s += s2

    open(f,'wb').write(s)
