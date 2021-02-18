# The floor planning example section 9.2 (Problems with linear objectives). 

from cvxopt import solvers, matrix, spmatrix, mul, div  
solvers.options['show_progress'] = False
 
def floorplan(Amin):  
 
    #     minimize    W+H  
    #     subject to  Amink / hk <= wk, k = 1,..., 5  
    #                 x1 >= 0,  x2 >= 0, x4 >= 0  
    #                 x1 + w1 + rho <= x3  
    #                 x2 + w2 + rho <= x3  
    #                 x3 + w3 + rho <= x5  
    #                 x4 + w4 + rho <= x5  
    #                 x5 + w5 <= W  
    #                 y2 >= 0,  y3 >= 0,  y5 >= 0  
    #                 y2 + h2 + rho <= y1  
    #                 y1 + h1 + rho <= y4  
    #                 y3 + h3 + rho <= y4  
    #                 y4 + h4 <= H  
    #                 y5 + h5 <= H  
    #                 hk/gamma <= wk <= gamma*hk,  k = 1, ..., 5  
    #  
    # 22 Variables W, H, x (5), y (5), w (5), h (5).  
    #  
    # W, H:  scalars; bounding box width and height  
    # x, y:  5-vectors; coordinates of bottom left corners of blocks  
    # w, h:  5-vectors; widths and heights of the 5 blocks  
 
    rho, gamma = 1.0, 5.0   # min spacing, min aspect ratio  
 
    # The objective is to minimize W + H.  There are five nonlinear  
    # constraints  
    #  
    #     -wk + Amink / hk <= 0,  k = 1, ..., 5  
 
    c = matrix(2*[1.0] + 20*[0.0])  
 
    def F(x=None, z=None):  
        if x is None:  return 5, matrix(17*[0.0] + 5*[1.0])  
        if min(x[17:]) <= 0.0:  return None  
        f = -x[12:17] + div(Amin, x[17:])  
        Df = matrix(0.0, (5,22))  
        Df[:,12:17] = spmatrix(-1.0, range(5), range(5))  
        Df[:,17:] = spmatrix(-div(Amin, x[17:]**2), range(5), range(5))  
        if z is None: return f, Df  
        H = spmatrix( 2.0* mul(z, div(Amin, x[17::]**3)), range(17,22), 
            range(17,22) )  
        return f, Df, H  
 
    G = matrix(0.0, (26,22))  
    h = matrix(0.0, (26,1))  

    # -x1 <= 0  
    G[0,2] = -1.0                                    

    # -x2 <= 0   
    G[1,3] = -1.0                                   

    # -x4 <= 0  
    G[2,5] = -1.0

    # x1 - x3 + w1 <= -rho
    G[3, [2, 4, 12]], h[3] = [1.0, -1.0, 1.0], -rho  

    # x2 - x3 + w2 <= -rho
    G[4, [3, 4, 13]], h[4] = [1.0, -1.0, 1.0], -rho  

    # x3 - x5 + w3 <= -rho
    G[5, [4, 6, 14]], h[5] = [1.0, -1.0, 1.0], -rho  

    # x4 - x5 + w4 <= -rho
    G[6, [5, 6, 15]], h[6] = [1.0, -1.0, 1.0], -rho  

    # -W + x5 + w5 <= 0  
    G[7, [0, 6, 16]] = -1.0, 1.0, 1.0                

    # -y2 <= 0  
    G[8,8] = -1.0                                    

    # -y3 <= 0  
    G[9,9] = -1.0                                    

    # -y5 <= 0  
    G[10,11] = -1.0                                  

    # -y1 + y2 + h2 <= -rho  
    G[11, [7, 8, 18]], h[11] = [-1.0, 1.0, 1.0], -rho  

    # y1 - y4 + h1 <= -rho  
    G[12, [7, 10, 17]], h[12] = [1.0, -1.0, 1.0], -rho  

    # y3 - y4 + h3 <= -rho  
    G[13, [9, 10, 19]], h[13] = [1.0, -1.0, 1.0], -rho  

    # -H + y4 + h4 <= 0  
    G[14, [1, 10, 20]] = -1.0, 1.0, 1.0                 

    # -H + y5 + h5 <= 0  
    G[15, [1, 11, 21]] = -1.0, 1.0, 1.0                 

    # -w1 + h1/gamma <= 0  
    G[16, [12, 17]] = -1.0, 1.0/gamma                   

    # w1 - gamma * h1 <= 0  
    G[17, [12, 17]] = 1.0, -gamma                       

    # -w2 + h2/gamma <= 0  
    G[18, [13, 18]] = -1.0, 1.0/gamma                   

    #  w2 - gamma * h2 <= 0  
    G[19, [13, 18]] = 1.0, -gamma                       

    # -w3 + h3/gamma <= 0  
    G[20, [14, 18]] = -1.0, 1.0/gamma                   

    #  w3 - gamma * h3 <= 0  
    G[21, [14, 19]] = 1.0, -gamma                       

    # -w4  + h4/gamma <= 0  
    G[22, [15, 19]] = -1.0, 1.0/gamma                   

    #  w4 - gamma * h4 <= 0  
    G[23, [15, 20]] = 1.0, -gamma                       

    # -w5 + h5/gamma <= 0  
    G[24, [16, 21]] = -1.0, 1.0/gamma                   

    #  w5 - gamma * h5 <= 0.0  
    G[25, [16, 21]] = 1.0, -gamma                       

    # solve and return W, H, x, y, w, h  
    sol = solvers.cpl(c, F, G, h)  
    return  sol['x'][0], sol['x'][1], sol['x'][2:7], sol['x'][7:12], \
        sol['x'][12:17], sol['x'][17:]  
 
try: 
    import pylab
except ImportError: 
    pass
else:
    pylab.figure(facecolor='w')  
    pylab.subplot(221)  
    Amin = matrix([100., 100., 100., 100., 100.])  
    W, H, x, y, w, h =  floorplan(Amin)  
    for k in range(5):  
        pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],  
                   [y[k], y[k]+h[k], y[k]+h[k], y[k]], 
                   facecolor = '#D0D0D0', edgecolor = 'black')  
        pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))  
    pylab.axis([-1.0, 26, -1.0, 26])  
    pylab.xticks([])  
    pylab.yticks([])  
     
    pylab.subplot(222)  
    Amin = matrix([20., 50., 80., 150., 200.])  
    W, H, x, y, w, h =  floorplan(Amin)  
    for k in range(5):  
        pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],  
                   [y[k], y[k]+h[k], y[k]+h[k], y[k]], 
                   facecolor = '#D0D0D0', edgecolor = 'black')  
        pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))  
    pylab.axis([-1.0, 26, -1.0, 26])  
    pylab.xticks([])  
    pylab.yticks([])  
     
    pylab.subplot(223)  
    Amin = matrix([180., 80., 80., 80., 80.])  
    W, H, x, y, w, h =  floorplan(Amin)  
    for k in range(5):  
        pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],  
                   [y[k], y[k]+h[k], y[k]+h[k], y[k]], 
                   facecolor = '#D0D0D0', edgecolor = 'black')  
        pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))  
    pylab.axis([-1.0, 26, -1.0, 26])  
    pylab.xticks([])  
    pylab.yticks([])  
     
    pylab.subplot(224)  
    Amin = matrix([20., 150., 20., 200., 110.])  
    W, H, x, y, w, h =  floorplan(Amin)  
    for k in range(5):  
        pylab.fill([x[k], x[k], x[k]+w[k], x[k]+w[k]],  
                   [y[k], y[k]+h[k], y[k]+h[k], y[k]], 
                   facecolor = '#D0D0D0', edgecolor = 'black')  
        pylab.text(x[k]+.5*w[k], y[k]+.5*h[k], "%d" %(k+1))  
    pylab.axis([-1.0, 26, -1.0, 26])  
    pylab.xticks([])  
    pylab.yticks([])  
     
    pylab.show()
