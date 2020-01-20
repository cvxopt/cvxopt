"""
Convex optimization solvers.

conelp:   solves linear cone programs.
coneqp:   solves quadratic cone programs.
cp:       solves nonlinear convex problem.
cpl:      solves nonlinear convex problems with linear objectives.
gp:       solves geometric programs.
lp:       solves linear programs.
qp:       solves quadratic programs.
sdp:      solves semidefinite programs.
socp:     solves second-order cone programs.
options:  dictionary with customizable algorithm parameters.
"""

# Copyright 2012-2020 M. Andersen and L. Vandenberghe.
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

import cvxopt
from cvxopt.cvxprog import cp, cpl, gp 
from cvxopt.coneprog import conelp, lp, sdp, socp, coneqp, qp
options = {}
cvxopt.cvxprog.options = options
cvxopt.coneprog.options = options
__all__ = ['conelp', 'coneqp', 'lp', 'socp', 'sdp', 'qp', 'cp', 'cpl', 'gp']
