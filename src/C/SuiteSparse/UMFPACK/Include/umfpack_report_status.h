/* ========================================================================== */
/* === umfpack_report_status ================================================ */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License for License.                      */
/* -------------------------------------------------------------------------- */

void umfpack_di_report_status
(
    const double Control [UMFPACK_CONTROL],
    int status
) ;

void umfpack_dl_report_status
(
    const double Control [UMFPACK_CONTROL],
    SuiteSparse_long status
) ;

void umfpack_zi_report_status
(
    const double Control [UMFPACK_CONTROL],
    int status
) ;

void umfpack_zl_report_status
(
    const double Control [UMFPACK_CONTROL],
    SuiteSparse_long status
) ;

/*
double int Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    int status ;
    umfpack_di_report_status (Control, status) ;

double SuiteSparse_long Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    SuiteSparse_long status ;
    umfpack_dl_report_status (Control, status) ;

complex int Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    int status ;
    umfpack_zi_report_status (Control, status) ;

complex SuiteSparse_long Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    SuiteSparse_long status ;
    umfpack_zl_report_status (Control, status) ;

Purpose:

    Prints the status (return value) of other umfpack_* routines.

Arguments:

    double Control [UMFPACK_CONTROL] ;   Input argument, not modified.

	If a (double *) NULL pointer is passed, then the default control
	settings are used.  Otherwise, the settings are determined from the
	Control array.  See umfpack_*_defaults on how to fill the Control
	array with the default settings.  If Control contains NaN's, the
	defaults are used.  The following Control parameters are used:

	Control [UMFPACK_PRL]:  printing level.

	    0 or less: no output, even when an error occurs
	    1: error messages only
	    2 or more: print status, whether or not an error occurred
	    4 or more: also print the UMFPACK Copyright
	    6 or more: also print the UMFPACK License
	    Default: 1

    Int status ;			Input argument, not modified.

	The return value from another umfpack_* routine.
*/
