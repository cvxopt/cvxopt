/* ========================================================================== */
/* === UMFPACK_report_status ================================================ */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License for License.                      */
/* -------------------------------------------------------------------------- */

/*
    User-callable.  Prints the return value from other UMFPACK_* routines.
    See umfpack_report_status.h for details.
*/

#include "umf_internal.h"

GLOBAL void UMFPACK_report_status
(
    const double Control [UMFPACK_CONTROL],
    Int status
)
{
    Int prl ;

    /* ---------------------------------------------------------------------- */
    /* get control settings and status to determine what to print */
    /* ---------------------------------------------------------------------- */

    prl = GET_CONTROL (UMFPACK_PRL, UMFPACK_DEFAULT_PRL) ;

    if (prl < 1)
    {
	/* no output generated if prl is less than 1 */
	return ;
    }

    if (status == UMFPACK_OK && prl <= 1)
    {
	/* no output generated if prl is 1 or less and no error occurred. */
	/* note that the default printing level is 1. */
	return ;
    }

    /* ---------------------------------------------------------------------- */
    /* print umfpack license, copyright, version, and status condition */
    /* ---------------------------------------------------------------------- */

    PRINTF  (("\n")) ;
    PRINTF4 (("%s\n", UMFPACK_COPYRIGHT)) ;
    PRINTF6 (("%s", UMFPACK_LICENSE_PART1)) ;
    PRINTF6 (("%s", UMFPACK_LICENSE_PART2)) ;
    PRINTF6 (("%s", UMFPACK_LICENSE_PART3)) ;
    PRINTF  (("UMFPACK V%d.%d.%d (%s): ", UMFPACK_MAIN_VERSION,
	UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION, UMFPACK_DATE)) ;

    switch (status)
    {
	case UMFPACK_OK:
	    PRINTF (("OK\n")) ;
	    break ;

	case UMFPACK_WARNING_singular_matrix:
	    PRINTF (("WARNING: matrix is singular\n")) ;
	    break ;

	case UMFPACK_ERROR_out_of_memory:
	    PRINTF (("ERROR: out of memory\n")) ;
	    break ;

	case UMFPACK_ERROR_invalid_Numeric_object:
	    PRINTF (("ERROR: Numeric object is invalid\n")) ;
	    break ;

	case UMFPACK_ERROR_invalid_Symbolic_object:
	    PRINTF (("ERROR: Symbolic object is invalid\n")) ;
	    break ;

	case UMFPACK_ERROR_argument_missing:
	    PRINTF (("ERROR: required argument(s) missing\n")) ;
	    break ;

	case UMFPACK_ERROR_n_nonpositive:
	    PRINTF (("ERROR: dimension (n_row or n_col) must be > 0\n")) ;
	    break ;

	case UMFPACK_ERROR_invalid_matrix:
	    PRINTF (("ERROR: input matrix is invalid\n")) ;
	    break ;

	case UMFPACK_ERROR_invalid_system:
	    PRINTF (("ERROR: system argument invalid\n")) ;
	    break ;

	case UMFPACK_ERROR_invalid_permutation:
	    PRINTF (("ERROR: invalid permutation\n")) ;
	    break ;

	case UMFPACK_ERROR_different_pattern:
	    PRINTF (("ERROR: pattern of matrix (Ap and/or Ai) has changed\n")) ;
	    break ;

        case UMFPACK_ERROR_ordering_failed:
	    PRINTF (("ERROR: ordering failed\n")) ;
	    break ;

	case UMFPACK_ERROR_internal_error:
	    PRINTF (("INTERNAL ERROR!\n"
	    "Input arguments might be corrupted or aliased, or an internal\n"
	    "error has occurred.  Check your input arguments with the\n"
	    "umfpack_*_report_* routines before calling the umfpack_*\n"
	    "computational routines.  Recompile UMFPACK with debugging\n"
	    "enabled, and look for failed assertions.  If all else fails\n"
	    "please report this error to Tim Davis\n"
            "(DrTimothyAldenDavis@gmail.com).\n"
	    )) ;
	    break ;

	default:
	    PRINTF (("ERROR: Unrecognized error code: "ID"\n", status)) ;

    }
    PRINTF  (("\n")) ;
}
