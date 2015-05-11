/* ========================================================================== */
/* === UMF_free ============================================================= */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License for License.                      */
/* -------------------------------------------------------------------------- */

/*
    Free a block previously allocated by UMF_malloc and return NULL.
    Usage is p = UMF_free (p), to ensure that we don't free it twice.
    Also maintains the UMFPACK malloc count.
*/

#include "umf_internal.h"
#include "umf_free.h"

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
#include "umf_malloc.h"
#endif

GLOBAL void *UMF_free
(
    void *p
)
{
    DEBUG0 (("UMF_free: "ID"\n", (Int) p)) ;
    if (p)
    {

	/* see AMD/Source/amd_global.c for the memory allocator selection */
	amd_free (p) ;

#if defined (UMF_MALLOC_COUNT) || !defined (NDEBUG)
	/* One more object has been free'd.  Keep track of the count. */
	/* (purely for sanity checks). */
	UMF_malloc_count-- ;
	DEBUG0 (("     new malloc count: "ID"\n", UMF_malloc_count)) ;
#endif

    }

    return ((void *) NULL) ;
}
