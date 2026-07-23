from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc
from pyscipopt.scip cimport Model, Row, SCIP_OKAY, SCIP_RETCODE, SCIP_ROW


cdef extern from *:
    """
    #include "scip/scip.h"
    #include "scip/scip_cutsel.h"
    #include "scip/pub_cutsel.h"
    #include "scip/cutsel_hybrid.h"

    typedef struct V2_HybridCutselData
    {
       SCIP_RANDNUMGEN* randnumgen;
       SCIP_Real goodscore;
       SCIP_Real badscore;
       SCIP_Real objparalweight;
       SCIP_Real efficacyweight;
       SCIP_Real dircutoffdistweight;
       SCIP_Real intsupportweight;
       SCIP_Real minortho;
       SCIP_Real minorthoroot;
    } V2_HYBRIDCUTSELDATA;

    static SCIP_RETCODE v2SelectCutsHybrid(
       SCIP* scip,
       SCIP_ROW** cuts,
       int ncuts,
       SCIP_ROW** forcedcuts,
       int nforcedcuts,
       SCIP_Bool root,
       int maxnselectedcuts,
       int* nselectedcuts
       )
    {
       SCIP_CUTSEL* cutsel;
       V2_HYBRIDCUTSELDATA* data;
       SCIP_Real minortho;
       SCIP_Real maxparall;
       SCIP_Real goodmaxparall;

       cutsel = SCIPfindCutsel(scip, "hybrid");
       if( cutsel == NULL )
          return SCIP_PLUGINNOTFOUND;

       data = (V2_HYBRIDCUTSELDATA*) SCIPcutselGetData(cutsel);
       if( data == NULL || data->randnumgen == NULL )
          return SCIP_INVALIDDATA;

       minortho = root ? data->minorthoroot : data->minortho;
       maxparall = 1.0 - minortho;
       goodmaxparall = MAX(0.5, maxparall);

       return SCIPselectCutsHybrid(
          scip, cuts, forcedcuts, data->randnumgen,
          data->goodscore, data->badscore, goodmaxparall, maxparall,
          data->dircutoffdistweight, data->efficacyweight,
          data->objparalweight, data->intsupportweight,
          ncuts, nforcedcuts, maxnselectedcuts, nselectedcuts
       );
    }
    """

    SCIP_RETCODE v2SelectCutsHybrid(
        void* scip,
        SCIP_ROW** cuts,
        int ncuts,
        SCIP_ROW** forcedcuts,
        int nforcedcuts,
        unsigned int root,
        int maxnselectedcuts,
        int* nselectedcuts,
    )


def model_pointer(Model model):
    """Return the opaque SCIP pointer owned by a PySCIPOpt Model."""
    return <uintptr_t>model._scip


def row_pointer(Row row):
    """Return the opaque SCIP_ROW pointer owned by a PySCIPOpt Row."""
    return <uintptr_t>row.scip_row


def select_hybrid_pointers(
    Model model,
    list cuts,
    list forcedcuts,
    bint root,
    int maxnselectedcuts,
):
    """Call SCIP 10.0.2 hybrid selection and return sorted opaque row pointers."""
    cdef int ncuts = len(cuts)
    cdef int nforcedcuts = len(forcedcuts)
    cdef SCIP_ROW** c_cuts = NULL
    cdef SCIP_ROW** c_forcedcuts = NULL
    cdef int nselectedcuts = 0
    cdef int index
    cdef Row row
    cdef SCIP_RETCODE retcode

    if ncuts <= 0:
        raise ValueError("cuts must be nonempty")
    if maxnselectedcuts < 0:
        raise ValueError("maxnselectedcuts must be nonnegative")

    c_cuts = <SCIP_ROW**>malloc(ncuts * sizeof(SCIP_ROW*))
    if c_cuts == NULL:
        raise MemoryError()
    if nforcedcuts > 0:
        c_forcedcuts = <SCIP_ROW**>malloc(nforcedcuts * sizeof(SCIP_ROW*))
        if c_forcedcuts == NULL:
            free(c_cuts)
            raise MemoryError()

    try:
        for index in range(ncuts):
            row = cuts[index]
            c_cuts[index] = row.scip_row
        for index in range(nforcedcuts):
            row = forcedcuts[index]
            c_forcedcuts[index] = row.scip_row

        retcode = v2SelectCutsHybrid(
            <void*>model._scip,
            c_cuts,
            ncuts,
            c_forcedcuts,
            nforcedcuts,
            <unsigned int>root,
            maxnselectedcuts,
            &nselectedcuts,
        )
        if retcode != SCIP_OKAY:
            raise RuntimeError(
                f"SCIPselectCutsHybrid returned SCIP_RETCODE {retcode}"
            )
        return (
            [<uintptr_t>c_cuts[index] for index in range(ncuts)],
            nselectedcuts,
        )
    finally:
        free(c_forcedcuts)
        free(c_cuts)
