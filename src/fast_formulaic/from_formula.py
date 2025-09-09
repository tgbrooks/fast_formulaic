from formulaic import Formula
from formulaic.model_spec import ModelSpec
import scipy.sparse
from .materializer import FastFormulaicMaterializer


def from_formula(
    formula: str | Formula,
    data,
) -> scipy.sparse.csc_matrix:
    """
    Transform a narwhals (polars, pandas, etc) data frame to a sparse matrix using a Wilkinson formula.

    Parameters
    ----------
    formula: str
        A formula accepted by formulaic.
    data: pd.DataFrame
        pandas data frame to be converted.
    """
    spec = ModelSpec(
        formula=Formula(
            formula,
        ),
        output="sparse",
    )
    materializer = FastFormulaicMaterializer(
        data,
    )
    result = materializer.get_model_matrix(spec)

    return result
