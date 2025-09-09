from __future__ import annotations

import functools
import itertools
from collections.abc import Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import narwhals.stable.v1 as nw
import numpy
import pandas
import scipy.sparse as spsparse
from interface_meta import override

from formulaic.model_matrix import ModelMatrix
from formulaic.parser.types import Term
from formulaic.utils.cast import as_columns
from formulaic.utils.null_handling import drop_rows as drop_nulls

from formulaic.materializers import NarwhalsMaterializer
from formulaic.materializers.base import EncodedTermStructure

import csc

if TYPE_CHECKING:  # pragma: no cover
    from formulaic.types import ScopedTerm
    from formulaic.model_spec import ModelSpec


class FastFormulaicMaterializer(NarwhalsMaterializer):
    REGISTER_NAME = "fast-formulaic"
    REGISTER_INPUTS: Sequence[str] = (
        "narwhals.DataFrame",
        "narwhals.stable.v1.DataFrame",
    )
    REGISTER_OUTPUTS: Sequence[str] = "sparse"
    REGISTER_PRECEDENCE = 200

    @override
    def _encode_categorical(
        self,
        values: Any,
        metadata: Any,
        encoder_state: dict[str, Any],
        spec: ModelSpec,
        drop_rows: Sequence[int],
        reduced_rank: bool = False,
    ) -> Any:
        # Even though we could reduce rank here, we do not, so that the same
        # encoding can be cached for both reduced and unreduced rank. The
        # rank will be reduced in the _encode_evaled_factor method.
        from formulaic.transforms import encode_contrasts

        if drop_rows:
            values = drop_nulls(values, indices=drop_rows)
        if nw.dependencies.is_narwhals_series(values):
            values = values.to_pandas()
            # TODO: do we need to use pandas here?

        return as_columns(
            encode_contrasts(
                values,
                reduced_rank=False,
                output="pandas" if spec.output == "narwhals" else spec.output,
                _metadata=metadata,
                _state=encoder_state,
                _spec=spec,
            )
        )

    @override
    def _get_columns_for_term(
        self, factors: list[dict[str, Any]], spec: ModelSpec, scale: float = 1
    ) -> tuple[Any, list[str]]:
        out = []

        names = [
            ":".join(reversed(product))
            for product in itertools.product(*reversed(factors))
        ]

        # Pre-multiply factors with only one set of values (improves performance)
        solo_factors = {}
        indices = []
        for i, factor in enumerate(factors):
            if len(factor) == 1:
                solo_factors.update(factor)
                indices.append(i)
        if solo_factors:
            for index in reversed(indices):
                factors.pop(index)
            if spec.output == "sparse":
                factors.append(
                    {
                        ":".join(solo_factors): functools.reduce(
                            spsparse.csc_matrix.multiply, solo_factors.values()
                        )
                    }
                )
            else:
                factors.append(
                    {
                        ":".join(solo_factors): functools.reduce(
                            numpy.multiply,
                            (numpy.asanyarray(p) for p in solo_factors.values()),
                        )
                    }
                )

        factor_mats = []
        for factor in factors:
            mat = spsparse.hstack(list(factor.values()), format="csc")
            mat.sort_indices()  # puts matrix into canonical form, as assumed by our Cython function
            factor_mats.append(mat)

        def _column_product(A, B):
            """Return matrix C whose columns are the products of each pair of columns from A and B"""
            # Operations like broadcasting to do this directly aren't available in scipy.sparse
            # So we perform this as a sequence of matrix-vector products stacked together
            # If A and B have very different column counts, then it's much faster to multiply
            # the bigger matrix times the column vectors of the other than vice-versa
            if A.shape[1] == 1 or B.shape[1] == 1:
                return A.multiply(B)
            else:
                assert A.has_canonical_format and B.has_canonical_format
                assert (
                    A.indices.dtype == numpy.int32 and B.indices.dtype == numpy.int32
                ), "Matrices are larger than is currently supported: at most 32-bit indices usable."
                print("using csc")  # TODO: remove print statement
                return csc.csc_column_product(A, B)

        out = scale * functools.reduce(_column_product, factor_mats)

        return names, out

    @override
    def _build_model_matrix(
        self, spec: ModelSpec, drop_rows: Sequence[int]
    ) -> ModelMatrix:
        print("start", spec.output)
        # Modified version of base._build_model_matrix
        # so that we don't have to separate out all columns

        # Step 0: Apply any requested column/term clustering
        # This must happen before Step 1 otherwise the greedy rank reduction
        # below would result in a different outcome than if the columns had
        # always been in the generated order.
        terms = self._cluster_terms(spec.formula, cluster_by=spec.cluster_by)

        # Step 1: Determine strategy to maintain structural full-rankness of output matrix
        # (reusing pre-generated structure if it is available)
        print(f"STEP1: {spec.output}")
        if spec.structure:
            scoped_terms_for_terms: Generator[
                tuple[Term, Iterable[ScopedTerm]], None, None
            ] = (
                (s.term, [st.rehydrate(self.factor_cache) for st in s.scoped_terms])
                for s in spec.structure
            )
        else:
            scoped_terms_for_terms = self._get_scoped_terms(
                terms,
                ensure_full_rank=spec.ensure_full_rank,
            )

        # Step 2: Generate the columns which will be collated into the full matrix
        print(f"STEP2: {spec.output}")
        cols = []
        for term, scoped_terms in scoped_terms_for_terms:
            scoped_cols = []
            for scoped_term in scoped_terms:
                if not scoped_term.factors:
                    scoped_cols.append(
                        (
                            ["Intercept"],
                            scoped_term.scale
                            * self._encode_constant(1, None, {}, spec, drop_rows),
                        )
                    )
                else:
                    scoped_cols.append(
                        self._get_columns_for_term(
                            [
                                self._encode_evaled_factor(
                                    scoped_factor.factor,
                                    spec,
                                    drop_rows,
                                    reduced_rank=scoped_factor.reduced,
                                )
                                for scoped_factor in scoped_term.factors
                            ],
                            spec=spec,
                            scale=scoped_term.scale,
                        )
                    )
            cols.append((term, scoped_terms, scoped_cols))

        # Step 3: Populate remaining model spec fields
        print(f"STEP3: {spec.output}")
        if spec.structure:
            cols = list(self._enforce_structure(cols, spec, drop_rows))
        else:
            spec = spec.update(
                structure=[
                    EncodedTermStructure(
                        term,
                        list(st.copy(without_values=True) for st in scoped_terms),
                        [name for name, col in scoped_cols],
                    )
                    for term, scoped_terms, scoped_cols in cols
                ],
            )

        # Step 4: Collate factors into one ModelMatrix
        return ModelMatrix(
            self._combine_columns(
                [
                    (name, values)
                    for term, scoped_terms, scoped_cols in cols
                    for name, values in scoped_cols
                ],
                spec=spec,
                drop_rows=drop_rows,
            ),
            spec=spec,
        )

    @override
    def _combine_columns(
        self, cols: Sequence[tuple[str, Any]], spec: ModelSpec, drop_rows: Sequence[int]
    ) -> pandas.DataFrame:
        # Special case no columns to empty csc_matrix, array, or DataFrame
        if not cols:
            values = numpy.empty((self.data.shape[0], 0))
            if spec.output == "sparse":
                return spsparse.csc_matrix(values)
            if spec.output == "narwhals":
                # TODO: This output type is inconsistent with the `.to_native()`
                # below.
                return nw.from_native(pandas.DataFrame(values), eager_only=True)
            if spec.output == "numpy":
                return values
            return pandas.DataFrame(values)

        # Otherwise, concatenate columns into model matrix
        if spec.output == "sparse":
            return spsparse.hstack([col[1] for col in cols], format="csc")

        # TODO: Can we do better than this? Having to reconstitute raw data
        # does not seem ideal.
        combined = nw.from_dict(
            {name: nw.to_native(col, pass_through=True) for name, col in cols},
            native_namespace=nw.get_native_namespace(self.__narwhals_data),
        )
        if spec.output == "narwhals":
            if nw.dependencies.is_narwhals_dataframe(self.data):
                return combined
            return combined.to_native()
        if spec.output == "pandas":
            df = combined.to_pandas()
            return df
        if spec.output == "numpy":
            return combined.to_numpy()
        raise ValueError(f"Invalid output type: {spec.output}")  # pragma: no cover
