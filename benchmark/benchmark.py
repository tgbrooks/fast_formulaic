import formulaic
import fast_formulaic
import time
import polars as pl


N = 100 * 1000 * 10 * 10
df1 = pl.DataFrame(
    {
        "cat1": [f"cat1_{i}" for i in range(100)] * (N // 100),
        "cat2": [f"cat2_{i}" for i in range(10000)] * (N // 10000),
    }
)


def timeit(name):
    def inner(func):
        def newfunc():
            start = time.time()
            func()
            end = time.time()
            return pl.DataFrame({"name": name, "time": end - start})

        return newfunc

    return inner


@timeit("ff: ~0+cat1:cat2")
def ff_cat1cat2():
    fast_formulaic.from_formula("~ 0 + cat2:cat1", df1)


@timeit("f: ~0+cat1:cat2")
def f_cat1cat2():
    spec = formulaic.ModelSpec(
        "~ 0 + cat2:cat1",
        materializer=formulaic.materializers.NarwhalsMaterializer,
        output="sparse",
    )
    formulaic.model_matrix(spec, df1, output="sparse")


funcs = [
    ff_cat1cat2,
    f_cat1cat2,
]

output = pl.concat([func() for func in funcs])

print(output)
