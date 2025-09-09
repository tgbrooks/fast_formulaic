# fast-formulaic

A faster materializer for formulaic for the case where you have large numbers of levels (> 100) in categories as well as interactions.

## Installation

```
pip install git+git://github.com/tgbrooks/fast_formulaic.git
```

## Tutorial

``` python
import fast_formulaic
import polars as pl
df = pl.DataFrame({
    'cat1': ['a', 'b', 'a', 'b'],
    'cat2': ['c', 'c', 'd', 'd'],
})

model_matrix = fast_formulaic.from_formula("~ cat1*cat2", df)
```


## Alternatives

See the `tabmat` project, which has specialized, efficient matrix formats for categorical design matrices.
