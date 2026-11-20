# Legacy data fixtures

These small synthetic files were generated with NumPy 1.26.4, pandas 2.3.3,
and PyArrow 23.0.1. They contain one talk (`id0="talk"`,
`Speaker="Legacy speaker"`) and an embedding `[1.0, 0.0]`.

`numpy1-pandas2.pkl` stores the dataframe and embedding with pickle protocol 4;
`pyarrow23.parquet` stores the same dataframe without its index. The regression
test loads them using the upgraded libraries to verify backward compatibility
with data saved before this migration. They contain no production data.
