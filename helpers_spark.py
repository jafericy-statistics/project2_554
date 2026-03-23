from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
import pandas as 
import os
import urllib.request

class SparkDataCheck:
    """
    Wrapper around Spark SQL DataFrame for simple
    vali and summ methods.
    """

    def __init__(self, dataframe: DataFrame):
        self.df = dataframe

    @classmethod
    def from_csv(cls, spark, path: str):
        """
        Instance of the class that reads CSV file with Spark.
        We can use this for local file but also urls on HTTP/HTTPS protos.
        """
        if path.startswith("http://") or path.startswith("https://"):
            local_path = os.path.basename(path)

            if not os.path.exists(local_path):
                urllib.request.urlretrieve(path, local_path)

            path = local_path

        df = spark.read.load(
            path,
            format="csv",
            sep=",",
            header=True
        )
        return cls(df)

@classmethod
    def from_pandas(cls, spark, pdf: pd.DataFrame):
        """
        Instance of the class from a standard pandas DF.
        """
        df = spark.createDataFrame(pdf)
        return cls(df)

    def _get_dtype_dict(self):
        return dict(self.df.dtypes)

    def _safe_col(self, col: str):
        """
        Reference a Spark column, including names with special characters and others.
        """
        return F.col(f"`{col}`")

    def _is_numeric_type(self, col: str) -> bool:
        dtype = self._get_dtype_dict().get(col)

        numeric_prefixes = (
            "int", "bigint", "smallint", "tinyint",
            "float", "double", "long", "decimal"
        )
        return any(dtype.startswith(prefix) for prefix in numeric_prefixes)

    def _is_string_type(self, col: str) -> bool:
        return self._get_dtype_dict().get(col) == "string"
