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

    def count_levels(self, col1: str, col2: str = None):
            """
            Return counts for one or two string columns as a pandas DF.
            """
            if not self._is_string_type(col1):
                print(f"Column '{col1}' is numeric or not string.")
                return None
    
            if col2 is not None and not self._is_string_type(col2):
                print(f"Column '{col2}' is numeric or not string.")
                return None
    
            if col2 is None:
                result = (
                    self.df.groupBy(self._safe_col(col1).alias(col1))
                    .count()
                    .toPandas()
                )
                return result.sort_values(col1).reset_index(drop=True)

            return result.sort_values([col1, col2]).reset_index(drop=True)
