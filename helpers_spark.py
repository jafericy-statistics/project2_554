from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
import pandas as pd


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
        import os
        import urllib.request

        # If the path is a web URL, download it locally first
        if path.startswith("http://") or path.startswith("https://"):
            local_path = os.path.basename(path)

            if not os.path.exists(local_path):
                urllib.request.urlretrieve(path, local_path)

            path = local_path

        df = spark.read.load(
            path,
            format="csv",
            sep=",",
            inferSchema=True,
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
        if dtype is None:
            return False

        numeric_prefixes = (
            "int", "bigint", "smallint", "tinyint",
            "float", "double", "long", "decimal"
        )
        return any(dtype.startswith(prefix) for prefix in numeric_prefixes)

    def _is_string_type(self, col: str) -> bool:
        return self._get_dtype_dict().get(col) == "string"

    def check_numeric_range(self, col: str, lower=None, upper=None, new_col: str = None):
        if lower is None and upper is None:
            print("At least one of lower or upper must be provided.")
            return self

        if not self._is_numeric_type(col):
            print(f"Column '{col}' is not numeric. No changes made.")
            return self

        if new_col is None:
            new_col = f"{col}_in_range"

        safe = self._safe_col(col)

        if lower is not None and upper is not None:
            condition = safe.between(lower, upper)
        elif lower is not None:
            condition = safe >= lower
        else:
            condition = safe <= upper

        self.df = self.df.withColumn(
            new_col,
            F.when(safe.isNull(), F.lit(None)).otherwise(condition)
        )
        return self

    def check_string_levels(self, col: str, levels, new_col: str = None):
        if not self._is_string_type(col):
            print(f"Column '{col}' is not string. No changes made.")
            return self

        if new_col is None:
            new_col = f"{col}_valid_level"

        safe = self._safe_col(col)

        self.df = self.df.withColumn(
            new_col,
            F.when(safe.isNull(), F.lit(None)).otherwise(safe.isin(levels))
        )
        return self

    def check_missing(self, col: str, new_col: str = None):
        if new_col is None:
            new_col = f"{col}_is_missing"

        self.df = self.df.withColumn(new_col, self._safe_col(col).isNull())
        return self

    def min_max(self, col: str = None, group_col: str = None):
        
        dtype_dict = self._get_dtype_dict()
        numeric_cols = [c for c in self.df.columns if self._is_numeric_type(c)]

        if col is not None:
            if col not in dtype_dict:
                print(f"Column '{col}' not found.")
                return None
            if not self._is_numeric_type(col):
                print(f"Column '{col}' is not numeric.")
                return None

            agg_exprs = [
                F.min(self._safe_col(col)).alias(f"{col}_min"),
                F.max(self._safe_col(col)).alias(f"{col}_max")
            ]

            if group_col is None:
                return self.df.agg(*agg_exprs).toPandas()
            else:
                grouped = (
                    self.df.groupBy(self._safe_col(group_col).alias(group_col))
                    .agg(*agg_exprs)
                )
                return grouped.toPandas()

        if len(numeric_cols) == 0:
            return None

        if group_col is None:
            agg_exprs = []
            for c in numeric_cols:
                agg_exprs.extend([
                    F.min(self._safe_col(c)).alias(f"{c}_min"),
                    F.max(self._safe_col(c)).alias(f"{c}_max")
                ])
            return self.df.agg(*agg_exprs).toPandas()

        grouped_pdfs = []
        for c in numeric_cols:
            temp = (
                self.df.groupBy(self._safe_col(group_col).alias(group_col))
                .agg(
                    F.min(self._safe_col(c)).alias(f"{c}_min"),
                    F.max(self._safe_col(c)).alias(f"{c}_max")
                )
                .toPandas()
            )
            grouped_pdfs.append(temp)

        return reduce(
            lambda left, right: pd.merge(left, right, on=group_col, how="outer"),
            grouped_pdfs
        )

    def count_levels(self, col1: str, col2: str = None):

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

        result = (
            self.df.groupBy(
                self._safe_col(col1).alias(col1),
                self._safe_col(col2).alias(col2)
            )
            .count()
            .toPandas()
        )
        return result.sort_values([col1, col2]).reset_index(drop=True)
