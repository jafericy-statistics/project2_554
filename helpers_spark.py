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
