from pyspark.sql import dataframe, SparkSession, DataFrameReader
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

from data_filters import movies_filter, ratings_filter, users_filter


class MoviesAnalysis:
    READ_FORMAT = 'csv'
    READ_DELIMITER = '::'

    def __init__(self, spark):
        self.spark = spark

    @staticmethod
    def write_to_parquet(dataframe: dataframe, filename: str):
        try:
            dataframe.write.mode('overwrite').parquet(filename)
        except Exception as e:
            print('Could not write to output path due to: ', str(e))
            raise

    def read_from_parquet(self, filename: str) -> DataFrameReader:
        return self.spark.read.parquet(filename)

    @staticmethod
    def data_join(df1: dataframe, df2: dataframe, key: str):
        assert key in df1.columns and key in df2.columns, \
               f'The key column {key} is not present in both dataframes'
        assert df1.schema[key].dataType == df2.schema[key].dataType, \
               f'{key} columns in the dataframes have different data types'
        return df1.join(df2, key)

    def run_analysis(self, file_path: str, output_path: str):
        movies, ratings, users = self.load_datas(file_path)

        # Write to parquet
        self.write_to_parquet(movies, os.path.join(output_path, 'movies.parquet'))
        self.write_to_parquet(ratings, os.path.join(output_path, 'ratings.parquet'))
        self.write_to_parquet(users, os.path.join(output_path, 'users.parquet'))

        # Read from parquet
        movies = self.read_from_parquet(os.path.join(output_path, 'movies.parquet'))
        ratings = self.read_from_parquet(os.path.join(output_path, 'ratings.parquet'))
        users = self.read_from_parquet(os.path.join(output_path, 'users.parquet'))

        # Data filters
        movies = movies_filter(movies)
        ratings = ratings_filter(ratings)
        users = users_filter(users)

        # Joining and aggregating data
        movie_ratings = self.data_join(movies, ratings, 'MovieID')
        movie_ratings_agg = movie_ratings.groupby('MovieID').agg(
            F.max('Rating').alias('Max_rating'),
            F.min('Rating').alias('Min_rating'),
            F.avg('Rating').alias('Avg_rating'),
        )
        movie_ratings_info = self.data_join(movies, movie_ratings_agg, 'MovieID')
        self.write_to_parquet(movie_ratings_info, os.path.join(output_path, 'movie_ratings_info.parquet'))

        ratings_with_avg = self.data_join(ratings, movie_ratings_info, 'MovieID')
        window = Window.partitionBy(ratings_with_avg['UserID']).orderBy(F.col('Rating').desc(),
                                                                        F.col('Avg_rating').desc())
        user_top_movies = ratings_with_avg.withColumn('Rank', F.row_number().over(window)).filter(F.col('Rank') <= 3)

        user_top_movies = self.data_join(users, user_top_movies, 'UserID')
        result = user_top_movies.select('UserID', 'MovieID', 'Title', 'Rating', 'Avg_rating', 'Rank')

        self.write_to_parquet(result, os.path.join(output_path, 'user_top_movies.parquet'))

    def load_data(self, filename: str) -> dataframe:
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'The file {filename} does not exist.')

        return self.spark.read.format(self.READ_FORMAT).option('delimiter', self.READ_DELIMITER).load(filename)

    def load_datas(self, file_path) -> (dataframe, dataframe, dataframe):
        movies = self.load_data(os.path.join(file_path, 'movies.dat'))\
            .withColumnRenamed('_c0', 'MovieID')\
            .withColumnRenamed('_c1', 'Title')\
            .withColumnRenamed('_c2', 'Genres')
        ratings = self.load_data(os.path.join(file_path, 'ratings.dat'))\
            .withColumnRenamed('_c0', 'UserID')\
            .withColumnRenamed('_c1', 'MovieID')\
            .withColumnRenamed('_c2', 'Rating')\
            .withColumnRenamed('_c3', 'Timestamp')
        users = self.load_data(os.path.join(file_path, 'users.dat'))\
            .withColumnRenamed('_c0', 'UserID')\
            .withColumnRenamed('_c1', 'Gender')\
            .withColumnRenamed('_c2', 'Age')\
            .withColumnRenamed('_c3', 'Occupation')\
            .withColumnRenamed('_c4', 'Zip-Code')

        return movies, ratings, users


if __name__ == '__main__':
    spark_session = SparkSession.builder.appName('Movie_Analysis').getOrCreate()

    analysis = MoviesAnalysis(spark_session)
    analysis.run_analysis('ml-1m/', 'output/')
