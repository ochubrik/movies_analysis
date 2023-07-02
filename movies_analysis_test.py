import pytest
from pyspark.sql import SparkSession

from data_filters import movies_filter, ratings_filter, users_filter
from movies_analysis import MoviesAnalysis


@pytest.fixture(scope='module')
def spark():
    return SparkSession.builder \
        .master('local[2]') \
        .appName('pytest-pyspark-local-testing') \
        .getOrCreate()


@pytest.fixture(scope='module')
def movie_analysis(spark):
    return MoviesAnalysis(spark)


def test_data_join(movie_analysis, spark):
    df1 = spark.createDataFrame([(1, 'a'), (2, 'b')], ['id', 'value'])
    df2 = spark.createDataFrame([(1, 'c'), (2, 'd')], ['id', 'other_value'])
    result = movie_analysis.data_join(df1, df2, 'id')

    assert result.count() == 2
    assert set(result.columns) == {'id', 'value', 'other_value'}


def test_movies_filter(spark):
    movies = spark.createDataFrame(
        [(1, 'a'), (2, 'b'), (1, 'c'), (None, 'd')],
        ['MovieID', 'Title']
    )

    result = movies_filter(movies)

    assert result.count() == 2
    assert result.filter(result.MovieID.isNull()).count() == 0


def test_ratings_filter(spark):
    ratings = spark.createDataFrame(
        [(1, 1, 4), (2, 2, 6), (3, None, 5), (None, 4, 4), (5, 5, 0)],
        ['UserID', 'MovieID', 'Rating']
    )

    result = ratings_filter(ratings)

    assert result.count() == 1
    assert result.filter(result.UserID.isNull()).count() == 0
    assert result.filter(result.MovieID.isNull()).count() == 0
    assert result.filter(~result.Rating.between(1, 5)).count() == 0


def test_users_filter(spark):
    users = spark.createDataFrame(
        [(1, 'M', 1, 1), (2, 'F', 18, 2), (3, 'X', 25, 3), (4, 'M', 30, 4), (5, 'F', 35, 22)],
        ['UserID', 'Gender', 'Age', 'Occupation']
    )

    result = users_filter(users)

    assert result.count() == 2
    assert result.filter(result.UserID.isNull()).count() == 0
    assert result.filter(~result.Gender.isin(['M', 'F'])).count() == 0
    assert result.filter(~result.Age.isin([1, 18, 25, 35, 45, 50, 56])).count() == 0
    assert result.filter(~result.Occupation.isin(list(range(21)))).count() == 0
