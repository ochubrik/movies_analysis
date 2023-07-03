from pyspark.sql import dataframe


def movies_filter(movies: dataframe):
    return movies.dropDuplicates(['MovieID']).filter(movies.MovieID.isNotNull())


def ratings_filter(ratings: dataframe):
    return ratings.filter(
        ratings.UserID.isNotNull() &
        ratings.MovieID.isNotNull() &
        ratings.Rating.between(1, 5)
    )


def users_filter(users: dataframe):
    valid_genders = ['M', 'F']
    valid_ages = [1, 18, 25, 35, 45, 50, 56]
    valid_occupations = list(range(21))

    return users.filter(
        users.UserID.isNotNull() &
        users.Gender.isin(valid_genders) &
        users.Age.isin(valid_ages) &
        users.Occupation.isin(valid_occupations)
    )
