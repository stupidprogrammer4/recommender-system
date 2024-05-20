import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds


MOVIE_FEATURES = ['movie_title', 'movie_genres', 'movie_title_text']
USER_FEATURES = ['user_id', 'timestamp', 'bucketized_user_age']

"""
dsc: representaion of Users or queries
args:
    unique_user_ids: all unique users
    embedding_size: size of final embedding repr of users
    additional_features: additional features infos are used to improve model
"""
class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids, embedding_size=32, additional_features={}):
        super().__init__()
        self.additional_embeddings = {}

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_size)
        ])

        self.additional_embeddings['timestamp'] = tf.keras.Sequential([
            tf.keras.layers.Discretization(additional_features['timestamp_buckets'].tolist()),
            tf.keras.layers.Embedding(len(additional_features['timestamp_buckets']) + 1, embedding_size),
        ])

        self.user_age_normalizer = tf.keras.layers.Normalization(axis=None)
        self.user_age_normalizer.adapt(additional_features['bucketized_user_age'])
        self.additional_embeddings['bucketized_user_age'] = tf.keras.Sequential([self.user_age_normalizer,
                                                                                     tf.keras.layers.Reshape([1])])

    def call(self, inputs):
        return tf.concat([self.user_embedding(inputs['user_id'])] +
                         [self.additional_embeddings[k](inputs[k]) for k in self.additional_embeddings],
                         axis=1)
        

"""
dsc: representaion of Movies or candidates
args:
    unique_movie_titles: all unique movies
    additional_features: additional features infos are used to improve model
    embedding_size: size of final embedding repr of movies
"""
class MovieModel(tf.keras.Model):
    def __init__(self, unique_movie_titles, additional_features={}, embedding_size=32):
        super().__init__()
        self.additional_embeddings = {}

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_size)
        ])


        self.additional_embeddings['movie_genres'] = tf.keras.Sequential([
            tf.keras.layers.Embedding(max(additional_features['unique_movie_genres']) + 1, embedding_size),
            tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))
        ])

        max_tokens = 10_000
        self.title_vectorizer = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
        self.title_vectorizer.adapt(unique_movie_titles)
        self.additional_embeddings['movie_title_text'] = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, embedding_size, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])


    def call(self, inputs):
        return tf.concat([self.title_embedding(inputs['movie_title'])] +
                         [self.additional_embeddings[k](inputs[k]) for k in self.additional_embeddings],
                         axis=1)
        

"""
dsc: this class is used to represend query or candidate tower
args:
    layer_sizes: size of each dense layer of query or candidate tower
    embedding_model: user model for query and item model for candidate
"""
class QueryCandidateModel(tf.keras.Model):
    def __init__(self, layer_sizes, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(layer_sizes[-1]))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


"""
dsc: implementaion of Two Tower Neural Network
args:
    layer_sizes: size of each dens layers
    movies: candidate dataset
    unique_movie_titles: all unique movies
    n_unique_user_ids: all unique user ids
    additional_features: additional features infos are used in towers
"""
class TwoTowerModel(tfrs.models.Model):
    def __init__(self, layer_sizes, movies, unique_movie_titles, 
                 n_unique_user_ids, embedding_size, additional_features):
        super().__init__()
      
        self.additional_features = additional_features
        self.query_model = QueryCandidateModel(layer_sizes, UserModel(n_unique_user_ids,
                                                                      embedding_size=embedding_size,
                                                                      additional_features=self.additional_features))
        
        self.candidate_model = QueryCandidateModel(layer_sizes, MovieModel(unique_movie_titles,
                                                                           embedding_size=embedding_size,
                                                                           additional_features=self.additional_features))
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=(movies
                            .apply(tf.data.experimental.dense_to_ragged_batch(128))
                            .map(self.candidate_model)),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model({
            'user_id': features['user_id'],
            'bucketized_user_age': features['bucketized_user_age'],
            'timestamp': features['timestamp'],
        })
        movie_embeddings = self.candidate_model({
            'movie_title': features['movie_title'],
            'movie_title_text': features['movie_title_text'],
            'movie_genres': features['movie_genres']
        })
        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)
