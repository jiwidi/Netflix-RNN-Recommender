# Netflix-RNN-Recommender
I create a RNN recommender based on google [work](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46488.pdf) for youtube recommendations we could try a similar approach. We try to predict next movie a user will watch based on its movie history. Netflix price was a explicit recommendation system as it had ratings for each movie, for this problem transform it into a implicit recommendation problem where watching a movie is the implicit feedback, no distintion between like/dislike movies.


## Data
We need to transform netflix data into an implicit format where we build a context feature containing movie_id for each user. Download the data from https://www.kaggle.com/netflix-inc/netflix-prize-data and uncompress it under `data/` then run process_netflix_data.py

## Model

The model is a very simple RNN network with embeddings for both users and movies.

```
RNN(
  (embeddings_user): Embedding(476422, 300)
  (embeddings_past): Embedding(17771, 300)
  (lstm): LSTM(300, 512, num_layers=3)
  (linear): Linear(in_features=812, out_features=17771, bias=True)
  (criterion): CrossEntropyLoss()
)
```

## Future work
This line of research has been continued by other teams by adding a hierarchical context [here](https://arxiv.org/pdf/1706.04148.pdf) and even adding attention layers to optimize it [here](https://www.ijcai.org/Proceedings/2018/0546.pdf). This could be an improvement on the model present here


