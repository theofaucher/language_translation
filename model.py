import tensorflow as tf

#Ceci est une classe enfant de Model de tensorflow
class Encoder(tf.keras.Model):
  #Constructeur
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__() #Appel du constructeur de la classe parent
    self.batch_sz = batch_sz #Nombre de batchs
    self.enc_units = enc_units #Taille d'un batch
    #Création d'une couche d'embedding
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    #Création d'une couche de cellule GRU
    #return_sequences: Permet de retourner la dernière sortie de la séquence de sortie
    #return_state: Retourne le dernier état en plus de la sortie
    #recurrent_initializer: Initialiseur pour la matrice des poids, utilisé pour la transformation linéaire
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  #Permet de mettre les hidden states à 0
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

#Ceci est une classe enfant de Layer de tensorflow
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()  #Appel du constructeur de la classe parent
    #Création de trois objects de type Dense 
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):

    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

#Ceci est une classe enfant de Model de tensorflow
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__() #Appel du constructeur de la classe parent
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    #Pour le mécanisme d'attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    #Permet de récupérer le vecteur de contexte et les poids de celui-ci
    context_vector, attention_weights = self.attention(hidden, enc_output)

    #x est la shape après être passé dans la couche d'Embedding
    x = self.embedding(x)
    #x est la shape après la concaténation de batch_size, 1, embedding_dim + hidden_size
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    #Permet de passer la concaténation dans le GRU
    output, state = self.gru(x)

    output = tf.reshape(output, (-1, output.shape[2]))
    
    x = self.fc(output)

    return x, state, attention_weights