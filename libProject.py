import numpy as np
import tensorflow as tf
import io, re, unidecode
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def tokenizeText(listOfStentences):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='' , oov_token='<unk>')
    tokenizer.fit_on_texts(listOfStentences)
    tensor = tokenizer.texts_to_sequences(listOfStentences)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer

def loadAndCreateDataset(path, nombreLignes):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [['<start> ' + preprocessSentences(w) + ' <end>' for w in l.split('\t')]  for l in lines[:nombreLignes]]
    inputTensor, targetTensor = zip(*word_pairs) 
    inputTensor, inputVocab = tokenizeText(inputTensor)
    targetTensor, targetVocab = tokenizeText(targetTensor)
    inputSentencesSize = inputTensor.shape[1]
    targetSentencesSize = targetTensor.shape[1]
    return inputTensor, inputVocab, inputSentencesSize, targetTensor, targetVocab, targetSentencesSize

def preprocessSentences(lines):
  lines = re.sub(r"([?.!,¿])", r" \1 ", lines)
  lines = re.sub(r'[" "]+', " ", lines)
  lines = unidecode.unidecode(lines) 
  lines = re.sub(r"[^a-zA-Z?.!,¿]+", " ", lines)
  return lines.lower().strip()

def statsDataPreprocessed(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer):
    print(preproc_english_sentences[1])
    print(preproc_french_sentences[1])
    print("La phrase d'entrée la plus longue:", preproc_english_sentences.shape[1])
    print("La phrase de sortie la plus longue:", preproc_french_sentences.shape[1])
    print("Taille du vocabulaire d'entrée:", len(english_tokenizer.word_index))
    print("Taille du vocabulaire de sortie:", len(french_tokenizer.word_index))

def createDataset(preproc_english_sentences, preproc_french_sentences, batchSize, bufferSize):
    dataset = tf.data.Dataset.from_tensor_slices(
        (preproc_english_sentences, preproc_french_sentences)).shuffle(bufferSize)
    dataset = dataset.batch(batchSize, drop_remainder=True)
    english_batch, french_batch = next(iter(dataset))
    return english_batch, french_batch, dataset

def lossFunction(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def trainStep(inp, targ, enc_hidden, encoderModel, decoder, targetVocab, lossObject, optimizer, batchSize):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoderModel(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targetVocab.word_index['<start>']] * batchSize, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += lossFunction(targ[:, t], predictions, lossObject)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoderModel.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def evaluate(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):
  attention_plot = np.zeros((targetTensorTrain, inputTensorTrain))

  sentence = '<start> ' + preprocessSentences(sentence) + ' <end>'
  inputs = [inputVocab.word_index[i] if i in inputVocab.word_index   else inputVocab.word_index['<unk>'] for i in sentence.split(' ')]
  # inputs = [inputVocab.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=inputSize,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoderModel(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targetVocab.word_index['<start>']], 0)

  for t in range(int(targetSize)):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    # attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targetVocab.index_word[predicted_id] + ' '

    if targetVocab.index_word[predicted_id] == '<end>':
      return result, sentence#, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return  result, sentence#, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator((ticker.MultipleLocator(1)))
  ax.yaxis.set_major_locator((ticker.MultipleLocator(1)))

  plt.show()

def translate(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):
  # result, sentence, attention_plot = evaluate(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize,inputVocab, targetVocab, units, encoderModel, decoder)
  result, sentence = evaluate(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize,inputVocab, targetVocab, units, encoderModel, decoder)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))

  # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

def evaluateAPI(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):

  sentence = '<start> ' + preprocessSentences(sentence) + ' <end>'
  print(sentence)
  inputs = [inputVocab.word_index[i] if i in inputVocab.word_index else inputVocab.word_index['<unk>'] for i in sentence.split(' ')]
  # inputs = [inputVocab.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=inputSize,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoderModel(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targetVocab.word_index['<start>']], 0)

  for t in range(targetTensorTrain):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targetVocab.index_word[predicted_id] + ' '

    if targetVocab.index_word[predicted_id] == '<end>':
      resultFind = ''
      for word in result.split(' '):
        if(word != '<end>'):
          resultFind += word + ' '
      return resultFind, sentence

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence

def translateAPI(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder):
  result, sentence = evaluateAPI(sentence, targetTensorTrain, inputTensorTrain, inputSize, targetSize, inputVocab, targetVocab, units, encoderModel, decoder)
  return result