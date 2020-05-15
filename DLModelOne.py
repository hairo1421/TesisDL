
#################################################
# Modulos
#################################################
import pandas as pd
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import repeat
import time
#################################################
# Funciones
#################################################
def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

# muestra baches de serie
def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

# genera representaciones univariadas
def univariate_data(dataset, start_index, end_index, history_size , target_size):
  data = []
  labels = []

  start_index = start_index + history_size # 2
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index): # del 2 al 99
    indices = range(i-history_size, i) # 2 -0 = , al (0,2)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return
  np.array(data), np.array(labels)

# genera representaciones multivariadas
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step) # Step por si necesita predecir no el periodo
                                             # o saltando
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

# transforma series
def cpm(x, q = 1):
  index = np.zeros([len(x)-q, 2], dtype = int)
  index[:,0] = list(range(index.shape[0]))
  index[:,1] = list(range((q), (index.shape[0]+q)))

  xm  = np.zeros([1, index.shape[0]]).flatten()
  for i in range(len(x)-q):
    xm[i] = x.iloc[index[i,1]]/x.iloc[index[i,0]]*100-100
  return xm

# limpia igae
def PreProcesoIGAE(x):
    x = pd.read_csv(x, encoding='latin-1')
    x.index = x.Mes
    x = x.iloc[:,1:]
    return x.loc['2000/01':,:]

# imputa 2019/12 al 2020/01 media ultimos dos datos
def atipicos(x):
    x = x.apply(lambda x : x.astype(float))
    x = x.loc[:, (x==0).mean() < .40]
    x = x.apply(lambda y: y.fillna(np.nanmean(y[-5:-1]) ))
    x = x.loc[:'2020/01', ]
    x = x.iloc[:, 1:]
    return x

# normaliza seires
def normalizar(x):
    from sklearn.preprocessing import StandardScaler
    # Initialise the Scaler
    scaler = StandardScaler()
    # To scale data
    temp = scaler.fit_transform(x)
    return temp

# indexa fechas
def LastPreproces(x, y):
    x.index =  pd.DatetimeIndex(start = '2000-01', freq = 'M', periods = x.shape[0])
    x = x.replace([np.inf, -np.inf, np.nan], 0)
    return pd.concat([y.reset_index(), x.reset_index()], axis = 1).drop(['Mes', 'index'], axis = 1)

# limpia covariables
def PreProcesoX(x, y, z):
    diccionario = pd.read_csv(x, encoding='latin-1')
    covariables = pd.read_csv(y, encoding='latin-1').iloc[1:, :]
    covariables.index = covariables.iloc[:, 0]
    covariables = covariables.iloc[:,1:]
    covariables.columns = diccionario.COD_VAR.values.tolist()
    covariables = covariables.loc['1999/12':,]

    covariables = atipicos(covariables)

    new = diccionario[diccionario.Var == 'NEWS'].COD_VAR.values.tolist()
    noticias = normalizar(covariables.loc[:, [text for text in list(covariables.columns) if text  in new]])
    noticias = pd.DataFrame(noticias[1:,:])
    noticias.columns = new

    macro = covariables.loc[:, [text for text in list(covariables.columns) if text not in noticias]].apply(cpm, axis = 0)
    macro = macro.loc[:, [x for x in macro.columns if x not in ['AUTO_VENTA_DEP', 'AUTO_VENTA_LUX','AUTO_VENTA_AUTOBUS']]]
    nombres_macro = macro.columns
    macro = normalizar(macro)
    macro = pd.DataFrame(macro)
    macro.columns = nombres_macro

    covariables_st = pd.concat([macro, noticias], axis = 1 )
    dataset = LastPreproces(covariables_st, z)
    return dataset

#################################################
# Datos
#################################################
print('COMIENZA LA DESCARGA DE LOS DATOS')
import os
os.chdir('/home/hairomiranda/Desktop/git_workspace/TesisDL')
igae_file = 'IGAE.csv'
igae = PreProcesoIGAE(igae_file)
print('LISTO IGAE')

dicct = 'cat_all_vars_2.csv'
cov = 'BIE_all_vars_200320_2.csv'
dataset = PreProcesoX(dicct, cov, igae)
dataset = dataset.loc[:, ['IGAE'] + pd.read_csv('06expErroresPLS.csv').variables.values.tolist()]
print('LISTO X')
#################################################
# ATTENTION MECHANISM
#################################################


#################################################
# ENCODE
#################################################
def encodeAttention(x, u):
  tf.random.set_seed(16)
  np.random.seed(16)

  # Encoder
  encoder_input = tf.keras.Input(shape = (timestep,1), name = f'encode{x}') # Input
  encoder = tf.keras.layers.LSTM(u, return_sequences=True, return_state=True,  kernel_regularizer=l1(0.0001),dropout=0.0, recurrent_dropout=0.0, name = f'lstm{x}') # X LSTM

  #encoder = tf.keras.layers.Dropout(0.2)(encoder) # lo puse

  encoder_outputs, state_h, state_c = encoder(encoder_input) # x's, h's, c's
  W1 = tf.keras.layers.Dense(u)
  W2 = tf.keras.layers.Dense(u)
  V = tf.keras.layers.Dense(1)
  query_with_time_axis = tf.expand_dims(state_h, 1)
  score = V(tf.nn.tanh(
        W1(query_with_time_axis) + W2(encoder_outputs)))
  attention_weights = tf.nn.softmax(score, axis=1)
  context_vector = attention_weights * encoder_outputs
  context_vector = tf.reduce_sum(context_vector, axis=1)
  print('VA UN ENCODE')
  return tf.expand_dims(context_vector, -1) , encoder_input
#################################################
# DATA PROCESS
  #################################################
def train(x, temp):
    x_temp = temp[:,:,x].reshape(temp.shape[0],timestep,1)
    print('Encoder input train', x_temp.shape)
    return(x_temp)

def val(x, temp):
    x_temp = temp[:,:,x].reshape(temp.shape[0],timestep,1)
    print('Encoder input val', x_temp.shape)
    return(x_temp)    

def test(x, temp):
    x_temp = temp[:,:,x].reshape(temp.shape[0],timestep,1)
    print('Encoder input test', x_temp.shape)    
    return(x_temp)  
#################################################
from tensorflow.keras.regularizers import l1, l2, l1_l2
import time
#################################################
# HYPERPARAMETERS
H = 24
timestep = 3
k = 0
batch = 25
epocas = 70 
paciencia = 30
unidades_decode = 70 
unidades_encode = 50 
errores_horizonte = []
yreal_horizonte = []
yhat_horizonte = []

print('Aquí comienza el loop')


for i in range(H):
    t_inicial = time.time()


    # ENCODE LAYER K VARIABLES
    encode_modelos = []
    for j in range(dataset.shape[1]-1-k):
        encode_modelos.append(encodeAttention(j, unidades_encode))
        if j%50 == 0:
                  print('Modelo', j)

    modelos_atencion = [encode_modelos[x][0] for x in range(dataset.shape[1]-1-k) ]
    entrada_encode = [encode_modelos[x][1] for x in range(dataset.shape[1]-1-k) ]

    # MULTIVARIATE ATTENTION MECHANISM
    input_decode = tf.concat(modelos_atencion, axis=2, name = 'EntradaDecoder')

    # DECODER LAYER

    # me falta un input


    decoder_lstm = tf.keras.layers.LSTM(unidades_decode, return_sequences=False, kernel_regularizer=l1_l2(0.01), dropout=0.0,recurrent_dropout=0.0, name ='lstm_decoder') # Y LSTM'
    # kernel_regularizer=l1(0.001),
    # me falta un merge


    # INPUT MULTIVARIATE ATTENTION LAYER
    x = decoder_lstm(input_decode)          # vector pegarselo al input de aquí ¿cómo se lo inyecto?
    #x = tf.keras.layers.Dropout(0.2)(x) # lo puse

    # FULL CONECTION OUTPUT LAYER
    output = tf.keras.layers.Dense(1, activation= 'linear')(x)

    # MODEL
    model = tf.keras.models.Model(inputs= entrada_encode, outputs=output, )

    # COMPILE MODEL
    model.compile(loss='mse', optimizer='adam')


    #model.load_weights(f'Pesos{i}_iniciales.h5')
    #model.save_weights(f'Pesos{i}.h5', overwrite=True)


    import sklearn.metrics as me
    import math
    import time
    # numero de baches que se predicen depende del time teep
    t_inicial2 = time.time()
    size = len(igae.values.flatten())  - (H + timestep) + i # predicciones a dos año; i.e 24 meses
    TRAIN_SPLIT = size
    tf.random.set_seed(16)
    print("Train split:", TRAIN_SPLIT)

    #dataset = dataset.iloc[:241,:]

    past_history = timestep # equivalente a un VAR(5)
    future_target = 0

    STEP = 1

    x_train_multivariate_o, y_train_multivariate_o = multivariate_data(dataset.values, dataset.values[:, 0], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)


    x_test_mult, y_test_mult = multivariate_data(dataset.values, dataset.values[:, 0],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

    frac = int(x_train_multivariate_o.shape[0]*.9) # validación

    x_train_multivariate = x_train_multivariate_o[:frac,:,:]
    y_train_multivariate = y_train_multivariate_o[:frac]

    x_train_multivariate_v = x_train_multivariate_o[frac:,:,:]
    y_train_multivariate_v = y_train_multivariate_o[frac:]

    x_tf_train = list(map(train, range(dataset.shape[1]-k), repeat(x_train_multivariate)))

    x_tf_v = list(map(val, range(dataset.shape[1]-k), repeat(x_train_multivariate_v)))

    x_tf_test = list(map(test, range(dataset.shape[1]-k), repeat(x_test_mult[:1,:,:])))

    from tensorflow.keras.callbacks import EarlyStopping #5:28 a 5:45
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import CSVLogger
    print('Comienza el entrenamiento para h=', i)
    # Early Stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience= paciencia, restore_best_weights=True)
    csv_logger = CSVLogger(f'training{i}.log', separator=',', append=False)
    tf.random.set_seed(16)
    np.random.seed(16) 

    history = model.fit(x_tf_train, y_train_multivariate,  epochs= epocas, batch_size= batch,  callbacks=[es,csv_logger], verbose=1,  validation_data = (x_tf_v, y_train_multivariate_v ), 
                        shuffle=False) 

    import os
    os.chdir('/home/hairomiranda/Desktop/git_workspace/TesisDL/Train')
    print('guardando')
    model.save_weights(f'Pesos{i}_entrenados.h5', overwrite=True)
    print('guardado')



    # Predict test data 
    test_predict = model.predict(x_tf_test) 



    errores_horizonte.append(math.sqrt(me.mean_squared_error([y_test_mult[0]], [test_predict.flatten()[0]])))
    yreal_horizonte.append(y_test_mult[0])
    yhat_horizonte.append(test_predict.flatten()[0] )
    del model, modelos_atencion, entrada_encode, encode_modelos

    print('error:', errores_horizonte )
    print('real:', yreal_horizonte)
    print('hat:', yhat_horizonte)
    print('Pasamos al siguiente h...')
    t_avance2 = time.time()
    print(str(t_avance2-t_inicial2) + ' ' + 'Segundos')  
    pd.DataFrame({'rmse':errores_horizonte, 'y_real':yreal_horizonte, 'y_hat':yhat_horizonte}).to_csv(f'Entrenar{i}.csv')

t_avance = time.time()
print(str(t_avance-t_inicial) + ' ' + 'Segundos Fin...')
print(str(t_avance-t_inicial) + ' ' + 'Segundos Fin...')
