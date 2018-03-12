import tensorflow as tf
from src.loader import PredictionModel
from scipy.misc import imread
import os

if __name__ == '__main__':

    model_dir = './model/Model.pb'
    path = '/Users/liba2/Desktop/size'
    result_path = './'

    result = ''
    newInput_X=tf.placeholder(tf.float32,[None,None,1],name='Placeholder')
    with open(model_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # initTable = tf.import_graph_def(graph_def,return_elements=['init_all_tables'])

        output,output2 ,initTable= tf.import_graph_def(graph_def,
                                     input_map={'Placeholder:0': newInput_X},
                                     return_elements=['deep_bidirectional_lstm/raw_prediction:0','code2str_conversion/output:0','init_all_tables'])



    with tf.Session() as sess:
        # print(tf.get_default_graph())

        sess.run(initTable)

        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.find('.jpg')>0:
                    print(filename)

                    image = imread(os.path.join(dirpath, filename), mode='L')[: ,:, None]
                    print(image)
                    predictions = sess.run(output, feed_dict={'Placeholder:0': image})

                    predictionss = sess.run(output2, feed_dict={'Placeholder:0': image})

                    transcription = predictions
                    print(str(transcription))
                    print(str(predictionss))
                    result = result + filename + ' : ' + str(transcription) + '\t\r'

    f = open(os.path.join(result_path, 'transcription.txt'), 'a')
    f.write(result)
    f.close()