import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from model import neural_network
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'          #for windows
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():

    # tf flag
    flags = tf.flags
    flags.DEFINE_string("data_path", "D:/M1_lecture/lecture/image_analysis/no_normalized_data.csv", "data path")
    flags.DEFINE_string("outdir", "D:/M1_lecture/lecture/image_analysis/no_norm", "outdir path")
    flags.DEFINE_string("gpu_index", "0", "GPU-index")
    flags.DEFINE_integer("num_epoch", 1000, "number of iteration")
    flags.DEFINE_integer("kfold", 10, "number of fold")
    flags.DEFINE_list("input_size", [22], "input vector size")
    flags.DEFINE_bool("cv", True, "if 10 fold CV or not")
    FLAGS = flags.FLAGS

    # load train data
    data = np.loadtxt(FLAGS.data_path, delimiter=",").astype(np.float32)
    feature = data[:, :data.shape[1]-1]
    label = data[:, data.shape[1]-1]

    # initializer
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = utils.config(index=FLAGS.gpu_index)) as sess:
        # Resubstitution Method
        if FLAGS.cv == False:
            # set network
            kwargs = {
                'sess': sess,
                'input_size': FLAGS.input_size,
                'learning_rate': 1e-3
            }
            NN = neural_network(**kwargs)

            # print parmeters
            utils.cal_parameter()

            # check floder
            if not (os.path.exists(os.path.join(FLAGS.outdir, 'R', 'tensorboard'))):
                os.makedirs(os.path.join(FLAGS.outdir, 'R', 'tensorboard'))
            if not (os.path.exists(os.path.join(FLAGS.outdir, 'R', 'model'))):
                os.makedirs(os.path.join(FLAGS.outdir, 'R', 'model'))
            if not (os.path.exists(os.path.join(FLAGS.outdir, 'R', 'predict'))):
                os.makedirs(os.path.join(FLAGS.outdir, 'R', 'predict'))

            # prepare tensorboard
            writer_train = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'R', 'tensorboard', 'train'), sess.graph)
            writer_test = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'R', 'tensorboard', 'test'))
            value_loss = tf.Variable(0.0)
            tf.summary.scalar("loss", value_loss)
            merge_op = tf.summary.merge_all()

            # initialize
            sess.run(init_op)

            # training
            tbar = tqdm(range(FLAGS.num_epoch), ascii=True)
            for i in tbar:
                train_step, train_data = utils.batch_iter(feature, label, batch_size=feature.shape[0], shuffle=True)
                train_data_batch = next(train_data)

                train_loss = NN.update(train_data_batch[0], np.reshape(train_data_batch[1] , (train_data_batch[1].shape[0],1)))
                s = "Loss: {:.4f}".format(np.mean(train_loss))
                tbar.set_description(s)

                summary_train_loss = sess.run(merge_op, {value_loss: np.mean(train_loss)})
                writer_train.add_summary(summary_train_loss, i+1)

                NN.save_model(i+1, outdir=os.path.join(FLAGS.outdir, 'R'))

            # test
            test_loss_min = []
            sess.run(init_op)
            test_step, test_data = utils.batch_iter(feature, label, batch_size=feature.shape[0], shuffle=False)

            tbar = tqdm(range(FLAGS.num_epoch), ascii=True)
            for i in tbar:
                NN.restore_model(os.path.join(FLAGS.outdir, 'R', 'model', 'model_{}'.format(i+1)))
                test_data_batch = next(test_data)
                test_loss, predict = NN.test(test_data_batch[0], np.reshape(test_data_batch[1], (test_data_batch[1].shape[0], 1)))
                s = "Loss: {:.4f}".format(np.mean(test_loss))
                tbar.set_description(s)
                test_loss_min.append(np.mean(test_loss))

                summary_test_loss = sess.run(merge_op, {value_loss: np.mean(test_loss)})
                writer_test.add_summary(summary_test_loss, i+1)

                predict_label = np.zeros((test_data_batch[0].shape[0], 2))
                predict_label[:, 0] = test_data_batch[1]
                predict_label[:, 1] = np.where(predict > 0.5, 1, 0)[:, 0]
                np.savetxt(os.path.join(FLAGS.outdir, 'R', 'predict', 'result_{}.csv'.format(i + 1)),
                           predict_label.astype(np.int), delimiter=',', fmt="%d")

            test_loss_min = np.argmin(np.asarray(test_loss_min))
            np.savetxt(os.path.join(FLAGS.outdir, 'R' ,'min_loss_index.txt'), [test_loss_min+1], fmt="%d")

        # Leave-one-out method (10-fold CV)
        if FLAGS.cv == True:
            kfold = StratifiedKFold(n_splits=FLAGS.kfold, shuffle=True, random_state=1)
            fold_index = 0

            # set network
            kwargs = {
                'sess': sess,
                'input_size': FLAGS.input_size,
                'learning_rate': 1e-3
            }
            NN = neural_network(**kwargs)

            # print parmeters
            utils.cal_parameter()
            for train, test in kfold.split(feature, label):
                fold_index += 1

                # check folder
                if not (os.path.exists(os.path.join(FLAGS.outdir, 'L', str(fold_index) , 'tensorboard'))):
                    os.makedirs(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'tensorboard'))
                if not (os.path.exists(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'model'))):
                    os.makedirs(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'model'))
                if not (os.path.exists(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'predict'))):
                    os.makedirs(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'predict'))

                # prepare tensorboard
                writer_train = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'tensorboard', 'train'),
                                                     sess.graph)
                writer_test = tf.summary.FileWriter(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'tensorboard', 'test'))
                value_loss = tf.Variable(0.0)
                tf.summary.scalar("loss", value_loss)
                merge_op = tf.summary.merge_all()

                # initialize
                sess.run(tf.global_variables_initializer())

                # training
                tbar = tqdm(range(FLAGS.num_epoch), ascii=True)
                for i in tbar:
                    train_step, train_data = utils.batch_iter(feature[train], label[train],
                                                              batch_size=feature[train].shape[0], shuffle=True)
                    train_data_batch = next(train_data)

                    train_loss = NN.update(train_data_batch[0],
                                           np.reshape(train_data_batch[1], (train_data_batch[1].shape[0], 1)))
                    s = "Loss: {:.4f}".format(np.mean(train_loss))
                    tbar.set_description(s)

                    summary_train_loss = sess.run(merge_op, {value_loss: np.mean(train_loss)})
                    writer_train.add_summary(summary_train_loss, i + 1)

                    NN.save_model(i + 1, outdir=os.path.join(FLAGS.outdir, 'L', str(fold_index)))

                # test
                sess.run(init_op)
                test_loss_min = []
                test_step, test_data = utils.batch_iter(feature[test], label[test],
                                                        batch_size=feature[test].shape[0], shuffle=False)
                tbar = tqdm(range(FLAGS.num_epoch), ascii=True)
                for i in tbar:
                    NN.restore_model(os.path.join(FLAGS.outdir, 'L', str(fold_index) , 'model', 'model_{}'.format(i + 1)))
                    test_data_batch = next(test_data)
                    test_loss, predict = NN.test(test_data_batch[0],
                                                 np.reshape(test_data_batch[1], (test_data_batch[1].shape[0], 1)))
                    s = "Loss: {:.4f}".format(np.mean(test_loss))
                    tbar.set_description(s)
                    test_loss_min.append(np.mean(test_loss))

                    summary_test_loss = sess.run(merge_op, {value_loss: np.mean(test_loss)})
                    writer_test.add_summary(summary_test_loss, i + 1)

                    predict_label = np.zeros((test_data_batch[0].shape[0], 2))
                    predict_label[:, 0] = test_data_batch[1]
                    predict_label[:, 1] = np.where(predict > 0.5, 1, 0)[:,0]
                    np.savetxt(os.path.join(FLAGS.outdir, 'L', str(fold_index), 'predict', 'result_{}.csv'.format(i + 1)),
                               predict_label.astype(np.int), delimiter=',', fmt="%d")

                test_loss_min = np.argmin(np.asarray(test_loss_min))
                np.savetxt(os.path.join(FLAGS.outdir, 'L', str(fold_index),'min_loss_index.txt'), [test_loss_min+1], fmt="%d")

if __name__ == '__main__':
    main()