from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os
from datetime import datetime
import json
import multiprocessing
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorboard.plugins import projector

from skynet.DeepCoder.dsl.DeepCoder_program import Program
from skynet.DeepCoder.dsl.DeepCoder_value import Value
from skynet.DeepCoder.env.env import ProgramState
from skynet.DeepCoder.env.operator import Operator, operator_to_index, num_operators
from skynet.DeepCoder.env.statement import Statement, statement_to_index, num_statements
from skynet.DeepCoder.params import deepCoder_params as params
from skynet.PCCoder.model_tf.PCCoder import PCCoder
from skynet.dsl.example import Example
from skynet.env.env import ProgramEnv

learn_rate = 0.1
batch_size = 100
num_epochs = 40

test_iterator_size = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to data')
    parser.add_argument('output_path', type=str, help='Output path of trained model')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Optional limit to the dataset size (usually for debugging)')
    args = parser.parse_args()
    train(args)


def generate_prog_data(line):
    data = json.loads(line.rstrip())
    examples = Example.from_line(data, Value)
    env = ProgramEnv(examples, ProgramState)
    program = Program.parse(data['program'])

    inputs = []
    statements = []
    drop = []
    operators = []
    for i, statement in enumerate(program.statements):
        inputs.append(env.get_encoding())

        # Translate absolute indices to post-drop indices
        f, args = statement.function, list(statement.args)
        for j, arg in enumerate(args):
            if isinstance(arg, int):
                args[j] = env.real_var_idxs.index(arg)

        statement = Statement(f, args)
        statements.append(statement_to_index[statement])

        used_args = []
        for next_statement in program.statements[i:]:
            used_args += [x for x in next_statement.args if isinstance(x, int)]

        to_drop = []
        for j in range(params.max_program_vars):
            if j >= env.num_vars or env.real_var_idxs[j] not in used_args:
                to_drop.append(1)
            else:
                to_drop.append(0)

        drop.append(to_drop)

        operator = Operator.from_statement(statement)
        operators.append(operator_to_index[operator])

        if env.num_vars < params.max_program_vars:
            env.step(statement)
        else:
            # Choose a random var (that is not used anymore) to drop.
            env.step(statement, random.choice([j for j in range(len(to_drop)) if to_drop[j] > 0]))

    return inputs, statements, drop, operators


def load_data(fileobj, max_len):
    X = []
    Y = []
    Z = []
    W = []

    print("Loading dataset...")
    lines = fileobj.read().splitlines()
    if max_len is not None:
        lines = lines[:max_len]

    pool = multiprocessing.Pool()
    res = list(tqdm(pool.imap(generate_prog_data, lines), total=len(lines)))

    for input, target, to_drop, operators in res:
        X += input
        Y += target
        Z += to_drop
        W += operators

    return np.array(X), Y, Z, W


def dataset_shuffle(data, size):
    data, statement_target, drop_target, operator_target = data
    data, statement_target, drop_target, operator_target = data.numpy(), statement_target.numpy(), drop_target.numpy(), operator_target.numpy()
    shuffled_index = np.random.permutation(size)

    return tf.convert_to_tensor(data[shuffled_index]), tf.convert_to_tensor(
        statement_target[shuffled_index]), tf.convert_to_tensor(drop_target[shuffled_index]), tf.convert_to_tensor(
        operator_target[shuffled_index])


def butch_loader(data, dataset_size, batch_size=100):
    data, statement_target, drop_target, operator_target = data

    indices = tf.range(start=0, limit=dataset_size, dtype=tf.int32)

    while True:
        shuffled_indices = tf.random.shuffle(indices)

        batch_data = tf.gather(data, tf.slice(shuffled_indices, [0], [batch_size]))
        batch_statement_target = tf.gather(statement_target, tf.slice(shuffled_indices, [0], [batch_size]))
        batch_drop_target = tf.gather(drop_target, tf.slice(shuffled_indices, [0], [batch_size]))
        batch_operator_target = tf.gather(operator_target, tf.slice(shuffled_indices, [0], [batch_size]))

        yield batch_data, batch_statement_target, batch_drop_target, batch_operator_target


def train(args):
    with open(args.input_path, 'r') as f:
        data, statement_target, drop_target, operator_target = load_data(f, args.max_len)

    data, statement_target, drop_target, operator_target = tf.convert_to_tensor(data, dtype=tf.int64), \
                                                           tf.one_hot(statement_target,
                                                                      on_value=1.0,
                                                                      off_value=0.0,
                                                                      depth=num_statements), \
                                                           tf.convert_to_tensor(drop_target), \
                                                           tf.one_hot(operator_target, on_value=1.0, off_value=0.0,
                                                                      depth=num_operators)

    model = PCCoder.load('D:\KPI\diplom\skynet\PCCoder\\result\model_weights\\tf')
    model.save_config(args.output_path)
    log_dir = args.output_path + "\\logs\\fit"+ datetime.now().strftime("%Y%m%d-%H-%M-%S")
    model.set_tensorboard(log_dir)
    emb=model.inputs[0].layers['PCCoder_Input.Embedding_0']


    model.create_model()
    dataset_size = data.shape[0]
    print(f"dataset size :{dataset_size}")

    data, statement_target, drop_target, operator_target = dataset_shuffle(
        (data, statement_target, drop_target, operator_target), dataset_size)

    train_size = int(0.9 * dataset_size)
    train_data = tf.slice(data, [0, 0, 0, 0], [train_size, -1, -1, -1])
    train_statement_target = tf.slice(statement_target, [0, 0], [train_size, -1])
    train_drop_target = tf.slice(drop_target, [0, 0], [train_size, -1])
    train_operator_target = tf.slice(operator_target, [0, 0], [train_size, -1])

    test_data = tf.slice(data, [train_size, 0, 0, 0], [-1, -1, -1, -1])
    test_statement_target = tf.slice(statement_target, [train_size, 0], [-1, -1])
    test_drop_target = tf.slice(drop_target, [train_size, 0], [-1, -1])
    test_operator_target = tf.slice(operator_target, [train_size, 0], [-1, -1])

    train_dataloader = butch_loader(data=(train_data, train_statement_target, train_drop_target, train_operator_target),
                                    dataset_size=train_size, batch_size=batch_size)

    for epoch in range(num_epochs):
        print("Epoch %d" % epoch)

        statement_losses = []
        drop_losses = []
        operator_losses = []

        for i in tqdm(range(0, train_size, batch_size)):
            batch = next(train_dataloader)
            x = batch[0]
            y = batch[1]
            z = batch[2]
            w = batch[3]

            with tf.GradientTape() as tape:
                pred_act, pred_drop, pred_operator = model(x, training=True)

                loss_act, loss_drop, loss_operator = [model.model.loss[out.node.layer.name] for out in model.output]
                loss_act_value, loss_drop_value, loss_operator_value = loss_act(y, pred_act), loss_drop(z,
                                                                                                        pred_drop), loss_operator(
                    w, pred_operator)

                statement_losses.append(float(loss_act_value.numpy()))
                drop_losses.append(float(loss_drop_value.numpy()))
                operator_losses.append(float(loss_operator_value.numpy()))

                # Compute the loss value for this minibatch.
                loss_value = loss_act_value + loss_drop_value + loss_operator_value
                with model.summary_writer.as_default():
                    with tf.name_scope('train'):
                        step =(train_size * epoch) + i
                        tf.summary.scalar("train_loss", data=loss_value, step=step)
                        tf.summary.scalar("train_loss_act", data=loss_act_value, step=step)
                        tf.summary.scalar("train_loss_drop", data=loss_drop_value, step=step)
                        tf.summary.scalar("train_loss_operator", data=loss_operator_value, step=step)
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            model.model.optimizer.apply_gradients(zip(grads, model.model.trainable_weights))

        avg_statement_train_loss = np.array(statement_losses).mean()
        avg_drop_train_loss = np.array(drop_losses).mean()
        avg_operator_train_loss = np.array(operator_losses).mean()

        # Iterate through test set to avoid out of memory issues
        statement_pred, drop_pred, operator_pred = [], [], []
        for i in range(0, dataset_size - train_size, test_iterator_size):
            if (i + test_iterator_size) > (dataset_size - train_size):
                end = -1
            else:
                end = test_iterator_size
            output = model(tf.slice(test_data, [i, 0, 0, 0], [end, -1, -1, -1]))

            statement_pred.append(output[0])
            drop_pred.append(output[1])
            operator_pred.append(output[2])

        statement_pred, drop_pred, operator_pred = tf.concat(statement_pred, axis=0), tf.concat(drop_pred,
                                                                                                axis=0), tf.concat(
            operator_pred, axis=0)

        loss_act, loss_drop, loss_operator = [model.model.loss[out.node.layer.name] for out in model.output]

        test_statement_loss = float(loss_act(test_statement_target, statement_pred).numpy())
        test_drop_loss = float(loss_drop(test_drop_target, drop_pred).numpy())
        test_operator_loss = float(loss_operator(test_operator_target, operator_pred).numpy())

        print("Train loss: S %f" % avg_statement_train_loss, "D %f" % avg_drop_train_loss,
              "F %f" % avg_operator_train_loss)
        print("Test loss: S %f" % test_statement_loss, "D %f" % test_drop_loss,
              "F %f" % test_operator_loss)

        predict = tf.math.argmax(statement_pred, axis=1)

        test_error = 1 - int(
            tf.math.count_nonzero(tf.equal(predict, tf.math.argmax(test_statement_target, axis=1))).numpy()) / float(
            test_data.shape[0])
        print("Test classification error: %f" % test_error)
        with model.summary_writer.as_default():
            with tf.name_scope('test'):
                tf.summary.scalar("test_statement_loss", data=test_statement_loss, step=epoch)
                tf.summary.scalar("test_drop_loss", data=test_drop_loss, step=epoch)
                tf.summary.scalar("test_operator_loss", data=test_operator_loss, step=epoch)
                tf.summary.scalar("Test classification error", data=test_error, step=epoch)

        model.save_model(args.output_path + "\\epoch_%d" % epoch)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    wordindex_dict={str(i):i for i in range(params.integer_min,params.integer_max + 1)}
    # write wordindex dictionary
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for w in wordindex_dict:
            f.write("{}\n".format(w))

    weights = tf.Variable(emb.get_weights()[0][1:])
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # configuration set-up
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)

if __name__ == '__main__':
    main()
