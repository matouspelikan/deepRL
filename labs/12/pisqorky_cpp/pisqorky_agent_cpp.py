#!/usr/bin/env python3
import argparse
import collections
import datetime
import math
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import keras
import numpy as np
import tensorflow as tf

from pisqorky import Pisqorky
import pisqorky_cpp
import pisqorky_evaluator
import pisqorky_player_heuristic
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.1, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=200, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="pisqorky_cpp.keras", type=str, help="Model path")
parser.add_argument("--num_simulations", default=100, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=16, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=1, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100_000, type=int, help="Replay buffer max length.")
parser.add_argument("--workers", default=512, type=int, help="Number of MCTS worker threads.")


#########
# Agent #
#########
class Agent:
    def __init__(self, args: argparse.Namespace):
        # Define an agent network in `self._model`.
        # Architecture:
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters
        # - a policy head with 3x3 convolution to reduce channels to 2, flatten, and dense layer with softmax
        # - a value head with 3x3 convolution to reduce channels to 2, flatten, and dense layer with tanh

        inputs = keras.Input([Pisqorky.N, Pisqorky.N, Pisqorky.C])
        hidden = inputs
        for i in range(5):
            hidden = keras.layers.Conv2D(20 if i == 4 else 15, 3, padding="same", activation="relu")(hidden)

        actions = keras.layers.Conv2D(2, 3, padding="same", activation="relu")(hidden)
        actions = keras.layers.Flatten()(actions)
        actions = keras.layers.Dense(Pisqorky.ACTIONS, activation="softmax")(actions)

        value = keras.layers.Conv2D(2, 3, padding="same", activation="relu")(hidden)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(1, activation="tanh")(value)

        self._model = keras.Model(inputs=inputs, outputs=[actions, value])
        self._model.compile(
            optimizer=keras.optimizers.Adam(args.learning_rate),
            loss=[keras.losses.CategoricalCrossentropy(), keras.losses.MeanSquaredError()],
        )

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent.__new__(Agent)
        agent._model = keras.models.load_model(path)
        # In some older models, the last `softmax` was not used, so we add it if needed.
        if "softmax" not in str(agent._model.get_config()):
            print("Adding `softmax` for model {}".format(path))
            agent._model = keras.Model(agent._model.inputs, [keras.ops.softmax(agent._model.outputs[0]), agent._model.outputs[1]])
        return agent

    def save(self, path: str) -> None:
        self._model.save(path)

    @wrappers.raw_typed_tf_function(tf.float32, tf.float32, tf.float32)
    def _train_tf(self, boards: tf.Tensor, target_policies: tf.Tensor, target_values: tf.Tensor) -> tuple[float, float, float]:
        # Train the model based on given boards, target policies and target values.
        with tf.GradientTape() as tape:
            policy_pred, value_pred = self._model(boards, training=True)
            policy_loss = tf.keras.losses.categorical_crossentropy(target_policies, policy_pred)
            value_loss = tf.keras.losses.mean_squared_error(target_values, value_pred[..., 0])
            total_loss = policy_loss + value_loss
            
        self._model.optimizer.apply(
            tape.gradient(total_loss, self._model.trainable_variables), 
            self._model.trainable_variables
        )
        
        return tf.reduce_mean(total_loss), tf.reduce_mean(policy_loss), tf.reduce_mean(value_loss)

    def train(self, boards: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> tuple[float, float, float]:
        boards = np.asarray(boards, np.float32)
        target_policies = np.asarray(target_policies, np.float32)
        target_values = np.asarray(target_values, np.float32)
        return self._train_tf(boards, target_policies, target_values)

    @wrappers.raw_typed_tf_function(tf.float32)
    def _predict_tf(self, boards: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # Return the predicted policy and the value function.
        policy, value = self._model(boards)
        return policy, value[..., 0]

    def predict(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boards = np.asarray(boards, np.float32)
        policy, value = self._predict_tf(boards)
        return policy, value


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])

def train(args: argparse.Namespace) -> Agent:
    # Perform training
    agent = Agent(args)
    replay_buffer = wrappers.ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True
    best_evaluation = 0
    pisqorky_cpp.simulated_games_start(args.workers, args.num_simulations, args.sampling_moves, args.epsilon, args.alpha)
    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = pisqorky_cpp.simulated_game(agent.predict)
            replay_buffer.extend(game)

        if iteration % args.evaluate_each == 0:
            # If required, show the generated game
            if args.show_sim_games:
                log = [[] for _ in range(Pisqorky.N + 1)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append("Move {}, result {}".format(i, outcome).center(28))
                    action = 0
                    for row in range(Pisqorky.N):
                        for col in range(Pisqorky.N):
                            log[1 + row].append(
                                " XX " if board[row, col, 1] else
                                " .. " if board[row, col, 2] else
                                "{:>3.0f} ".format(policy[action] * 100))
                            action += 1
                print(*["".join(line) for line in log], sep="\n")

        # Train
        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        for _ in range(args.train_for):
            # Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            batch = replay_buffer.sample(min(len(replay_buffer), args.batch_size), np.random)
            boards, policies, outcomes = zip(*batch)
            total_loss, policy_loss, value_loss = agent.train(boards, policies, outcomes)
            total_loss_sum += total_loss
            policy_loss_sum += policy_loss
            value_loss_sum += value_loss
        
        # Log losses every 10 iterations
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Loss = {total_loss_sum/args.train_for:.4f}, " +
                  f"Policy Loss = {policy_loss_sum/args.train_for:.4f}, " +
                  f"Value Loss = {value_loss_sum/args.train_for:.4f}")

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Run an evaluation against the heuristic player
            print("Evaluation after iteration {}, {}".format(iteration, datetime.datetime.now()))
            score = pisqorky_evaluator.evaluate(
                [Player(agent, argparse.Namespace(num_simulations=args.num_simulations)), pisqorky_player_heuristic.Player()],
                games=100, render=False, verbose=False)
            print("Evaluation: {:.1f}%".format(100 * score))
            if score > best_evaluation or score == 1:
                agent.save("{}-{:06.1f}k-{:.0f}.keras".format(args.model_path, iteration / 1000, 100 * score))
                best_evaluation = score
            print(flush=True)
    pisqorky_cpp.simulated_games_stop()

    return agent


#####################
# Evaluation Player #
#####################
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: Pisqorky) -> int:
        # Predict a best possible action.
        policy = pisqorky_cpp.mcts(game._board, game._to_play, self.agent.predict, self.args.num_simulations, 0., 0.)
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set random seeds and the number of threads
    if args.seed is not None:
        keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(args)

    # Run an evaluation versus the heuristic player with the same parameters as in ReCodEx.
    pisqorky_evaluator.evaluate(
        [player, pisqorky_player_heuristic.Player(seed=args.seed)],
        games=100, render=False, verbose=True,
    )
