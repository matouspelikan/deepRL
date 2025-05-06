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

from az_quiz import AZQuiz
import az_quiz_cpp
import az_quiz_evaluator
import az_quiz_player_simple_heuristic
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
parser.add_argument("--model_path", default="az_quiz_cpp.keras", type=str, help="Model path")
parser.add_argument("--num_simulations", default=100, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--randomized", default=False, action="store_true", help="Use randomized variant.")
parser.add_argument("--rotate", default=False, action="store_true", help="Use random rotations.")
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
        # TODO: Define an agent network in `self._model`.
        #
        # A possible architecture known to work consists of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer with softmax
        #   activation to produce the policy,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.

        inputs = keras.Input([AZQuiz.N, AZQuiz.N, AZQuiz.C])
        hidden = inputs
        for i in range(5):
            hidden = keras.layers.Conv2D(20 if i == 4 else 15, 3, padding="same", activation="relu")(hidden)

        actions = keras.layers.Conv2D(2, 3, padding="same", activation="relu")(hidden)
        actions = keras.layers.Flatten()(actions)
        actions = keras.layers.Dense(AZQuiz.ACTIONS, activation="softmax")(actions)

        value = keras.layers.Conv2D(2, 3, padding="same", activation="relu")(hidden)
        value = keras.layers.Flatten()(value)
        value = keras.layers.Dense(1, activation="tanh")(value)

        self._model = keras.Model(inputs=inputs, outputs=[actions, value])
        self._model.compile(
            optimizer=keras.optimizers.Adam(args.learning_rate),
            loss=[keras.losses.CategoricalCrossentropy(), keras.losses.MeanSquaredError()],
        )
        self._rotate = args.rotate

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent.__new__(Agent)
        agent._model = keras.models.load_model(path)
        # In some older models, the last `softmax` was not used, so we add it if needed.
        if "softmax" not in str(agent._model.get_config()):
            print("Adding `softmax` for model {}".format(path))
            agent._model = keras.Model(agent._model.inputs, [keras.ops.softmax(agent._model.outputs[0]), agent._model.outputs[1]])
        agent._rotate = args.rotate
        return agent

    def save(self, path: str) -> None:
        self._model.save(path)

    @wrappers.raw_typed_tf_function(tf.float32, tf.float32, tf.float32)
    def _train_tf(self, boards: tf.Tensor, target_policies: tf.Tensor, target_values: tf.Tensor) -> None:
        # TODO: Train the model based on given boards, target policies and target values.
        with tf.GradientTape() as tape:
            loss = self._model.compute_loss(boards, [target_policies, target_values], self._model(boards, training=True))
        self._model.optimizer.apply(tape.gradient(loss, self._model.trainable_variables), self._model.trainable_variables)

    def train(self, boards: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> None:
        boards = np.asarray(boards, np.float32)
        target_policies = np.asarray(target_policies, np.float32)
        target_values = np.asarray(target_values, np.float32)
        if self._rotate:
            permutation = np.random.randint(AZQuiz.board_rotations.shape[0], size=boards.shape[0])
            boards = boards.reshape([-1, AZQuiz.N * AZQuiz.N, AZQuiz.C])
            boards = boards[np.arange(boards.shape[0])[:, np.newaxis], AZQuiz.board_rotations[permutation]]
            boards = boards.reshape([-1, AZQuiz.N, AZQuiz.N, AZQuiz.C])
            target_policies = target_policies[np.arange(target_policies.shape[0])[:, np.newaxis], AZQuiz.policy_rotations[permutation]]
        self._train_tf(boards, target_policies, target_values)

    @wrappers.raw_typed_tf_function(tf.float32)
    def _predict_tf(self, boards: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function.
        policy, value = self._model(boards)
        return policy, value[..., 0]

    def predict(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boards = np.asarray(boards, np.float32)
        if self._rotate and boards.shape[0] > 1:
            permutation = np.random.randint(AZQuiz.board_rotations.shape[0], size=boards.shape[0])
            boards = boards.reshape([-1, AZQuiz.N * AZQuiz.N, AZQuiz.C])
            boards = boards[np.arange(boards.shape[0])[:, np.newaxis], AZQuiz.board_rotations[permutation]]
            boards = boards.reshape([-1, AZQuiz.N, AZQuiz.N, AZQuiz.C])
        policy, value = self._predict_tf(boards)
        if self._rotate and boards.shape[0] > 1:
            policy = policy[np.arange(policy.shape[0])[:, np.newaxis], AZQuiz.policy_rotations_inverse[permutation]]
        return policy, value

board = np.arange(AZQuiz.N * AZQuiz.N, dtype=np.int32).reshape([AZQuiz.N, AZQuiz.N])
AZQuiz.board_rotations, AZQuiz.policy_rotations, AZQuiz.policy_rotations_inverse = [], [], []
for _ in range(3):
    for _ in range(2):
        AZQuiz.board_rotations.append(board.ravel())
        AZQuiz.policy_rotations_inverse.append(np.argsort(board.ravel()[[AZQuiz.N * i + j for i in range(AZQuiz.N) for j in range(i + 1)]]))
        AZQuiz.policy_rotations.append(np.argsort(AZQuiz.policy_rotations_inverse[-1]))
        board_before_mirroring, board = board, board.copy()
        for i in range(AZQuiz.N):
            board[i, :i + 1] = board_before_mirroring[i, i::-1]
    board_before_rotation, board = board, board.copy()
    for i in range(AZQuiz.N):
        board[AZQuiz.N - 1 - i:, AZQuiz.N - 1 - i] = board_before_rotation[i, :i + 1]
AZQuiz.board_rotations = np.array(AZQuiz.board_rotations, dtype=np.int32)
AZQuiz.policy_rotations = np.array(AZQuiz.policy_rotations, dtype=np.int32)
AZQuiz.policy_rotations_inverse = np.array(AZQuiz.policy_rotations_inverse, dtype=np.int32)

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
    import az_quiz_player_fork_heuristic
    players = {
        "sihe": az_quiz_player_simple_heuristic.Player(),
        "fork": az_quiz_player_fork_heuristic.Player(),
    }
    best_evaluation = 0
    az_quiz_cpp.simulated_games_start(args.workers, args.randomized, args.num_simulations, args.sampling_moves, args.epsilon, args.alpha);
    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = az_quiz_cpp.simulated_game(agent.predict);
            replay_buffer.extend(game)

        if iteration % args.evaluate_each == 0:
            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games:
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append("Move {}, result {}".format(i, outcome).center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                "{:>3.0f} ".format(policy[action] * 100))
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                print(*["".join(line) for line in log], sep="\n")

        # Train
        for _ in range(args.train_for):
            # TODO: Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            batch = replay_buffer.sample(min(len(replay_buffer), args.batch_size), np.random)
            boards, policies, outcomes = zip(*batch)
            agent.train(boards, policies, outcomes)

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            print("Evaluation after iteration {}, {}".format(iteration, datetime.datetime.now()))
            for randomized in [False] + ([True] if args.randomized else []):
                for mode in ["normal", "first_chosen"]:
                    for name, player in players.items():
                        score = az_quiz_evaluator.evaluate(
                            [Player(agent, argparse.Namespace(num_simulations=0)), player],
                            games=112 if randomized else 56 if "az" not in name else 28,
                            randomized=randomized, first_chosen=mode == "first_chosen", render=False, verbose=False)
                        print("{} {}: {:.1f}%".format(mode, name, 100 * score))
                        if randomized == False and mode == "normal" and name == "sihe" and (score > best_evaluation or score == 1):
                            agent.save("{}-{:06.1f}k-{:.0f}.keras".format(args.model_path, iteration / 1000, 100 * score))
                            best_evaluation = score
            print(flush=True)
    az_quiz_cpp.simulated_games_stop();

    return agent


#####################
# Evaluation Player #
#####################
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
        # Predict a best possible action.

        policy = az_quiz_cpp.mcts(game._board, game._to_play, game._randomized, self.agent.predict, self.args.num_simulations, 0., 0.)
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

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    az_quiz_evaluator.evaluate(
        [player, az_quiz_player_simple_heuristic.Player(seed=args.seed)],
        games=56, randomized=False, first_chosen=False, render=False, verbose=True,
    )
