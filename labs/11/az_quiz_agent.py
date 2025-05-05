#!/usr/bin/env python3

#python .\az_quiz_evaluator.py az_quiz_agent.py:--num_simulations=0 az_quiz_player_simple_heuristic.py

import argparse
import collections
import os

import numpy as np
import torch

from az_quiz import AZQuiz
import az_quiz_evaluator
import az_quiz_player_simple_heuristic
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.03, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=128, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz2.pt", type=str, help="Model path")
parser.add_argument("--num_simulations", default=30, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=10, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=15, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100_000, type=int, help="Replay buffer max length.")


class Network(torch.nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 3, 1, "same"),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 16, 3, 1, "same"),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 16, 3, 1, "same"),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 16, 3, 1, "same"),
            torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 16, 3, 1, "same"),
            torch.nn.ReLU()
        ).to(self.device)

        self.policy_head = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 2, 3, 1, "same"),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(98, 28), 
            torch.nn.Softmax(dim=1)
        ).to(self.device)

        self.value_head = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 2, 3, 1, "same"),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(98, 1),
            torch.nn.Tanh()
        ).to(self.device)

        self.dropout = torch.nn.Dropout2d(0.2)

    def forward(self, x):
        hidden = self.shared(x)
        hidden = self.dropout(hidden)
        policy = self.policy_head(hidden)
        value = self.value_head(hidden)
        return policy, value

#########
# Agent #
#########
class Agent:
    # Use GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.policy_loss = torch.nn.CrossEntropyLoss()
        self.value_loss = torch.nn.MSELoss()

        self._model = Network()
        params = self._model.parameters()
        # print([p for p in self._model.named_parameters()])
        self.optimizer = torch.optim.Adam(params, args.learning_rate)


    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> "Agent":
        # A static method returning a new Agent loaded from the given path.
        agent = Agent(args)
        agent._model.load_state_dict(torch.load(path, map_location=agent.device))
        return agent

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    @wrappers.typed_torch_function(device, torch.float32, torch.float32, torch.float32, via_np=True)
    def train(self, boards: torch.Tensor, target_policies: torch.Tensor, target_values: torch.Tensor) -> None:
        # TODO: Train the model based on given boards, target policies and target values.
        self._model.train()
        boards = boards.permute(0, 3, 1, 2)
        policy, value = self._model(boards)

        # print(policy.shape, value.shape)
        # print(target_policies.shape, target_values.shape)

        value_loss = self.value_loss(value, target_values)
        policy_loss = self.policy_loss(policy, target_policies)

        # print(value_loss, policy_loss)
        total_loss = value_loss + policy_loss
        # print(total_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        with torch.no_grad():
            self.optimizer.step()


    @wrappers.typed_torch_function(device, torch.float32, via_np=True)
    def predict_train(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function.
        boards = boards.permute(0, 3, 1, 2)
        self._model.train()
        with torch.no_grad():
            policy, value = self._model(boards)
        return policy, value

    @wrappers.typed_torch_function(device, torch.float32, via_np=True)
    def predict(self, boards: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function.
        boards = boards.permute(0, 3, 1, 2)
        self._model.eval()
        with torch.no_grad():
            policy, value = self._model(boards)
        return policy, value

    def board(self, game: AZQuiz) -> np.ndarray:
        # TODO: Generate the boards from the current `AZQuiz` game.
        #
        # The `game.board` returns a board representation, but you also need to
        # somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `swap_players` of `AZQuiz.clone` might come handy);
        # - indicate the current player by adding channels to the representation.
        if game.to_play:
            game = game.clone(True)
        return game.board
        


########
# MCTS #
########
class MCTNode:
    def __init__(self, prior: float | None, pov = None):
        self.prior = prior  # Prior probability from the agent.
        self.game = None    # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0
        self.pov = pov

    def value(self) -> float:
        # TODO: Return the value of the current node, handling the
        # case when `self.visit_count` is 0.
        return self.total_value / self.visit_count if self.visit_count > 0 else 0

    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # TODO: Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.

        def winner_to_value(winner):
            if winner == self.pov:
                value = 1
            elif winner == 1 - self.pov:
                value = -1
            else:
                value = 0
            return value

        if game.winner is not None:
            value = winner_to_value(game.winner)
            policy = None
        else:
            policy_network, value_network = agent.predict(agent.board(game)[np.newaxis])
            policy_network = policy_network[0]
            value = value_network[0, 0]

            # winner = self.simulation(game)
            # value = winner_to_value(winner)

            policy = np.zeros(game.ACTIONS)
            for i in range(game.ACTIONS):
                if game.valid(i):
                    policy[i] = 1.
            policy = policy / np.sum(policy)

            for i in range(game.ACTIONS):
                if game.valid(i):
                    self.children[i] = MCTNode(policy_network[i], game.to_play)

        self.visit_count, self.total_value = 1, value

    def simulation(self, game):
        game = game.clone()
        while game.winner is None:
            game.move(np.random.choice(game.valid_actions()))
        return game.winner

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # TODO: Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior
        dirichlets = np.random.dirichlet([0.3 for _ in range(self.game.ACTIONS)])
        for i, child in enumerate(self.children.values()):
            child.prior = epsilon*dirichlets[i] + (1 - epsilon)*child.prior

    def select_child(self) -> tuple[int, "MCTNode"]:
        # Select a child according to the PUCT formula.
        def ucb_score(child: "MCTNode"):
            # TODO: For a given child, compute the UCB score as
            #   Q(s, a) + C(s) * P(s, a) * (sqrt(N(s)) / (N(s, a) + 1)),
            # where:
            # - Q(s, a) is the estimated value of the action stored in the
            #   `child` node. However, the value in the `child` node is estimated
            #   from the view of the player playing in the `child` node, which
            #   is usually the other player than the one playing in `self`,
            #   and in that case the estimated value must be "inverted";
            # - C(s) in AlphaZero is defined as
            #     log((1 + N(s) + 19652) / 19652) + 1.25
            #   Personally I used 1965.2 to account for shorter games, but I do not
            #   think it makes any difference;
            # - P(s, a) is the prior computed by the agent;
            # - N(s) is the number of visits of state `s`;
            # - N(s, a) is the number of visits of action `a` in state `s`.
            Cs = np.log((1 + self.visit_count + 19652.2) / 19652.2) + 1.25
            return child.value() + Cs * child.prior * (np.sqrt(self.visit_count) / (child.visit_count + 1))

        # TODO: Return the (action, child) pair with the highest `ucb_score`.
        actions, scores = zip(*self.children.items())
        scores = list(map(ucb_score, scores))
        idx_max = np.argmax(scores)
        return actions[idx_max], self.children[actions[idx_max]]


def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    # Run the MCTS search and return the policy proportional to the visit counts,
    # optionally including exploration noise to the root children.
    root = MCTNode(None, game.to_play)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Perform the `args.num_simulations` number of MCTS simulations.
    for _ in range(args.num_simulations):
        # TODO: Starting in the root node, traverse the tree using `select_child()`,
        # until a `node` without `children` is found.
        trace: list[MCTNode] = []
        node = root
        while len(node.children) > 0:
            trace.append(node)
            action, node = node.select_child()
        # print(len(trace))
        # trace.pop(-1)
        parent = trace[-1]

        # If the node has not been evaluated, evaluate it.
        if not node.is_evaluated():
            # TODO: Evaluate the `node` using the `evaluate` method. To that
            # end, create a suitable `AZQuiz` instance for this node by cloning
            # the `game` from its parent and performing a suitable action.
            _game = parent.game.clone()
            # _value = None
            # if not _game.valid(action):
            #     _value = 1.0 if _game.to_play == 0 else -1.0
            # else:
            _game.move(action)
            node.evaluate(_game, agent)
        else:
            # TODO: If the node has been evaluated but has no children, the
            # game ends in this node. Update it appropriately.
            node.total_value += node.value()
            node.visit_count += 1

        # Get the value of the node.
        value = node.value()
        value_pov = node.pov

        # TODO: For all parents of the `node`, update their value estimate,
        # i.e., the `visit_count` and `total_value`.
        for t in trace:
            if t.pov == value_pov:
                t.total_value += value
            else:
                t.total_value -= value
            t.visit_count += 1

    # TODO: Compute a policy proportional to visit counts of the root children.
    # Note that invalid actions are not the children of the root, but the
    # policy should still return 0 for them.
    policy = np.zeros(game.ACTIONS)
    for a in range(game.ACTIONS):
        if a in root.children:
            policy[a] = root.children[a].visit_count
    policy = policy / np.sum(policy)
    return policy


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])

def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz(randomized=False)
    buffer = []
    move_n = 0
    while game.winner is None:
        # TODO: Run the `mcts` with exploration.
        policy = mcts(game, agent, args, True)
        buffer.append((agent.board(game), game.to_play, policy))

        # TODO: Select an action, either by sampling from the policy or greedily,
        # according to the `args.sampling_moves`.
        if move_n < args.sampling_moves:
            action = np.random.choice(np.arange(game.ACTIONS), p=policy)
        else:
            action = np.argmax(policy)

        game.move(action)
        move_n += 1


    replayBuffer: list[ReplayBufferEntry] = []
    for board, pov, policy in buffer:
        if game.winner == pov:
            value = 1
        elif game.winner == 1 - pov:
            value = -1
        else:
            value = 0
        replayBuffer.append(ReplayBufferEntry(board, policy, value))

    # TODO: Return all encountered game states, each consisting of
    # - the board (probably via `agent.board`),
    # - the policy obtained by MCTS,
    # - the outcome based on the outcome of the whole game.
    return replayBuffer


def train(args: argparse.Namespace) -> Agent:
    # Perform training
    agent = Agent(args)
    replay_buffer = wrappers.ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True
    while training:
        iteration += 1
        if iteration > 20:
            training = False

        # Generate simulated games
        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)

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
            # print(len(replay_buffer))
            sample = replay_buffer.sample(args.batch_size)
            boards, policies, outcomes = zip(*sample)
            boards = np.stack(boards)
            policies = np.stack(policies)
            outcomes = np.stack(outcomes)[:, np.newaxis]
            # print(boards.shape, policies.shape, outcomes.shape)
            agent.train(boards, policies, outcomes)

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            score = az_quiz_evaluator.evaluate(
                [Player(agent, argparse.Namespace(num_simulations=0)),
                 az_quiz_player_simple_heuristic.Player(seed=args.seed)],
                games=56, randomized=False, first_chosen=False, render=False, verbose=False)
            print("Evaluation after iteration {}: {:.1f}%".format(iteration, 100 * score), flush=True)

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
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            # policy = torch.nn.functional.one_hot(torch.tensor(np.random.choice(game.ACTIONS)), game.ACTIONS).numpy()

            policy, value = self.agent.predict(self.agent.board(game)[np.newaxis])
            policy = policy[0]

            # policy = np.ones(game.ACTIONS)
            # for i in range(game.ACTIONS):
            #     if not game.valid(i):
            #         policy[i] = 0
            # policy = policy / np.sum(policy)
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = mcts(game, self.agent, self.args, False)

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set random seeds and the number of threads
    np.random.seed(args.seed)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)
        agent.save(args.model_path)

    return Player(agent, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    az_quiz_evaluator.evaluate(
        [player, az_quiz_player_simple_heuristic.Player(seed=args.seed)],
        games=56, randomized=False, first_chosen=False, render=False, verbose=True,
    )
