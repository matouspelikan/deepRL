#pragma once

#include <array>
#include <functional>
#include <random>

#include "az_quiz.h"

thread_local std::mt19937* generator = new std::mt19937{std::random_device()()};

typedef std::array<float, AZQuiz::ACTIONS> Policy;

typedef std::function<void(const AZQuiz&, Policy&, float&)> Evaluator;

struct Node {
  AZQuiz game;
  unsigned n;
  float w;
  Policy priors;
  std::array<int, AZQuiz::ACTIONS> children;

  Node(const AZQuiz& game, const Policy& priors, float w) : game(game), n(1), w(w), priors(priors) { children.fill(-1); }
};

void zero_out_invalid_actions(const AZQuiz& game, Policy& policy) {
  float sum = 0.;
  for (int action = 0; action < AZQuiz::ACTIONS; action++)
    if (game.valid(action))
      sum += policy[action];
    else
      policy[action] = 0;

  if (sum) {
    sum = 1. / sum;
    for (int action = 0; action < AZQuiz::ACTIONS; action++)
      policy[action] *= sum;
  }
}

void mcts(const AZQuiz& game, const Evaluator& evaluator, int num_simulations, float epsilon, float alpha, Policy& policy) {
  // TODO: Implement MCTS, returning the generated `policy`.
  //
  // To run the neural network, use the given `evaluator`, which returns a policy and
  // a value function for the given game.
  bool visit_all_root_children = epsilon < 0;
  epsilon = epsilon > 0 ? epsilon : -epsilon;

  std::vector<Node> tree;
  tree.reserve(num_simulations + 1);

  Policy priors;
  float value;
  evaluator(game, priors, value);
  zero_out_invalid_actions(game, priors);
  if (epsilon) {
    Policy gammas;
    float sum = 0;
    std::gamma_distribution<float> gamma_distribution(alpha);
    for (int action = 0; action < AZQuiz::ACTIONS; action++) {
      gammas[action] = gamma_distribution(*generator);
      sum += gammas[action];
    }
    for (int action = 0; action < AZQuiz::ACTIONS; action++)
      priors[action] = (1 - epsilon) * priors[action] + epsilon * gammas[action] / sum;
  }
  tree.emplace_back(game, priors, value);

  std::vector<int> path;
  for (int simulation = 0; simulation < num_simulations; simulation++) {
    path.clear();
    int node = 0, child;
    while (tree[node].game.winner < 0) {
      path.push_back(node);

      // Select a child
      child = -1;
      float best_score = 0;
      for (int i = 0; i < AZQuiz::ACTIONS; i++)
        if (tree[node].game.valid(i)) {
          float score = visit_all_root_children && (node == 0) ? INFINITY : 0;
          float child_visit = 1;
          if (tree[node].children[i] >= 0) {
            auto& child = tree[tree[node].children[i]];
            score = -child.w / child.n;
            child_visit += child.n;
          }
          score += 1.25 * tree[node].priors[i] * sqrtf(tree[node].n) / child_visit;

          if (child < 0 || score > best_score) {
            child = i;
            best_score = score;
          }
        }

      // Enter the child if it exists
      if (tree[node].children[child] < 0)
        break;
      node = tree[node].children[child];
    }

    // Either we are in a node that is end of game, or we have an unvisited child.
    if (tree[node].game.winner < 0) {
      AZQuiz moved = tree[node].game;
      moved.move(child);
      tree[node].children[child] = tree.size();
      node = tree.size();
      if (moved.winner < 0) {
        evaluator(moved, priors, value);
        zero_out_invalid_actions(moved, priors);
      } else {
        priors.fill(0);
        value = moved.winner == moved.to_play ? 1 : -1;
      }
      tree.emplace_back(moved, priors, value);
    } else {
      path.push_back(node);
      value = tree[node].game.winner == tree[node].game.to_play ? 1 : -1;
    }

    for (auto& parent : path) {
      tree[parent].n += 1;
      tree[parent].w += value * (tree[node].game.to_play == tree[parent].game.to_play ? 1 : -1);
    }
  }

  // Now generate the final policy. As a special-case, we return the priors when there are no simulations.
  if (num_simulations) {
    for (int action = 0; action < AZQuiz::ACTIONS; action++)
      policy[action] = (tree[0].children[action] >= 0 ? tree[tree[0].children[action]].n : 0) / float(num_simulations);
  } else {
    policy = tree[0].priors;
  }
}
