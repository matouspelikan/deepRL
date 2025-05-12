#pragma once

#include <array>
#include <functional>
#include <random>
#include <vector>
#include <cmath>

#include "pisqorky.h"

thread_local std::mt19937* generator = new std::mt19937{std::random_device()()};

typedef std::array<float, Pisqorky::ACTIONS> Policy;

typedef std::function<void(const Pisqorky&, Policy&, float&)> Evaluator;

struct Node {
  Pisqorky game;
  unsigned n;
  float w;
  Policy priors;
  std::array<int, Pisqorky::ACTIONS> children;

  Node(const Pisqorky& game, const Policy& priors, float w) : game(game), n(1), w(w), priors(priors) { children.fill(-1); }
};

// Check if a position is adjacent to any occupied tile
bool is_adjacent_to_occupied(const Pisqorky& game, int action) {
  if (!game.valid(action))
    return false;
    
  // Convert action to 2D coordinates
  int x = action % Pisqorky::N;
  int y = action / Pisqorky::N;
  
  // Check all 8 adjacent positions
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      // Skip the center position (the action itself)
      if (dx == 0 && dy == 0)
        continue;
        
      int nx = x + dx;
      int ny = y + dy;
      
      // Check if the adjacent position is within bounds
      if (nx >= 0 && nx < Pisqorky::N && ny >= 0 && ny < Pisqorky::N) {
        int adjacent_action = ny * Pisqorky::N + nx;
        // If the adjacent position is occupied, return true
        if (game.board[adjacent_action] != 0)
          return true;
      }
    }
  }
  
  return false;
}

void zero_out_invalid_actions(const Pisqorky& game, Policy& policy) {
  float sum = 0.;
  bool any_occupied = false;
  
  // Check if there are any occupied tiles on the board
  for (int action = 0; action < Pisqorky::ACTIONS; action++) {
    if (game.board[action] != 0) {
      any_occupied = true;
      break;
    }
  }
  
  for (int action = 0; action < Pisqorky::ACTIONS; action++) {
    // If the board is empty (first move), consider all valid actions
    // Otherwise, only consider actions adjacent to occupied tiles
    if (game.valid(action) && (!any_occupied || is_adjacent_to_occupied(game, action)))
      sum += policy[action];
    else
      policy[action] = 0;
  }

  if (sum) {
    sum = 1. / sum;
    for (int action = 0; action < Pisqorky::ACTIONS; action++)
      policy[action] *= sum;
  }
}

void mcts(const Pisqorky& game, const Evaluator& evaluator, int num_simulations, float epsilon, float alpha, Policy& policy) {
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
    for (int action = 0; action < Pisqorky::ACTIONS; action++) {
      gammas[action] = gamma_distribution(*generator);
      sum += gammas[action];
    }
    for (int action = 0; action < Pisqorky::ACTIONS; action++)
      priors[action] = (1 - epsilon) * priors[action] + epsilon * gammas[action] / sum;
  }
  tree.emplace_back(game, priors, value);

  std::vector<int> path;
  for (int simulation = 0; simulation < num_simulations; simulation++) {
    path.clear();
    int node = 0, child;
    while (tree[node].game.winner < 0) {
      path.push_back(node);

      // Check if there are any occupied tiles on the board
      bool any_occupied = false;
      for (int action = 0; action < Pisqorky::ACTIONS; action++) {
        if (tree[node].game.board[action] != 0) {
          any_occupied = true;
          break;
        }
      }

      // Select a child
      child = -1;
      float best_score = 0;
      for (int i = 0; i < Pisqorky::ACTIONS; i++) {
        // If the board is empty (first move), consider all valid actions
        // Otherwise, only consider actions adjacent to occupied tiles
        if (tree[node].game.valid(i) && (!any_occupied || is_adjacent_to_occupied(tree[node].game, i))) {
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
      }

      // Enter the child if it exists
      if (tree[node].children[child] < 0)
        break;
      node = tree[node].children[child];
    }

    // Either we are in a node that is end of game, or we have an unvisited child.
    if (tree[node].game.winner < 0) {
      Pisqorky moved = tree[node].game;
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
    for (int action = 0; action < Pisqorky::ACTIONS; action++)
      policy[action] = (tree[0].children[action] >= 0 ? tree[tree[0].children[action]].n : 0) / float(num_simulations);
  } else {
    policy = tree[0].priors;
  }
}
