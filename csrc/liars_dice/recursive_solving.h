#pragma once

#include <memory>
#include <random>
#include <vector>

#include "liars_dice.h"
#include "net_interface.h"
#include "subgame_solving.h"

namespace liars_dice {

struct RecursiveSolvingParams {
  int num_dice;
  int num_faces;
  // BR玩家采取随机动作的概率
  float random_action_prob = 1.0;
  bool sample_leaf = false;
  SubgameSolvingParams subgame_params;
};

class RlRunner {
 public:
  RlRunner(const RecursiveSolvingParams& params, std::shared_ptr<IValueNet> net,
           int seed)
      : game_(Game(params.num_dice, params.num_faces)),
        subgame_params_(params.subgame_params),
        random_action_prob_(params.random_action_prob),
        sample_leaf_(params.sample_leaf),
        net_(net),
        gen_(seed) {}

  // 已弃用的构造函数
  RlRunner(const Game& game, const SubgameSolvingParams& params,
           std::shared_ptr<IValueNet> net, int seed)
      : RlRunner(build_params(game, params), net, seed) {}

  void step();

 private:
  static RecursiveSolvingParams build_params(
      const Game& game, const SubgameSolvingParams& fp_params) {
    RecursiveSolvingParams params;
    params.subgame_params = fp_params;
    params.num_dice = game.num_dice;
    params.num_faces = game.num_faces;
    return params;
  }

  // 从求解器中采样新状态并更新信念
  void sample_state(const ISubgameSolver* solver);
  void sample_state_single(const ISubgameSolver* solver);
  void sample_state_to_leaf(const ISubgameSolver* solver);

  // 拥有所有小型资源
  const Game game_;
  const SubgameSolvingParams subgame_params_;
  const float random_action_prob_;
  const bool sample_leaf_;
  std::shared_ptr<IValueNet> net_;

  // 当前状态
  PartialPublicState state_;
  // 信念缓冲区
  Pair<std::vector<double>> beliefs_;

  std::mt19937 gen_;
};

// 通过递归求解子游戏计算策略。仅使用子游戏根节点的策略，并继续处理其子节点
TreeStrategy compute_strategy_recursive(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net);
// 通过递归求解子游戏计算策略。对所有非叶子子游戏节点使用完整游戏策略，并继续处理子游戏中的叶子节点
TreeStrategy compute_strategy_recursive_to_leaf(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net);
// 通过模拟训练的方式递归求解子游戏计算策略：
// 1. 使用线性权重采样随机迭代
// 2. 将求解器的采样策略复制到完整游戏策略中
// 3. 使用belief_propogation_strategy在叶子节点计算信念并开始递归
TreeStrategy compute_sampled_strategy_recursive_to_leaf(
    const Game& game, const SubgameSolvingParams& subgame_params,
    std::shared_ptr<IValueNet> net, int seed, bool root_only = false);

}  // namespace liars_dice