#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <torch/script.h>

#include "rela/context.h"
#include "rela/data_loop.h"
#include "rela/model_locker.h"
#include "rela/prioritized_replay.h"

#include "real_net.h"
#include "recursive_solving.h"
#include "subgame_solving.h"
#include "util.h"

using namespace liars_dice;
using namespace rela;
// 获取树的深度
int get_depth(const Tree& tree, int root = 0) {
  int depth = 1;
  for (auto child : ChildrenIt(tree[root])) {
    depth = std::max(depth, 1 + get_depth(tree, child));
  }
  return depth;
}
// 计时器 输出到创建时的时间
struct Timer {
  std::chrono::time_point<std::chrono::system_clock> start =
      std::chrono::system_clock::now();

  double tick() {
    const auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
  }
};

int main(int argc, char* argv[]) {



  int num_dice = 1;
  int num_faces = 4;
  int fp_iters = 1024;
  int mdp_depth = 2;
  int num_threads = 10;
  int per_gpu = 1;
  int num_cycles = 6;
  std::string device = "cuda:1";
  std::string net_path;
  {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      
      // 骰子数量参数
      if (arg == "--num_dice") {
        assert(i + 1 < argc);  // 确保有下一个参数
        num_dice = std::stoi(argv[++i]);  // 将下一个参数转换为整数
      } 
      // 骰子面数参数
      else if (arg == "--num_faces") {
        assert(i + 1 < argc);
        num_faces = std::stoi(argv[++i]);
      } 
      // 固定点迭代次数参数
      else if (arg == "--fp_iters") {
        assert(i + 1 < argc);
        fp_iters = std::stoi(argv[++i]);
      } 
      // MDP（马尔可夫决策过程）深度参数
      else if (arg == "--mdp_depth") {
        assert(i + 1 < argc);
        mdp_depth = std::stoi(argv[++i]);
      } 
      // 线程数量参数
      else if (arg == "--num_threads") {
        assert(i + 1 < argc);
        num_threads = std::stoi(argv[++i]);
      } 
      // 每个GPU处理的样本数量参数
      else if (arg == "--per_gpu") {
        assert(i + 1 < argc);
        per_gpu = std::stoi(argv[++i]);
      } 
      // 训练循环次数参数
      else if (arg == "--num_cycles") {
        assert(i + 1 < argc);
        num_cycles = std::stoi(argv[++i]);
      } 
      // 计算设备参数（CPU/GPU）
      else if (arg == "--device") {
        assert(i + 1 < argc);
        device = argv[++i];  // 直接使用字符串，不需要转换为整数
      } 
      // 神经网络模型路径参数
      else if (arg == "--net") {
        assert(i + 1 < argc);
        net_path = argv[++i];  // 直接使用字符串路径
      } 
      // 未知参数处理
      else {
        std::cerr << "Unknown flag: " << arg << "\n";
        return -1;  // 返回错误码
      }
    }
  }
//创建一个游戏实例
  const Game game(num_dice, num_faces);
  std::cout << "num_dice=" << num_dice << " num_faces=" << num_faces << "\n";
  {
    const auto full_tree = unroll_tree(game);
    std::cout << "Tree of depth " << get_depth(full_tree) << " has "
              << full_tree.size() << " nodes\n";
  }
// 创建一个模型实例
  std::vector<TorchJitModel> models;
  for (int i = 0; i < per_gpu; ++i) {
    auto module = torch::jit::load(net_path);
    module.eval();
    module.to(device);
    models.push_back(module);
  }

// 1. 首先创建模型指针数组
std::vector<TorchJitModel*> model_ptrs;
for (int i = 0; i < per_gpu; ++i) {
    model_ptrs.push_back(&models[i]);  // 将每个模型的地址存入指针数组
}

// 2. 创建模型锁管理器
auto locker = std::make_shared<ModelLocker>(model_ptrs, device);
// - 使用智能指针管理 ModelLocker 对象
// - 传入模型指针数组和计算设备
// - 用于多线程环境下安全地访问模型

// 3. 创建优先经验回放缓冲区
auto replay = std::make_shared<ValuePrioritizedReplay>(
    1 << 20,    // 缓冲区大小：2^20 = 1,048,576 个样本
    1000,       // 最小缓冲区大小
    1.0,        // 优先级系数
    0.4,        // 采样系数
    3,          // 优先级幂次
    false,      // 不使用绝对优先级
    false       // 不使用相对优先级
);
// - 用于存储和管理训练数据
// - 根据样本的重要性进行优先级采样

// 4. 创建上下文管理器
auto context = std::make_shared<Context>();
// - 用于管理线程和资源


  RecursiveSolvingParams cfg;
  cfg.num_dice = num_dice;
  cfg.num_faces = num_faces;
  cfg.subgame_params.num_iters = fp_iters;
  cfg.subgame_params.linear_update = true;
  cfg.subgame_params.optimistic = false;
  cfg.subgame_params.max_depth = mdp_depth;

// 5. 创建数据线程循环
  for (int i = 0; i < num_threads; ++i) {
    const int seed = i;
    auto connector = std::make_shared<CVNetBufferConnector>(locker, replay);
    std::shared_ptr<ThreadLoop> loop =
        std::make_shared<DataThreadLoop>(std::move(connector), cfg, seed);
    context->pushThreadLoop(loop);
  }
  std::cout << "Starting the context" << std::endl;
  context->start();
  Timer t;
  for (int i = 0; i < num_cycles; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
    double secs = t.tick();
    auto added = replay->numAdd();
    std::cout << "time=" << secs << " "
              << "items=" << added << " per_second=" << added / secs << "\n";
  }
}