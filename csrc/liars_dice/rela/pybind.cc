#include <stdio.h>  // 标准输入输出头文件

#include <pybind11/pybind11.h>  // Pybind11主头文件，用于C++和Python的绑定
#include <pybind11/stl.h>  // Pybind11 STL容器支持

#include <torch/extension.h>  // PyTorch扩展支持

#include "real_net.h"  // 神经网络实现
#include "recursive_solving.h"  // 递归求解相关
#include "stats.h"  // 统计相关

#include "rela/context.h"  // 上下文管理
#include "rela/data_loop.h"  // 数据循环处理
#include "rela/prioritized_replay.h"  // 优先经验回放
#include "rela/thread_loop.h"  // 线程循环

namespace py = pybind11;  // 命名空间别名
using namespace rela;  // 使用rela命名空间

namespace {  // 匿名命名空间

// 创建CFR线程
std::shared_ptr<ThreadLoop> create_cfr_thread(
    std::shared_ptr<ModelLocker> modelLocker,  // 模型锁
    std::shared_ptr<ValuePrioritizedReplay> replayBuffer,  // 回放缓冲区
    const liars_dice::RecursiveSolvingParams& cfg,  // 配置参数
    int seed) {  // 随机种子
  auto connector =
      std::make_shared<CVNetBufferConnector>(modelLocker, replayBuffer);
  return std::make_shared<DataThreadLoop>(std::move(connector), cfg, seed);
}

// 计算可剥削性（使用神经网络）
float compute_exploitability(liars_dice::RecursiveSolvingParams params,
                             const std::string& model_path) {
  py::gil_scoped_release release;  // 释放GIL锁
  liars_dice::Game game(params.num_dice, params.num_faces);  // 创建游戏实例
  std::shared_ptr<IValueNet> net =
      liars_dice::create_torchscript_net(model_path);  // 加载神经网络
  const auto tree_strategy =
      compute_strategy_recursive(game, params.subgame_params, net);  // 计算策略
  liars_dice::print_strategy(game, unroll_tree(game), tree_strategy);  // 打印策略
  std::cout << "here : " << std::endl ;
  return liars_dice::compute_exploitability(game, tree_strategy);  // 返回可剥削性
}

// 使用神经网络计算统计信息
auto compute_stats_with_net(liars_dice::RecursiveSolvingParams params,
                            const std::string& model_path) {
                              // const std::string& model_path ,bool test=false) {
  py::gil_scoped_release release;  // 释放GIL锁
  std::cout << "params.num_iters = " << params.subgame_params.num_iters
            << std::endl;

  // params.subgame_params.num_iters = 1024;
  liars_dice::Game game(params.num_dice, params.num_faces);  // 创建游戏实例
  std::shared_ptr<IValueNet> net =
      liars_dice::create_torchscript_net(model_path);  // 加载神经网络
  const auto net_strategy =
      compute_strategy_recursive_to_leaf(game, params.subgame_params, net);  // 计算策略
  liars_dice::print_strategy(game, unroll_tree(game), net_strategy);  // 打印策略
// // 创建随机策略
// std::vector<float> random_strategy(game.num_actions(), 1.0f / game.num_actions());
  const float explotability =
      liars_dice::compute_exploitability(game, net_strategy);  // 计算可剥削性

//   auto full_params = params.subgame_params;  // 获取子游戏参数
//   full_params.max_depth = 100000;  // 设置最大深度
//   auto fp = build_solver(game, full_params);  // 构建求解器
//   fp->multistep();  // 多步求解
//   const auto& full_strategy = fp->get_strategy();  // 获取完整策略

//   // 评估网络遍历的MSE
//   const float mse_net_traverse = eval_net(
//       game, net_strategy, full_strategy, params.subgame_params.max_depth,
//       params.subgame_params.num_iters, net, /*traverse_by_net=*/true,
//       /*verbose=*/true);
//   // 评估完整遍历的MSE
//   const float mse_full_traverse = eval_net(
//       game, net_strategy, full_strategy, params.subgame_params.max_depth,
//       params.subgame_params.num_iters, net, /*traverse_by_net=*/false,
//       /*verbose=*/true);
//   return std::make_tuple(explotability, mse_net_traverse, mse_full_traverse);  // 返回结果元组
  return std::make_tuple(explotability, 0.0, 0.0);  // 返回结果元组
}

// 计算可剥削性（不使用神经网络）
float compute_exploitability_no_net(liars_dice::RecursiveSolvingParams params) {
  py::gil_scoped_release release;  // 释放GIL锁
  liars_dice::Game game(params.num_dice, params.num_faces);  // 创建游戏实例
  auto fp = liars_dice::build_solver(game, game.get_initial_state(),
                                     liars_dice::get_initial_beliefs(game),
                                     params.subgame_params, /*net=*/nullptr);  // 构建求解器
  float values[2] = {0.0};  // 初始化值数组
  for (int iter = 0; iter < params.subgame_params.num_iters; ++iter) {  // 迭代求解
    if (((iter + 1) & iter) == 0 ||
        iter + 1 == params.subgame_params.num_iters) {
      auto values = compute_exploitability2(game, fp->get_strategy());  // 计算可剥削性
      printf("Iter=%8d exploitabilities=(%.3e, %.3e) sum=%.3e\n", iter + 1,
             values[0], values[1], (values[0] + values[1]) / 2.);  // 打印结果
    }
    // 检查Ctrl-C信号
    if (PyErr_CheckSignals() != 0) throw py::error_already_set();
  }
  liars_dice::print_strategy(game, unroll_tree(game), fp->get_strategy());  // 打印策略
  return values[0] + values[1];  // 返回可剥削性总和
}

}  // namespace
// Pybind11模块定义，将C++代码绑定到Python
PYBIND11_MODULE(rela, m) {
  // 定义ValueTransition类，用于存储值转换数据
  py::class_<ValueTransition, std::shared_ptr<ValueTransition>>(
      m, "ValueTransition")
      .def(py::init<>())  // 定义构造函数
      .def_readwrite("query", &ValueTransition::query)  // 定义可读写属性query
      .def_readwrite("values", &ValueTransition::values);  // 定义可读写属性values

  // 定义ValuePrioritizedReplay类，实现优先经验回放
  py::class_<ValuePrioritizedReplay, std::shared_ptr<ValuePrioritizedReplay>>(
      m, "ValuePrioritizedReplay")
      .def(py::init<int, int, float, float, int, bool, bool>(),  // 定义构造函数，包含7个参数
           py::arg("capacity"), py::arg("seed"), py::arg("alpha"),
           py::arg("beta"), py::arg("prefetch"), py::arg("use_priority"),
           py::arg("compressed_values"))
      .def("size", &ValuePrioritizedReplay::size)  // 定义size方法
      .def("num_add", &ValuePrioritizedReplay::numAdd)  // 定义num_add方法
      .def("sample", &ValuePrioritizedReplay::sample)  // 定义sample方法
      .def("pop_until", &ValuePrioritizedReplay::popUntil)  // 定义pop_until方法
      .def("load", &ValuePrioritizedReplay::load)  // 定义load方法
      .def("save", &ValuePrioritizedReplay::save)  // 定义save方法
      .def("extract", &ValuePrioritizedReplay::extract)  // 定义extract方法
      .def("push", &ValuePrioritizedReplay::push,  // 定义push方法
           py::call_guard<py::gil_scoped_release>())  // 释放GIL锁
      .def("update_priority", &ValuePrioritizedReplay::updatePriority);  // 定义update_priority方法

  // 定义ThreadLoop基类
  py::class_<ThreadLoop, std::shared_ptr<ThreadLoop>>(m, "ThreadLoop");

  // 定义SubgameSolvingParams类，用于配置子游戏求解参数
  py::class_<liars_dice::SubgameSolvingParams>(m, "SubgameSolvingParams")
      .def(py::init<>())  // 定义构造函数
      .def_readwrite("num_iters", &liars_dice::SubgameSolvingParams::num_iters)  // 定义迭代次数参数
      .def_readwrite("max_depth", &liars_dice::SubgameSolvingParams::max_depth)  // 定义最大深度参数
      .def_readwrite("linear_update", &liars_dice::SubgameSolvingParams::linear_update)  // 定义线性更新参数
      .def_readwrite("optimistic", &liars_dice::SubgameSolvingParams::optimistic)  // 定义乐观参数
      .def_readwrite("use_cfr", &liars_dice::SubgameSolvingParams::use_cfr)  // 定义是否使用CFR
      .def_readwrite("dcfr", &liars_dice::SubgameSolvingParams::dcfr)  // 定义是否使用DCFR
      .def_readwrite("dcfr_alpha", &liars_dice::SubgameSolvingParams::dcfr_alpha)  // 定义DCFR alpha参数
      .def_readwrite("dcfr_beta", &liars_dice::SubgameSolvingParams::dcfr_beta)  // 定义DCFR beta参数
      .def_readwrite("dcfr_gamma", &liars_dice::SubgameSolvingParams::dcfr_gamma);  // 定义DCFR gamma参数

  // 定义RecursiveSolvingParams类，用于配置递归求解参数
  py::class_<liars_dice::RecursiveSolvingParams>(m, "RecursiveSolvingParams")
      .def(py::init<>())  // 定义构造函数
      .def_readwrite("num_dice", &liars_dice::RecursiveSolvingParams::num_dice)  // 定义骰子数量
      .def_readwrite("num_faces", &liars_dice::RecursiveSolvingParams::num_faces)  // 定义骰子面数
      .def_readwrite("random_action_prob", &liars_dice::RecursiveSolvingParams::random_action_prob)  // 定义随机动作概率
      .def_readwrite("sample_leaf", &liars_dice::RecursiveSolvingParams::sample_leaf)  // 定义是否采样叶子节点
      .def_readwrite("subgame_params", &liars_dice::RecursiveSolvingParams::subgame_params);  // 定义子游戏参数

  // 定义DataThreadLoop类，继承自ThreadLoop
  py::class_<DataThreadLoop, ThreadLoop, std::shared_ptr<DataThreadLoop>>(
      m, "DataThreadLoop")
      .def(py::init<std::shared_ptr<CVNetBufferConnector>,
                    const liars_dice::RecursiveSolvingParams&, int>(),
           py::arg("connector"), py::arg("params"), py::arg("thread_id"));  // 定义构造函数

  // 定义Context类，用于管理线程上下文
  py::class_<rela::Context>(m, "Context")
      .def(py::init<>())  // 定义构造函数
      .def("push_env_thread", &rela::Context::pushThreadLoop,  // 定义push_env_thread方法
           py::keep_alive<1, 2>())  // 保持对象存活
      .def("start", &rela::Context::start)  // 定义start方法
      .def("pause", &rela::Context::pause)  // 定义pause方法
      .def("resume", &rela::Context::resume)  // 定义resume方法
      .def("terminate", &rela::Context::terminate)  // 定义terminate方法
      .def("terminated", &rela::Context::terminated);  // 定义terminated方法

  // 定义ModelLocker类，用于管理模型锁
  py::class_<ModelLocker, std::shared_ptr<ModelLocker>>(m, "ModelLocker")
      .def(py::init<std::vector<py::object>, const std::string&>())  // 定义构造函数
      .def("update_model", &ModelLocker::updateModel);  // 定义update_model方法

  // 定义Python可调用的函数
  m.def("compute_exploitability_fp", &compute_exploitability_no_net,  // 定义计算可剥削性函数(无网络)
        py::arg("params"));

  m.def("compute_exploitability_with_net", &compute_exploitability,  // 定义计算可剥削性函数(带网络)
        py::arg("params"), py::arg("model_path"));

  m.def("compute_stats_with_net", &compute_stats_with_net,  // 定义计算统计信息函数
        py::arg("params"), py::arg("model_path"));  // 默认参数test为false
        // py::arg("params"), py::arg("model_path"), py::arg("test") = false);  // 默认参数test为false
  m.def("create_cfr_thread", &create_cfr_thread,  // 定义创建CFR线程函数
        py::arg("model_locker"), py::arg("replay"), py::arg("cfg"), py::arg("seed"));

  // 注释掉的函数定义
  //   m.def("create_value_policy_agent", &create_value_policy_agent,
  //         py::arg("model_locker"), py::arg("replay"),
  //         py::arg("policy_replay"),
  //         py::arg("compress_policy_values"));
}
