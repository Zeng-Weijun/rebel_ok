cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cfr)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -I .. -Wall -mcmodel=medium -march=native -Wextra -Wno-register -fPIC -Wfatal-errors -O3")
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/../../third_party/googletest third_party/googletest EXCLUDE_FROM_ALL)
message(当前路径：${CMAKE_SOURCE_DIR})
# 查找 Python 相关包
find_package(PythonInterp 3.7 REQUIRED)  # 查找 Python 解释器
find_package(PythonLibs 3.7 REQUIRED)    # 查找 Python 库

if (NOT TARGET torch_library)
  # 执行 Python 命令获取 PyTorch 安装路径
  execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
  )
  # 将 PyTorch 路径添加到 CMake 搜索路径
  list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
  # 设置 CUDA 架构列表，支持 Pascal 和 Volta 架构
  set(TORCH_CUDA_ARCH_LIST "6.0;7.0")
  # 查找 PyTorch 包
  find_package(Torch REQUIRED)
endif()

# 创建 liars_dice_lib 静态库
# 包含多个源文件：liars_dice, subgame_solving, real_net, recursive_solving, stats
add_library(liars_dice_lib liars_dice subgame_solving real_net recursive_solving stats)
# 将 liars_dice_lib 链接到 PyTorch
target_link_libraries(liars_dice_lib torch)
# 设置 liars_dice_lib 使用 C++17 标准
set_target_properties(liars_dice_lib PROPERTIES CXX_STANDARD 17)

# 创建 recursive_eval 可执行文件
add_executable(recursive_eval recursive_eval)
# 将 recursive_eval 链接到 liars_dice_lib
target_link_libraries(recursive_eval liars_dice_lib)

# 添加 pybind11 作为子项目
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/../../third_party/pybind11 third_party/pybind11
)

add_library(_rela
  rela/types.cc
)
# 设置 _rela 的包含目录
target_include_directories(_rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(_rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(_rela PUBLIC $ENV{CONDA_PREFIX}/include)
# 将 _rela 链接到 PyTorch
target_link_libraries(_rela torch)

pybind11_add_module(rela rela/pybind.cc)

target_link_libraries(rela 
    PRIVATE
    pybind11::module  # 关键：链接 pybind11 目标
    PUBLIC 
    _rela 
    liars_dice_lib
)

add_executable(gen_benchmark gen_benchmark)
# 链接必要的库
target_link_libraries(gen_benchmark liars_dice_lib _rela)


# include(GoogleTest)
# # 启用测试
# enable_testing()

# add_executable(liar_game_test liars_dice_test.cc)
# target_link_libraries(liar_game_test liars_dice_lib gtest_main)
# add_test(NAME liar_game COMMAND liar_game_test)


# add_executable(liar_tree_test tree_test.cc)
# target_link_libraries(liar_tree_test liars_dice_lib gtest_main)
# add_test(NAME liar_tree COMMAND liar_tree_test)

# # liar_recursive_solving_test
# add_executable(liar_recursive_solving_test recursive_solving_test.cc)
# target_link_libraries(liar_recursive_solving_test liars_dice_lib gtest_main)
# add_test(NAME liar_recursive_solving COMMAND liar_recursive_solving_test)

# # liar_subgame_solving_test
# add_executable(liar_subgame_solving_test subgame_solving_test.cc)
# target_link_libraries(liar_subgame_solving_test liars_dice_lib gtest_main)
# add_test(NAME liar_subgame_solving COMMAND liar_subgame_solving_test)
