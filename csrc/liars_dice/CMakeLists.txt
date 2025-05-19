cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(cfr)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -I .. -Wall -Wextra -fPIC -Wfatal-errors -O3")
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# message(当前路径：${CMAKE_SOURCE_DIR})
# 查找 Python 相关包
find_package(PythonInterp 3.7 REQUIRED)  # 查找 Python 解释器
find_package(PythonLibs 3.7 REQUIRED)    # 查找 Python 库
if (NOT TARGET torch_library)
  execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TorchPath
  )
  list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
  set(TORCH_CUDA_ARCH_LIST "6.0;7.0")
  find_package(Torch REQUIRED)
endif()

add_library(liars_dice_lib liars_dice subgame_solving real_net recursive_solving stats)
target_link_libraries(liars_dice_lib torch)
set_target_properties(liars_dice_lib PROPERTIES CXX_STANDARD 17)
add_subdirectory(${CMAKE_SOURCE_DIR}/../../third_party/pybind11 third_party/pybind11)

add_library(_rela rela/types.cc)
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

# add_executable(gen_benchmark gen_benchmark)
# # 链接必要的库
# target_link_libraries(gen_benchmark liars_dice_lib _rela)


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
