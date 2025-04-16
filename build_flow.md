# CMake构建流程图

```mermaid
graph TD
    A[开始] --> B[设置CMake最低版本3.0]
    B --> C[创建cfr项目]
    C --> D[设置C++17标准]
    
    D --> E[设置编译选项]
    E -->|必要参数| F1[-I ..]
    E -->|必要参数| F2[-Wall -Wextra]
    E -->|必要参数| F3[-fPIC]
    E -->|必要参数| F4[-O3]
    
    D --> G[查找Python依赖]
    G --> G1[Python解释器 >= 3.7]
    G --> G2[Python库 >= 3.7]
    
    D --> H[查找PyTorch]
    H --> H1[获取PyTorch路径]
    H --> H2[设置CUDA架构]
    H --> H3[加载PyTorch包]
    
    subgraph 主要库构建
        I[构建liars_dice_lib静态库]
        I --> I1[liars_dice]
        I --> I2[subgame_solving]
        I --> I3[real_net]
        I --> I4[recursive_solving]
        I --> I5[stats]
    end
    
    subgraph Python绑定
        J[构建_rela库]
        J --> J1[编译rela/types.cc]
        J --> J2[设置包含目录]
        J --> J3[链接PyTorch]
        
        K[构建Python模块rela]
        K --> K1[编译rela/pybind.cc]
        K --> K2[链接pybind11]
        K --> K3[链接_rela]
        K --> K4[链接liars_dice_lib]
    end
    
    H3 --> I
    I --> J
    J --> K
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
    style K fill:#bfb,stroke:#333,stroke-width:2px
