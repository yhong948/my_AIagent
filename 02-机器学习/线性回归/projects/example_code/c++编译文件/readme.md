# C++ 编译文件目录

本目录用于存放线性回归相关的 C++ 代码编译后的可执行文件。

---

## C++ 命令行编译运行模板

### 通用模板

```bash
# 第1步：编译
g++ -std=c++17 -O2 <源文件.cpp> -o <输出文件名> <链接库>

# 第2步：运行
./<输出文件名>
```

**参数说明：**

| 参数 | 含义 |
|------|------|
| `g++` | C++ 编译器 |
| `-std=c++17` | 使用 C++17 标准 |
| `-O2` | 开启二级优化（让程序跑更快） |
| `<源文件.cpp>` | 你的 C++ 源代码文件 |
| `-o <输出文件名>` | 指定编译输出的可执行文件名 |
| `-l<库名>` | 链接第三方库（如 `-larmadillo`） |
| `-I<路径>` | 指定头文件搜索路径（如 `-I/usr/include/eigen3`） |

---

### 实例1：线性回归工业级部署2.cpp（mlpack 调库版）

```bash
# 进入源码目录
cd /mnt/c/Users/LENOVO/Desktop/my_AIagent/02-机器学习/线性回归/projects/example_code

# 编译（链接 armadillo + mlpack + boost 三个库）
g++ -std=c++17 -O2 '线性回归工业级部署2.cpp' -o 'c++编译文件/lr_sklearn' -larmadillo -lmlpack -lboost_serialization

# 运行
./c++编译文件/lr_sklearn
```

**用到的库：**
- `-larmadillo`：Armadillo 线性代数库（C++ 的 NumPy）
- `-lmlpack`：mlpack 机器学习库（C++ 的 sklearn）
- `-lboost_serialization`：Boost 序列化（mlpack 的依赖）

---

### 实例2：线性回归工业级部署.cpp（Eigen 手搓版）

```bash
# 编译（用 -I 指定 Eigen3 头文件路径）
g++ -std=c++17 -O2 -I/usr/include/eigen3 '线性回归工业级部署.cpp' -o 'c++编译文件/lr_eigen'

# 运行
./c++编译文件/lr_eigen
```

**用到的库：**
- `-I/usr/include/eigen3`：Eigen3 是纯头文件库，只需指定路径，不需要 `-l` 链接

---

### 常见问题

| 问题 | 解决方法 |
|------|----------|
| `fatal error: xxx.hpp: No such file` | 库没装，用 `sudo apt-get install lib<库名>-dev` 安装 |
| `undefined reference to ...` | 忘记链接库了，加上 `-l<库名>` |
| 中文文件名编译报错 | 用单引号包裹文件名：`'中文名.cpp'` |
| 编译很慢 | 模板库（Eigen/mlpack）编译慢是正常的，耐心等 |
