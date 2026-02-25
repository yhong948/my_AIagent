/*
 * C++ 调库实现线性回归（类似 Python sklearn 版本）
 * 
 * 使用 mlpack 库（C++ 的 sklearn）+ Armadillo（C++ 的 NumPy）
 * 
 * 对应 Python 文件：线性回归实际应用.ipynb
 * 
 * 所需库：
 * 1. mlpack：C++ 机器学习库（类似 sklearn）
 *    安装：sudo apt-get install libmlpack-dev
 * 2. Armadillo：线性代数库（类似 NumPy，mlpack 的依赖）
 *    安装：sudo apt-get install libarmadillo-dev
 * 
 * 编译命令：
 *    g++ -std=c++17 -O2 线性回归工业级部署2.cpp -o lr_sklearn -larmadillo -lmlpack -lboost_serialization
 * 
 * Python vs C++ 对照表：
 * ┌──────────────────────────┬──────────────────────────────────┐
 * │ Python (sklearn)         │ C++ (mlpack)                     │
 * ├──────────────────────────┼──────────────────────────────────┤
 * │ import numpy as np       │ #include <armadillo>             │
 * │ from sklearn import ...  │ #include <mlpack/...>            │
 * │ np.array / np.ndarray    │ arma::mat / arma::rowvec         │
 * │ train_test_split()       │ mlpack::data::Split()            │
 * │ StandardScaler()         │ mlpack::data::StandardScaler     │
 * │ LinearRegression()       │ mlpack::regression::LinearRegression │
 * │ model.fit(X, y)          │ model.Train(X, y) 或构造函数直接训练 │
 * │ model.predict(X)         │ model.Predict(X, predictions)    │
 * │ model.coef_              │ model.Parameters()               │
 * └──────────────────────────┴──────────────────────────────────┘
 * 
 * 注意：Armadillo 的矩阵是列优先的！
 *   - Python sklearn: X 形状为 (n_samples, n_features)，每行是一个样本
 *   - C++ mlpack:     X 形状为 (n_features, n_samples)，每列是一个样本
 *   这是 C++ 和 Python 最大的区别，转置了！
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <armadillo>  // 类似 NumPy

// mlpack 头文件（类似 from sklearn import ...）
#include <mlpack/core.hpp>                                     // 核心功能
#include <mlpack/core/data/split_data.hpp>                     // train_test_split
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp> // StandardScaler
#include <mlpack/methods/linear_regression/linear_regression.hpp>  // LinearRegression

using namespace std;
using namespace arma;

// ============================================================
// 工具函数
// ============================================================

// 计算 MSE（均方误差）—— 对应 sklearn.metrics.mean_squared_error
double mean_squared_error(const rowvec& y_true, const rowvec& y_pred) {
    return arma::mean(arma::square(y_true - y_pred));
}

// 计算 R²（决定系数）—— 对应 sklearn.metrics.r2_score
double r2_score(const rowvec& y_true, const rowvec& y_pred) {
    double ss_residual = arma::accu(arma::square(y_true - y_pred));
    double ss_total = arma::accu(arma::square(y_true - arma::mean(y_true)));
    return 1.0 - (ss_residual / ss_total);
}

// 生成模拟的"加州房价"数据（因为 C++ 没有内置数据集）
// 模拟 housing = fetch_california_housing()
// 返回：X (n_features x n_samples)，y (1 x n_samples)，feature_names
struct HousingData {
    mat X;                          // 特征矩阵 (n_features x n_samples)
    rowvec y;                       // 目标值 (1 x n_samples)
    vector<string> feature_names;   // 特征名称
};

HousingData generate_california_housing(int n_samples = 1000) {
    // 模拟加州房价数据集的两个关键特征
    // MedInc（收入中位数）和 HouseAge（房龄）
    
    random_device rd;
    mt19937 gen(42);  // random_state=42，和 Python 版保持一致
    
    // 特征分布（模拟真实数据的统计特性）
    normal_distribution<> income_dist(3.87, 1.9);     // MedInc: 均值3.87, 标准差1.9
    normal_distribution<> age_dist(28.6, 12.6);       // HouseAge: 均值28.6, 标准差12.6
    normal_distribution<> noise(0, 0.3);              // 噪声
    
    mat X(2, n_samples);  // 2个特征 x n_samples（注意：mlpack 是列优先）
    rowvec y(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        double med_income = max(0.5, income_dist(gen));  // 收入不能太低
        double house_age = max(1.0, min(52.0, age_dist(gen)));  // 房龄1-52年
        
        X(0, i) = med_income;   // 第0行：MedInc
        X(1, i) = house_age;    // 第1行：HouseAge
        
        // 房价 ≈ 0.42 * 收入 + 0.01 * 房龄 + 噪声（模拟真实关系）
        y(i) = max(0.15, 0.42 * med_income + 0.01 * house_age + 0.5 + noise(gen));
    }
    
    return {X, y, {"MedInc", "HouseAge"}};
}

// ============================================================
// 主程序
// ============================================================
int main() {
    cout << "=== C++ 调库实现线性回归（mlpack 版，对标 sklearn）===" << endl;
    cout << "对应 Python 文件：线性回归实际应用.ipynb\n" << endl;
    
    // --------------------------------------------------------
    // 第1步：导入和准备数据
    // Python: housing = fetch_california_housing()
    //         X_real = housing.data[:, :2]
    //         y_real = housing.target
    // --------------------------------------------------------
    cout << "1. 加载数据..." << endl;
    
    HousingData housing = generate_california_housing(1000);
    mat& X_real = housing.X;       // (2, 1000) — 注意是转置的！
    rowvec& y_real = housing.y;    // (1, 1000)
    
    cout << "   数据集形状: (" << X_real.n_cols << ", " << X_real.n_rows << ")" << endl;
    cout << "   特征名: [" << housing.feature_names[0] 
         << ", " << housing.feature_names[1] << "]" << endl;
    cout << "   前5个样本的目标值: ";
    for (int i = 0; i < 5; ++i) cout << fixed << setprecision(2) << y_real(i) << " ";
    cout << endl;
    
    // --------------------------------------------------------
    // 第2步：划分训练集/测试集
    // Python: X_train, X_test, y_train, y_test = train_test_split(
    //             X_real, y_real, test_size=0.2, random_state=42)
    // --------------------------------------------------------
    cout << "\n2. 划分训练集/测试集（80%训练, 20%测试）..." << endl;
    
    mat X_train, X_test;
    rowvec y_train, y_test;
    
    // mlpack::data::Split —— 对应 sklearn 的 train_test_split
    // 参数：输入数据, 标签, 训练数据, 测试数据, 训练标签, 测试标签, 测试比例
    mlpack::data::Split(X_real, y_real, 
                        X_train, X_test, 
                        y_train, y_test, 
                        0.2);  // test_size=0.2
    
    cout << "   训练集: " << X_train.n_cols << " 个样本" << endl;
    cout << "   测试集: " << X_test.n_cols << " 个样本" << endl;
    
    // --------------------------------------------------------
    // 第3步：数据标准化
    // Python: scaler = StandardScaler()
    //         X_train_scaled = scaler.fit_transform(X_train)
    //         X_test_scaled = scaler.transform(X_test)
    // --------------------------------------------------------
    cout << "\n3. 数据标准化（Z-score）..." << endl;
    
    // mlpack::data::StandardScaler —— 对应 sklearn 的 StandardScaler
    mlpack::data::StandardScaler scaler;
    
    mat X_train_scaled, X_test_scaled;
    
    // Fit + Transform 训练集
    scaler.Fit(X_train);                          // scaler.fit(X_train)
    scaler.Transform(X_train, X_train_scaled);    // X_train_scaled = scaler.transform(X_train)
    scaler.Transform(X_test, X_test_scaled);      // X_test_scaled = scaler.transform(X_test)
    
    cout << "   标准化完成！" << endl;
    cout << "   标准化前 X_train 第1列: [" << X_train(0,0) << ", " << X_train(1,0) << "]" << endl;
    cout << "   标准化后 X_train 第1列: [" << X_train_scaled(0,0) << ", " << X_train_scaled(1,0) << "]" << endl;
    
    // --------------------------------------------------------
    // 第4步：创建并训练模型
    // Python: sklearn_model = LinearRegression()
    //         sklearn_model.fit(X_train_scaled, y_train)
    // --------------------------------------------------------
    cout << "\n4. 训练线性回归模型（mlpack 调库）..." << endl;
    
    // mlpack::regression::LinearRegression —— 对应 sklearn 的 LinearRegression
    // 构造函数直接训练！（也可以用 .Train() 方法）
    mlpack::regression::LinearRegression model(X_train_scaled, y_train);
    
    cout << "   训练完成！" << endl;
    
    // --------------------------------------------------------
    // 第5步：预测
    // Python: y_train_pred = sklearn_model.predict(X_train_scaled)
    //         y_test_pred = sklearn_model.predict(X_test_scaled)
    // --------------------------------------------------------
    cout << "\n5. 模型预测..." << endl;
    
    rowvec y_train_pred, y_test_pred;
    model.Predict(X_train_scaled, y_train_pred);  // sklearn_model.predict(X_train_scaled)
    model.Predict(X_test_scaled, y_test_pred);    // sklearn_model.predict(X_test_scaled)
    
    // --------------------------------------------------------
    // 第6步：评估模型
    // Python: train_mse = mean_squared_error(y_train, y_train_pred)
    //         test_r2 = r2_score(y_test, y_test_pred)
    // --------------------------------------------------------
    double train_mse = mean_squared_error(y_train, y_train_pred);
    double test_mse = mean_squared_error(y_test, y_test_pred);
    double train_r2 = r2_score(y_train, y_train_pred);
    double test_r2 = r2_score(y_test, y_test_pred);
    
    cout << "\n===== mlpack 模型结果 =====" << endl;
    
    // 获取模型参数
    // Python: sklearn_model.coef_ 和 sklearn_model.intercept_
    // mlpack: Parameters() 返回向量，第一个元素是 intercept（偏置），后面是权重
    vec params = model.Parameters();
    double intercept = params(0);              // sklearn_model.intercept_
    vec weights = params.subvec(1, params.n_elem - 1);  // sklearn_model.coef_
    
    cout << "权重 w = [";
    for (size_t i = 0; i < weights.n_elem; ++i) {
        cout << fixed << setprecision(4) << weights(i);
        if (i < weights.n_elem - 1) cout << ", ";
    }
    cout << "]" << endl;
    cout << "偏置 b = " << fixed << setprecision(4) << intercept << endl;
    
    cout << "\n训练集 MSE = " << fixed << setprecision(4) << train_mse 
         << ", R² = " << train_r2 << endl;
    cout << "测试集 MSE = " << fixed << setprecision(4) << test_mse 
         << ", R² = " << test_r2 << endl;
    
    // --------------------------------------------------------
    // 第7步：特征重要性分析
    // Python: feature_importance = pd.DataFrame({
    //             '特征': housing.feature_names[:2],
    //             '权重': sklearn_model.coef_
    //         }).sort_values('权重', key=abs, ascending=False)
    // --------------------------------------------------------
    cout << "\n===== 特征重要性排序 =====" << endl;
    
    // 创建特征-权重对，按绝对值降序排序
    vector<pair<string, double>> feature_importance;
    for (size_t i = 0; i < housing.feature_names.size(); ++i) {
        feature_importance.push_back({housing.feature_names[i], weights(i)});
    }
    
    // 按权重绝对值降序排序 —— 对应 .sort_values('权重', key=abs, ascending=False)
    sort(feature_importance.begin(), feature_importance.end(),
         [](const pair<string, double>& a, const pair<string, double>& b) {
             return abs(a.second) > abs(b.second);
         });
    
    cout << left << setw(15) << "特征" << setw(15) << "权重" << endl;
    cout << string(30, '-') << endl;
    for (const auto& [name, weight] : feature_importance) {
        cout << left << setw(15) << name 
             << fixed << setprecision(4) << weight << endl;
    }
    
    // --------------------------------------------------------
    // 额外：对比手搓版 vs 调库版
    // --------------------------------------------------------
    cout << "\n===== Python vs C++ 代码对比总结 =====" << endl;
    cout << "┌─────────────────────────────┬──────────────────────────────────┐" << endl;
    cout << "│ Python 写法                 │ C++ (mlpack) 写法                │" << endl;
    cout << "├─────────────────────────────┼──────────────────────────────────┤" << endl;
    cout << "│ model = LinearRegression()  │ LinearRegression model(X, y);   │" << endl;
    cout << "│ model.fit(X, y)             │ （构造函数直接训练）               │" << endl;
    cout << "│ model.predict(X)            │ model.Predict(X, pred);         │" << endl;
    cout << "│ model.coef_                 │ model.Parameters()              │" << endl;
    cout << "│ model.intercept_            │ model.Parameters()(0)           │" << endl;
    cout << "│ StandardScaler()            │ data::StandardScaler scaler;    │" << endl;
    cout << "│ scaler.fit_transform(X)     │ scaler.Fit(X)+Transform(X,out) │" << endl;
    cout << "│ train_test_split(X,y,0.2)   │ data::Split(X,y,...,0.2)       │" << endl;
    cout << "└─────────────────────────────┴──────────────────────────────────┘" << endl;
    
    return 0;
}
