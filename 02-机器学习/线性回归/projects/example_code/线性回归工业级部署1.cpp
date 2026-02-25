/*
 * 所需库：
 * 1. Eigen3：线性代数运算（类似NumPy）
 *   安装：sudo apt-get install libeigen3-dev
 * 2. 可选：Matplot++ 用于可视化（类似matplotlib）
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <Eigen/Dense>  // Eigen库核心

using namespace std;
using namespace Eigen;

// 生成模拟数据函数
pair<MatrixXd, VectorXd> generate_linear_data(int n_samples, double noise_std = 1.0) {
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> noise(0, noise_std);
    
    MatrixXd X(n_samples, 1);
    VectorXd y(n_samples);
    
    double true_w = 3.0;
    double true_b = 4.0;
    
    for(int i = 0; i < n_samples; ++i) {
        X(i, 0) = 2.0 * i / n_samples;  // 0到2之间的均匀值
        y(i) = true_b + true_w * X(i, 0) + noise(gen);
    }
    
    return {X, y};
}

// 从零实现的线性回归类
class LinearRegressionCPP {
private:
    VectorXd weights;   // 权重
    double bias;        // 偏置
    double learning_rate;
    int n_iterations;
    vector<double> loss_history;
    
public:
    LinearRegressionCPP(double lr = 0.01, int iters = 1000) 
        : learning_rate(lr), n_iterations(iters) {}
    
    // 特征标准化（Z-score标准化）
    pair<MatrixXd, VectorXd> standardize(const MatrixXd& X, const VectorXd& y) {
        MatrixXd X_std = X;
        VectorXd y_std = y;
        
        // 标准化X
        VectorXd mean = X.colwise().mean();
        VectorXd std = ((X.rowwise() - mean.transpose()).array().square().colwise().sum() / X.rows()).sqrt();
        
        for(int i = 0; i < X.cols(); ++i) {
            if(std(i) > 1e-8) {
                X_std.col(i) = (X.col(i).array() - mean(i)) / std(i);
            }
        }
        
        // 标准化y（可选，通常只标准化特征）
        double y_mean = y.mean();
        double y_stddev = sqrt((y.array() - y_mean).square().sum() / y.size());
        y_std = ((y.array() - y_mean) / y_stddev).matrix();
        
        return {X_std, y_std};
    }
    
    // 训练函数（批量梯度下降）
    void fit(const MatrixXd& X, const VectorXd& y) {
        int n_samples = X.rows();
        int n_features = X.cols();
        
        // 初始化参数
        weights = VectorXd::Zero(n_features);
        bias = 0.0;
        loss_history.clear();
        
        auto start = chrono::high_resolution_clock::now();
        
        // 梯度下降迭代
        for(int iter = 0; iter < n_iterations; ++iter) {
            // 前向传播：计算预测值
            VectorXd y_pred = X * weights + VectorXd::Constant(n_samples, bias);
            
            // 计算损失（MSE）
            VectorXd error = y_pred - y;
            double loss = error.array().square().mean();
            loss_history.push_back(loss);
            
            // 计算梯度
            VectorXd dw = (2.0 / n_samples) * (X.transpose() * error);
            double db = (2.0 / n_samples) * error.sum();
            
            // 更新参数
            weights -= learning_rate * dw;
            bias -= learning_rate * db;
            
            // 打印进度
            if(iter % 100 == 0) {
                cout << "Iteration " << iter << ": Loss = " << loss << endl;
            }
        }
        
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        cout << "\n训练完成！耗时: " << duration.count() << " ms" << endl;
    }
    
    // 预测函数
    VectorXd predict(const MatrixXd& X) {
        return X * weights + VectorXd::Constant(X.rows(), bias);
    }
    
    // 评估函数：计算MSE和R²
    pair<double, double> evaluate(const MatrixXd& X, const VectorXd& y) {
        VectorXd y_pred = predict(X);
        
        // 计算MSE
        double mse = (y_pred - y).array().square().mean();
        
        // 计算R²
        double y_mean = y.mean();
        double ss_total = (y.array() - y_mean).square().sum();
        double ss_residual = (y.array() - y_pred.array()).square().sum();
        double r2 = 1.0 - (ss_residual / ss_total);
        
        return {mse, r2};
    }
    
    // 获取参数
    VectorXd get_weights() { return weights; }
    double get_bias() { return bias; }
    vector<double> get_loss_history() { return loss_history; }
};

// 使用Eigen内置的最小二乘法（解析解）
pair<VectorXd, double> linear_regression_closed_form(const MatrixXd& X, const VectorXd& y) {
    // 添加偏置项（增加一列1）
    MatrixXd X_aug(X.rows(), X.cols() + 1);
    X_aug << MatrixXd::Ones(X.rows(), 1), X;
    
    // 解析解：w = (XᵀX)⁻¹Xᵀy
    VectorXd w = (X_aug.transpose() * X_aug).ldlt().solve(X_aug.transpose() * y);
    
    // 分离偏置和权重
    double bias = w(0);
    VectorXd weights = w.tail(w.size() - 1);
    
    return {weights, bias};
}

int main() {
    cout << "=== C++ 线性回归实现 ===" << endl;
    
    // 1. 生成数据
    int n_samples = 100;
    auto [X_raw, y_raw] = generate_linear_data(n_samples);
    cout << "\n1. 数据生成完成" << endl;
    cout << "   数据形状: " << X_raw.rows() << " × " << X_raw.cols() << endl;
    
    // 2. 实例化模型
    LinearRegressionCPP model(0.1, 1000);
    
    // 3. 标准化数据
    auto [X_std, y_std] = model.standardize(X_raw, y_raw);
    
    // 4. 训练模型（梯度下降）
    cout << "\n2. 开始梯度下降训练..." << endl;
    model.fit(X_std, y_std);
    
    // 5. 评估模型
    auto [mse, r2] = model.evaluate(X_std, y_std);
    VectorXd weights = model.get_weights();
    double bias = model.get_bias();
    
    cout << "\n3. 模型评估结果:" << endl;
    cout << "   权重 w = " << weights(0) << endl;
    cout << "   偏置 b = " << bias << endl;
    cout << "   MSE = " << mse << endl;
    cout << "   R² = " << r2 << endl;
    
    // 6. 对比：解析解（正规方程）
    cout << "\n4. 解析解（正规方程）对比:" << endl;
    auto [weights_closed, bias_closed] = linear_regression_closed_form(X_std, y_std);
    
    // 计算解析解预测
    VectorXd y_pred_closed = X_std * weights_closed + VectorXd::Constant(n_samples, bias_closed);
    double mse_closed = (y_pred_closed - y_std).array().square().mean();
    double ss_total = (y_std.array() - y_std.mean()).square().sum();
    double ss_residual = (y_std.array() - y_pred_closed.array()).square().sum();
    double r2_closed = 1.0 - (ss_residual / ss_total);
    
    cout << "   权重 w = " << weights_closed(0) << endl;
    cout << "   偏置 b = " << bias_closed << endl;
    cout << "   MSE = " << mse_closed << endl;
    cout << "   R² = " << r2_closed << endl;
    
    // 7. 损失曲线数据输出（可用于Python绘图）
    cout << "\n5. 损失历史（前10个）:" << endl;
    vector<double> losses = model.get_loss_history();
    for(int i = 0; i < min(10, (int)losses.size()); ++i) {
        cout << "   迭代 " << i*10 << ": " << losses[i] << endl;
    }
    
    return 0;
}