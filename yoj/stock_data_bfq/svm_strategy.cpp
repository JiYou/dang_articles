/*
 * SVM 机器学习交易策略 (A股只做多)
 *
 * 特征工程:
 *   - EMA偏离 (5/10/20/30/60)
 *   - EMA斜率 (5/10/20)
 *   - 均线排列信号
 *   - MACD (DIF, DEA, MACD柱, MACD方向)
 *   - 布林带 (位置百分比, 带宽)
 *   - 量价 (量比, 换手率偏离)
 *   - RSI(14)
 *   - ATR(14) 相对值
 *   - 动量 (5/10/20日累计涨跌幅)
 *   - K线形态 (实体比例, 上影线, 下影线)
 *   总计 ~28 个特征
 *
 * 标签: 未来10天收盘价最高值 * 0.96 > 当前收盘价 * 1.03 → 1 (涨), 否则 0
 *
 * 模型: libsvm RBF kernel, C/gamma grid search, class_weight balanced
 *
 * 卖出策略 (三重退出):
 *   - 止盈: 持仓收益 >= 5%
 *   - 止损: 持仓收益 <= -3%
 *   - 超时: 持仓超过10天
 *
 * 用法:
 *   单只: ./svm_strategy 600610.csv 0.6
 *   批量: ./svm_strategy --batch 0.6 [--output results.csv] [--threads 8]
 *
 * 编译:
 *   g++ -O3 -std=c++17 -pthread -o svm_strategy svm_strategy.cpp -lsvm
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <libsvm/svm.h>

namespace fs = std::filesystem;

// ============================================================
// Configuration
// ============================================================

// Label generation
static const int    FUTURE_WINDOW   = 10;    // 未来10天窗口
static const double FUTURE_DISCOUNT = 0.96;  // 最高价打96折
static const double PROFIT_THRESH   = 1.03;  // 涨幅阈值3%

// Exit strategy
static const double TAKE_PROFIT = 0.05;   // 止盈 5%
static const double STOP_LOSS   = -0.03;  // 止损 -3%
static const int    MAX_HOLD    = 10;      // 最大持仓天数

// SVM grid search
static const double C_VALUES[]     = {0.01, 0.1, 1.0, 10.0, 100.0};
static const double GAMMA_VALUES[] = {0.0001, 0.001, 0.005, 0.01, 0.05, 0.1};
static const int    N_C     = sizeof(C_VALUES) / sizeof(C_VALUES[0]);
static const int    N_GAMMA = sizeof(GAMMA_VALUES) / sizeof(GAMMA_VALUES[0]);

// Feature count
static const int NUM_FEATURES = 28;

// ============================================================
// Data structures
// ============================================================

struct Bar {
    std::string date;
    double open;
    double close;
    double high;
    double low;
    double volume;
    double amount;     // 成交额
    double amplitude;  // 振幅
    double pct_change; // 涨跌幅
    double turnover;   // 换手率
};

struct TradeRecord {
    int buy_day;
    int sell_day;
    double buy_price;
    double sell_price;
    double return_pct;
    std::string exit_reason;
};

struct StockResult {
    std::string stock_code;
    int data_points;
    std::string date_start;
    std::string date_end;
    double best_C;
    double best_gamma;
    // In-sample metrics
    double is_accuracy;
    double is_precision;
    double is_recall;
    // Out-of-sample backtest
    int oos_trades;
    double oos_win_rate;
    double oos_total_return_pct;
    double oos_avg_return_pct;
    double oos_max_drawdown_pct;
    // OOS prediction metrics
    double oos_accuracy;
    double oos_precision;
    double oos_recall;
};

// ============================================================
// CSV parsing — reads all columns
// ============================================================

bool load_csv(const std::string& path, std::vector<Bar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string line;
    if (!std::getline(file, line)) return false; // skip header

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // Remove BOM if present
        if (line.size() >= 3 &&
            (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB &&
            (unsigned char)line[2] == 0xBF) {
            line = line.substr(3);
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> fields;
        while (std::getline(ss, token, ',')) {
            fields.push_back(token);
        }
        if (fields.size() < 12) continue;

        Bar bar;
        bar.date       = fields[0];
        bar.open       = std::stod(fields[2]);
        bar.close      = std::stod(fields[3]);
        bar.high       = std::stod(fields[4]);
        bar.low        = std::stod(fields[5]);
        bar.volume     = std::stod(fields[6]);
        bar.amount     = std::stod(fields[7]);
        bar.amplitude  = std::stod(fields[8]);
        bar.pct_change = std::stod(fields[9]);
        bar.turnover   = std::stod(fields[11]);
        bars.push_back(bar);
    }
    return !bars.empty();
}

// ============================================================
// Technical Indicator Calculations
// ============================================================

// EMA: exponential moving average
std::vector<double> calc_ema(const std::vector<double>& data, int span) {
    int n = (int)data.size();
    std::vector<double> ema(n, 0.0);
    if (n == 0 || span <= 0) return ema;
    double alpha = 2.0 / (span + 1.0);
    ema[0] = data[0];
    for (int i = 1; i < n; i++) {
        ema[i] = alpha * data[i] + (1.0 - alpha) * ema[i-1];
    }
    return ema;
}

// SMA: simple moving average
std::vector<double> calc_sma(const std::vector<double>& data, int period) {
    int n = (int)data.size();
    std::vector<double> sma(n, 0.0);
    if (n == 0 || period <= 0) return sma;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
        if (i >= period) sum -= data[i - period];
        if (i >= period - 1)
            sma[i] = sum / period;
        else
            sma[i] = sum / (i + 1); // partial average
    }
    return sma;
}

// Standard deviation over period
std::vector<double> calc_std(const std::vector<double>& data, const std::vector<double>& sma, int period) {
    int n = (int)data.size();
    std::vector<double> sd(n, 0.0);
    for (int i = period - 1; i < n; i++) {
        double sum_sq = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            double diff = data[j] - sma[i];
            sum_sq += diff * diff;
        }
        sd[i] = std::sqrt(sum_sq / period);
    }
    return sd;
}

// RSI
std::vector<double> calc_rsi(const std::vector<double>& close, int period = 14) {
    int n = (int)close.size();
    std::vector<double> rsi(n, 50.0); // default neutral
    if (n < period + 1) return rsi;

    std::vector<double> gains(n, 0.0), losses(n, 0.0);
    for (int i = 1; i < n; i++) {
        double diff = close[i] - close[i-1];
        if (diff > 0) gains[i] = diff;
        else losses[i] = -diff;
    }

    double avg_gain = 0, avg_loss = 0;
    for (int i = 1; i <= period; i++) {
        avg_gain += gains[i];
        avg_loss += losses[i];
    }
    avg_gain /= period;
    avg_loss /= period;

    if (avg_loss > 0)
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
    else
        rsi[period] = 100.0;

    for (int i = period + 1; i < n; i++) {
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period;
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period;
        if (avg_loss > 0)
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss);
        else
            rsi[i] = 100.0;
    }
    return rsi;
}

// ATR (Average True Range)
std::vector<double> calc_atr(const std::vector<Bar>& bars, int period = 14) {
    int n = (int)bars.size();
    std::vector<double> tr(n, 0.0);
    std::vector<double> atr(n, 0.0);
    if (n < 2) return atr;

    tr[0] = bars[0].high - bars[0].low;
    for (int i = 1; i < n; i++) {
        double hl = bars[i].high - bars[i].low;
        double hc = std::abs(bars[i].high - bars[i-1].close);
        double lc = std::abs(bars[i].low - bars[i-1].close);
        tr[i] = std::max({hl, hc, lc});
    }

    // EMA-style ATR
    atr[0] = tr[0];
    double alpha = 2.0 / (period + 1.0);
    for (int i = 1; i < n; i++) {
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i-1];
    }
    return atr;
}

// ============================================================
// Feature extraction — 28 features per bar
// ============================================================

// Returns feature matrix [n_samples][NUM_FEATURES]
// valid_start: first index with enough data for all indicators
void extract_features(const std::vector<Bar>& bars,
                      std::vector<std::vector<double>>& features,
                      int& valid_start) {
    int n = (int)bars.size();
    features.resize(n, std::vector<double>(NUM_FEATURES, 0.0));

    // Extract price/volume arrays
    std::vector<double> close(n), open(n), high(n), low(n), volume(n), turnover(n);
    for (int i = 0; i < n; i++) {
        close[i]    = bars[i].close;
        open[i]     = bars[i].open;
        high[i]     = bars[i].high;
        low[i]      = bars[i].low;
        volume[i]   = bars[i].volume;
        turnover[i] = bars[i].turnover;
    }

    // EMAs
    auto ema5  = calc_ema(close, 5);
    auto ema10 = calc_ema(close, 10);
    auto ema20 = calc_ema(close, 20);
    auto ema30 = calc_ema(close, 30);
    auto ema60 = calc_ema(close, 60);

    // MACD
    auto ema12 = calc_ema(close, 12);
    auto ema26 = calc_ema(close, 26);
    std::vector<double> dif(n);
    for (int i = 0; i < n; i++) dif[i] = ema12[i] - ema26[i];
    auto dea = calc_ema(dif, 9);
    std::vector<double> macd_hist(n);
    for (int i = 0; i < n; i++) macd_hist[i] = 2.0 * (dif[i] - dea[i]);

    // Bollinger Bands (20-period SMA, 2σ)
    auto sma20 = calc_sma(close, 20);
    auto std20 = calc_std(close, sma20, 20);

    // Volume MA
    auto vol_ma5  = calc_sma(volume, 5);
    auto turn_ma10 = calc_sma(turnover, 10);

    // RSI
    auto rsi = calc_rsi(close, 14);

    // ATR
    auto atr = calc_atr(bars, 14);

    // Need at least 60 bars for EMA60 to be meaningful
    valid_start = 60;

    for (int i = 0; i < n; i++) {
        int f = 0;

        // === Feature 0-4: EMA deviation (close - EMA) / EMA ===
        features[i][f++] = (ema5[i]  > 0) ? (close[i] - ema5[i])  / ema5[i]  : 0; // 0
        features[i][f++] = (ema10[i] > 0) ? (close[i] - ema10[i]) / ema10[i] : 0; // 1
        features[i][f++] = (ema20[i] > 0) ? (close[i] - ema20[i]) / ema20[i] : 0; // 2
        features[i][f++] = (ema30[i] > 0) ? (close[i] - ema30[i]) / ema30[i] : 0; // 3
        features[i][f++] = (ema60[i] > 0) ? (close[i] - ema60[i]) / ema60[i] : 0; // 4

        // === Feature 5-7: EMA slope (N-day change rate) ===
        if (i >= 3 && ema5[i-3] > 0)
            features[i][f] = (ema5[i] - ema5[i-3]) / ema5[i-3];
        f++; // 5

        if (i >= 5 && ema10[i-5] > 0)
            features[i][f] = (ema10[i] - ema10[i-5]) / ema10[i-5];
        f++; // 6

        if (i >= 10 && ema20[i-10] > 0)
            features[i][f] = (ema20[i] - ema20[i-10]) / ema20[i-10];
        f++; // 7

        // === Feature 8: MA alignment signal ===
        // +1 if short > mid > long, -1 if reversed, 0 otherwise
        {
            int score = 0;
            if (ema5[i] > ema10[i]) score++; else score--;
            if (ema10[i] > ema20[i]) score++; else score--;
            if (ema20[i] > ema60[i]) score++; else score--;
            features[i][f] = score / 3.0;
        }
        f++; // 8

        // === Feature 9-12: MACD ===
        features[i][f++] = (close[i] > 0) ? dif[i] / close[i] : 0;       // 9: DIF normalized
        features[i][f++] = (close[i] > 0) ? dea[i] / close[i] : 0;       // 10: DEA normalized
        features[i][f++] = (close[i] > 0) ? macd_hist[i] / close[i] : 0; // 11: MACD histogram normalized
        // MACD direction: histogram increasing or decreasing
        if (i >= 1)
            features[i][f] = (macd_hist[i] > macd_hist[i-1]) ? 1.0 : -1.0;
        f++; // 12

        // === Feature 13-14: Bollinger Bands ===
        {
            double bw = std20[i] * 4.0; // upper - lower = 4σ
            if (bw > 1e-9)
                features[i][f] = (close[i] - sma20[i]) / (bw / 2.0); // position: -1 to +1
            else
                features[i][f] = 0;
        }
        f++; // 13: BB position

        if (sma20[i] > 0)
            features[i][f] = (std20[i] * 4.0) / sma20[i]; // bandwidth
        f++; // 14

        // === Feature 15-16: Volume/Turnover ===
        if (vol_ma5[i] > 0)
            features[i][f] = volume[i] / vol_ma5[i]; // volume ratio
        else
            features[i][f] = 1.0;
        f++; // 15

        if (turn_ma10[i] > 0)
            features[i][f] = turnover[i] / turn_ma10[i]; // turnover deviation
        else
            features[i][f] = 1.0;
        f++; // 16

        // === Feature 17: RSI (normalized to [-1, 1]) ===
        features[i][f++] = (rsi[i] - 50.0) / 50.0; // 17

        // === Feature 18: ATR relative (ATR / close) ===
        if (close[i] > 0)
            features[i][f] = atr[i] / close[i];
        f++; // 18

        // === Feature 19-21: Momentum (cumulative return over N days) ===
        if (i >= 5 && close[i-5] > 0)
            features[i][f] = (close[i] - close[i-5]) / close[i-5];
        f++; // 19: 5-day momentum

        if (i >= 10 && close[i-10] > 0)
            features[i][f] = (close[i] - close[i-10]) / close[i-10];
        f++; // 20: 10-day momentum

        if (i >= 20 && close[i-20] > 0)
            features[i][f] = (close[i] - close[i-20]) / close[i-20];
        f++; // 21: 20-day momentum

        // === Feature 22: Average amplitude (recent 5 days) ===
        {
            double sum_amp = 0;
            int cnt = 0;
            for (int j = std::max(0, i-4); j <= i; j++) {
                sum_amp += bars[j].amplitude;
                cnt++;
            }
            features[i][f] = sum_amp / cnt / 100.0; // normalize from percentage
        }
        f++; // 22

        // === Feature 23-25: Candlestick patterns ===
        {
            double body = std::abs(close[i] - open[i]);
            double range = high[i] - low[i];
            if (range > 1e-9) {
                features[i][f++] = body / range;  // 23: body ratio
                features[i][f++] = (high[i] - std::max(open[i], close[i])) / range; // 24: upper shadow
                features[i][f++] = (std::min(open[i], close[i]) - low[i]) / range;  // 25: lower shadow
            } else {
                f += 3; // 23-25: all zero (no range)
            }
        }

        // === Feature 26: Price position in recent 20-day range ===
        if (i >= 19) {
            double max_h = *std::max_element(high.begin() + i - 19, high.begin() + i + 1);
            double min_l = *std::min_element(low.begin() + i - 19, low.begin() + i + 1);
            if (max_h - min_l > 1e-9)
                features[i][f] = (close[i] - min_l) / (max_h - min_l);
        }
        f++; // 26

        // === Feature 27: Volume trend (5-day volume slope) ===
        if (i >= 4 && vol_ma5[i] > 0) {
            // linear regression slope of volume over last 5 days, normalized
            double mean_x = 2.0, mean_y = 0;
            for (int j = 0; j < 5; j++) mean_y += volume[i - 4 + j];
            mean_y /= 5.0;
            double num = 0, den = 0;
            for (int j = 0; j < 5; j++) {
                double xd = j - mean_x;
                double yd = volume[i - 4 + j] - mean_y;
                num += xd * yd;
                den += xd * xd;
            }
            if (den > 0 && mean_y > 0)
                features[i][f] = (num / den) / mean_y;
        }
        f++; // 27
    }
}

// ============================================================
// Label generation
// ============================================================

// label[i] = 1 if max(close[i+1..i+10]) * 0.96 > close[i] * 1.03
void generate_labels(const std::vector<Bar>& bars, std::vector<int>& labels, int& label_end) {
    int n = (int)bars.size();
    labels.resize(n, 0);
    label_end = n - FUTURE_WINDOW; // last valid label index (exclusive)

    for (int i = 0; i < label_end; i++) {
        double max_close = 0;
        for (int j = 1; j <= FUTURE_WINDOW; j++) {
            max_close = std::max(max_close, bars[i + j].close);
        }
        if (max_close * FUTURE_DISCOUNT > bars[i].close * PROFIT_THRESH) {
            labels[i] = 1;
        }
    }
}

// ============================================================
// Feature scaling (StandardScaler)
// ============================================================

struct Scaler {
    std::vector<double> mean;
    std::vector<double> stddev;

    void fit(const std::vector<std::vector<double>>& X, int start, int end) {
        int nf = NUM_FEATURES;
        mean.resize(nf, 0.0);
        stddev.resize(nf, 0.0);
        int count = end - start;
        if (count <= 0) return;

        for (int i = start; i < end; i++) {
            for (int f = 0; f < nf; f++) {
                mean[f] += X[i][f];
            }
        }
        for (int f = 0; f < nf; f++) mean[f] /= count;

        for (int i = start; i < end; i++) {
            for (int f = 0; f < nf; f++) {
                double d = X[i][f] - mean[f];
                stddev[f] += d * d;
            }
        }
        for (int f = 0; f < nf; f++) {
            stddev[f] = std::sqrt(stddev[f] / count);
            if (stddev[f] < 1e-12) stddev[f] = 1.0; // avoid div by zero
        }
    }

    void transform(std::vector<std::vector<double>>& X, int start, int end) const {
        for (int i = start; i < end; i++) {
            for (int f = 0; f < NUM_FEATURES; f++) {
                X[i][f] = (X[i][f] - mean[f]) / stddev[f];
            }
        }
    }
};

// ============================================================
// SVM helpers
// ============================================================

// Suppress libsvm output
static void svm_print_null(const char*) {}

// Build svm_problem from feature matrix and labels
svm_problem build_problem(const std::vector<std::vector<double>>& X,
                          const std::vector<int>& y,
                          int start, int end,
                          std::vector<svm_node>& nodes_storage,
                          std::vector<svm_node*>& row_ptrs) {
    int count = end - start;
    // Each row has NUM_FEATURES + 1 (sentinel) nodes
    nodes_storage.resize(count * (NUM_FEATURES + 1));
    row_ptrs.resize(count);

    for (int i = 0; i < count; i++) {
        int idx = start + i;
        svm_node* row = &nodes_storage[i * (NUM_FEATURES + 1)];
        for (int f = 0; f < NUM_FEATURES; f++) {
            row[f].index = f + 1; // 1-indexed
            row[f].value = X[idx][f];
        }
        row[NUM_FEATURES].index = -1; // sentinel
        row[NUM_FEATURES].value = 0;
        row_ptrs[i] = row;
    }

    svm_problem prob;
    prob.l = count;
    prob.y = new double[count];
    for (int i = 0; i < count; i++) {
        prob.y[i] = (double)y[start + i];
    }
    prob.x = row_ptrs.data();
    return prob;
}

void free_problem(svm_problem& prob) {
    delete[] prob.y;
    prob.y = nullptr;
}

// Predict single sample
double predict_one(const svm_model* model, const std::vector<double>& x) {
    std::vector<svm_node> nodes(NUM_FEATURES + 1);
    for (int f = 0; f < NUM_FEATURES; f++) {
        nodes[f].index = f + 1;
        nodes[f].value = x[f];
    }
    nodes[NUM_FEATURES].index = -1;
    nodes[NUM_FEATURES].value = 0;
    return svm_predict(model, nodes.data());
}

// ============================================================
// SVM training with grid search
// ============================================================

svm_model* train_svm(const std::vector<std::vector<double>>& X,
                     const std::vector<int>& y,
                     int train_start, int train_end,
                     double& best_C, double& best_gamma,
                     double& best_accuracy,
                     std::vector<svm_node>& nodes_storage,
                     std::vector<svm_node*>& row_ptrs) {
    svm_set_print_string_function(svm_print_null);

    // Count class distribution for balanced weights
    int n_pos = 0, n_neg = 0;
    for (int i = train_start; i < train_end; i++) {
        if (y[i] == 1) n_pos++;
        else n_neg++;
    }

    best_accuracy = -1;
    best_C = 1.0;
    best_gamma = 0.1;

    // Build problem — nodes_storage and row_ptrs owned by caller
    svm_problem prob = build_problem(X, y, train_start, train_end, nodes_storage, row_ptrs);

    int n_train = train_end - train_start;
    std::vector<double> cv_targets(n_train);

    // Class weights (kept alive for entire function)
    int weight_labels[2] = {0, 1};
    double weights[2];
    double total_samples = n_pos + n_neg;
    if (n_pos > 0 && n_neg > 0) {
        weights[0] = total_samples / (2.0 * n_neg);
        weights[1] = total_samples / (2.0 * n_pos);
    } else {
        weights[0] = 1.0;
        weights[1] = 1.0;
    }

    for (int ci = 0; ci < N_C; ci++) {
        for (int gi = 0; gi < N_GAMMA; gi++) {
            svm_parameter param;
            memset(&param, 0, sizeof(param));
            param.svm_type = C_SVC;
            param.kernel_type = RBF;
            param.C = C_VALUES[ci];
            param.gamma = GAMMA_VALUES[gi];
            param.cache_size = 200;
            param.eps = 0.001;
            param.shrinking = 1;
            param.probability = 0;
            param.nr_weight = 2;
            param.weight_label = weight_labels;
            param.weight = weights;

            const char* err = svm_check_parameter(&prob, &param);
            if (err) continue;

            // 5-fold CV
            svm_cross_validation(&prob, &param, 5, cv_targets.data());

            // Calculate F1 score (prefers balanced precision/recall over pure accuracy)
            int cv_tp = 0, cv_fp = 0, cv_fn = 0;
            for (int i = 0; i < n_train; i++) {
                int pred = (int)cv_targets[i];
                int actual = (int)prob.y[i];
                if (pred == 1 && actual == 1) cv_tp++;
                else if (pred == 1 && actual == 0) cv_fp++;
                else if (pred == 0 && actual == 1) cv_fn++;
            }
            double cv_prec = (cv_tp + cv_fp > 0) ? (double)cv_tp / (cv_tp + cv_fp) : 0;
            double cv_rec  = (cv_tp + cv_fn > 0) ? (double)cv_tp / (cv_tp + cv_fn) : 0;
            double f1 = (cv_prec + cv_rec > 0) ? 2 * cv_prec * cv_rec / (cv_prec + cv_rec) : 0;

            if (f1 > best_accuracy) {  // best_accuracy stores best F1
                best_accuracy = f1;
                best_C = C_VALUES[ci];
                best_gamma = GAMMA_VALUES[gi];
            }
        }
    }

    // Train final model with best params
    svm_parameter param;
    memset(&param, 0, sizeof(param));
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.C = best_C;
    param.gamma = best_gamma;
    param.cache_size = 200;
    param.eps = 0.001;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 2;
    param.weight_label = weight_labels;
    param.weight = weights;

    svm_model* model = svm_train(&prob, &param);
    // NOTE: model's SVs point into nodes_storage (owned by caller)
    // prob.y is separately allocated — transfer ownership to caller too
    // We do NOT free prob or nodes here; caller keeps them alive
    // until model is freed

    // We only need to free prob.y eventually — but since it's just labels
    // and caller's nodes_storage keeps SVs alive, we can free it now
    // because model doesn't reference prob.y (only prob.x)
    free_problem(prob);

    return model;
}

// ============================================================
// Backtesting with triple exit
// ============================================================

void backtest(const std::vector<Bar>& bars,
              const std::vector<std::vector<double>>& features,
              const std::vector<int>& labels,
              const svm_model* model,
              int test_start, int test_end,
              std::vector<TradeRecord>& trades,
              double& total_return_pct,
              double& max_drawdown_pct,
              double& win_rate,
              // prediction metrics
              double& accuracy, double& precision, double& recall) {
    trades.clear();
    total_return_pct = 0;
    max_drawdown_pct = 0;
    win_rate = 0;

    // Prediction metrics
    int tp = 0, fp = 0, tn = 0, fn = 0;

    double equity = 1.0;
    double peak_equity = 1.0;
    double max_dd = 0;

    bool in_position = false;
    double buy_price = 0;
    int hold_days = 0;
    int buy_day = 0;

    for (int i = test_start; i < test_end; i++) {
        // Calculate prediction metrics (only where labels are valid)
        if (i < (int)labels.size() - FUTURE_WINDOW) {
            int pred = (int)predict_one(model, features[i]);
            int actual = labels[i];
            if (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 0) tn++;
            else fn++;
        }

        if (!in_position) {
            // Check buy signal: SVM predicts 1
            int pred = (int)predict_one(model, features[i]);
            if (pred == 1 && i + 1 < test_end) {
                // Buy at next day's open
                in_position = true;
                buy_price = bars[i + 1].open;
                buy_day = i + 1;
                hold_days = 0;
            }
        } else {
            hold_days++;
            double current_return = (bars[i].close - buy_price) / buy_price;

            bool should_sell = false;
            std::string reason;

            // Check triple exit
            if (current_return >= TAKE_PROFIT) {
                should_sell = true;
                reason = "take_profit";
            } else if (current_return <= STOP_LOSS) {
                should_sell = true;
                reason = "stop_loss";
            } else if (hold_days >= MAX_HOLD) {
                should_sell = true;
                reason = "timeout";
            }

            if (should_sell) {
                // Sell at close (or next open, but we use close for simplicity)
                double sell_price = bars[i].close;
                double trade_return = (sell_price - buy_price) / buy_price;

                TradeRecord tr;
                tr.buy_day = buy_day;
                tr.sell_day = i;
                tr.buy_price = buy_price;
                tr.sell_price = sell_price;
                tr.return_pct = trade_return * 100.0;
                tr.exit_reason = reason;
                trades.push_back(tr);

                equity *= (1.0 + trade_return);
                peak_equity = std::max(peak_equity, equity);
                double dd = (peak_equity - equity) / peak_equity;
                max_dd = std::max(max_dd, dd);

                in_position = false;
            }
        }
    }

    // Close any remaining position
    if (in_position && test_end > 0) {
        double sell_price = bars[test_end - 1].close;
        double trade_return = (sell_price - buy_price) / buy_price;
        TradeRecord tr;
        tr.buy_day = buy_day;
        tr.sell_day = test_end - 1;
        tr.buy_price = buy_price;
        tr.sell_price = sell_price;
        tr.return_pct = trade_return * 100.0;
        tr.exit_reason = "end_of_data";
        trades.push_back(tr);

        equity *= (1.0 + trade_return);
        peak_equity = std::max(peak_equity, equity);
        double dd = (peak_equity - equity) / peak_equity;
        max_dd = std::max(max_dd, dd);
    }

    total_return_pct = (equity - 1.0) * 100.0;
    max_drawdown_pct = max_dd * 100.0;

    int wins = 0;
    for (auto& t : trades) {
        if (t.return_pct > 0) wins++;
    }
    win_rate = trades.empty() ? 0 : (double)wins / trades.size() * 100.0;

    // Prediction metrics
    int total_pred = tp + fp + tn + fn;
    accuracy = total_pred > 0 ? (double)(tp + tn) / total_pred * 100.0 : 0;
    precision = (tp + fp) > 0 ? (double)tp / (tp + fp) * 100.0 : 0;
    recall = (tp + fn) > 0 ? (double)tp / (tp + fn) * 100.0 : 0;
}

// ============================================================
// Process single stock
// ============================================================

bool process_stock(const std::string& csv_path, double split_ratio,
                   StockResult& result, bool verbose = false) {
    // Extract stock code from filename
    std::string filename = fs::path(csv_path).stem().string();
    result.stock_code = filename;

    // Load data
    std::vector<Bar> bars;
    if (!load_csv(csv_path, bars)) {
        if (verbose) std::cerr << "Failed to load: " << csv_path << "\n";
        return false;
    }

    int n = (int)bars.size();
    if (n < 200) { // Need sufficient data
        if (verbose) std::cerr << "Insufficient data (" << n << " bars): " << csv_path << "\n";
        return false;
    }

    result.data_points = n;
    result.date_start = bars.front().date;
    result.date_end = bars.back().date;

    // Extract features
    std::vector<std::vector<double>> features;
    int valid_start;
    extract_features(bars, features, valid_start);

    // Generate labels
    std::vector<int> labels;
    int label_end;
    generate_labels(bars, labels, label_end);

    // Determine train/test split
    int usable_start = valid_start; // first index with valid features
    int usable_end = label_end;     // last index with valid labels

    if (usable_end - usable_start < 100) {
        if (verbose) std::cerr << "Insufficient usable data: " << csv_path << "\n";
        return false;
    }

    int train_end = usable_start + (int)((usable_end - usable_start) * split_ratio);
    int test_start = train_end;
    int test_end_bt = n; // backtest can go to end of data (no labels needed for trading)

    if (train_end - usable_start < 50 || test_end_bt - test_start < 30) {
        if (verbose) std::cerr << "Insufficient train/test split: " << csv_path << "\n";
        return false;
    }

    // Scale features
    Scaler scaler;
    scaler.fit(features, usable_start, train_end);
    scaler.transform(features, usable_start, test_end_bt);

    // Train SVM (model is serialized/deserialized internally, safe to use after)
    double best_C, best_gamma, is_accuracy;
    std::vector<svm_node> train_nodes;
    std::vector<svm_node*> train_ptrs;
    svm_model* model = train_svm(features, labels, usable_start, train_end,
                                 best_C, best_gamma, is_accuracy,
                                 train_nodes, train_ptrs);
    if (!model) {
        if (verbose) std::cerr << "SVM training failed: " << csv_path << "\n";
        return false;
    }

    result.best_C = best_C;
    result.best_gamma = best_gamma;

    // In-sample metrics (prediction on training data)
    {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (int i = usable_start; i < train_end; i++) {
            int pred = (int)predict_one(model, features[i]);
            int actual = labels[i];
            if (pred == 1 && actual == 1) tp++;
            else if (pred == 1 && actual == 0) fp++;
            else if (pred == 0 && actual == 0) tn++;
            else fn++;
        }
        int total = tp + fp + tn + fn;
        result.is_accuracy = total > 0 ? (double)(tp + tn) / total * 100.0 : 0;
        result.is_precision = (tp + fp) > 0 ? (double)tp / (tp + fp) * 100.0 : 0;
        result.is_recall = (tp + fn) > 0 ? (double)tp / (tp + fn) * 100.0 : 0;
    }

    // Out-of-sample backtest
    std::vector<TradeRecord> trades;
    backtest(bars, features, labels, model, test_start, test_end_bt, trades,
             result.oos_total_return_pct, result.oos_max_drawdown_pct, result.oos_win_rate,
             result.oos_accuracy, result.oos_precision, result.oos_recall);
    result.oos_trades = (int)trades.size();
    result.oos_avg_return_pct = trades.empty() ? 0 :
        result.oos_total_return_pct / trades.size();

    if (verbose) {
        printf("  Stock: %s | Data: %d bars [%s ~ %s]\n",
               result.stock_code.c_str(), n,
               result.date_start.c_str(), result.date_end.c_str());
        printf("  SVM: C=%.2f gamma=%.3f | IS acc=%.1f%% prec=%.1f%% rec=%.1f%%\n",
               best_C, best_gamma,
               result.is_accuracy, result.is_precision, result.is_recall);
        printf("  OOS: %d trades | win=%.1f%% | return=%.2f%% | avg=%.2f%% | maxDD=%.2f%%\n",
               result.oos_trades, result.oos_win_rate,
               result.oos_total_return_pct, result.oos_avg_return_pct,
               result.oos_max_drawdown_pct);
        printf("  OOS pred: acc=%.1f%% prec=%.1f%% rec=%.1f%%\n",
               result.oos_accuracy, result.oos_precision, result.oos_recall);

        // Print trade details
        if (!trades.empty()) {
            printf("  --- Trade details ---\n");
            for (int t = 0; t < std::min((int)trades.size(), 20); t++) {
                auto& tr = trades[t];
                printf("    #%d: buy@%.2f[%s] sell@%.2f[%s] ret=%.2f%% (%s)\n",
                       t+1, tr.buy_price, bars[tr.buy_day].date.c_str(),
                       tr.sell_price, bars[tr.sell_day].date.c_str(),
                       tr.return_pct, tr.exit_reason.c_str());
            }
            if ((int)trades.size() > 20)
                printf("    ... and %d more trades\n", (int)trades.size() - 20);
        }
    }

    svm_free_and_destroy_model(&model);
    return true;
}

// ============================================================
// Batch processing
// ============================================================

void batch_process(double split_ratio, const std::string& output_path, int num_threads) {
    // Find all CSV files
    std::vector<std::string> csv_files;
    for (auto& entry : fs::directory_iterator(".")) {
        if (entry.is_regular_file()) {
            std::string fname = entry.path().filename().string();
            if (fname.size() > 4 &&
                fname.substr(fname.size()-4) == ".csv" &&
                std::isdigit(fname[0])) {
                csv_files.push_back(fname);
            }
        }
    }
    std::sort(csv_files.begin(), csv_files.end());

    int total = (int)csv_files.size();
    printf("Found %d stock CSV files. Processing with %d threads...\n", total, num_threads);

    std::vector<StockResult> results(total);
    std::vector<bool> success(total, false);
    std::atomic<int> completed{0};
    std::atomic<int> failed{0};
    std::mutex print_mtx;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Worker function
    auto worker = [&](int thread_id) {
        for (int i = thread_id; i < total; i += num_threads) {
            bool ok = process_stock(csv_files[i], split_ratio, results[i], false);
            success[i] = ok;
            if (!ok) failed++;

            int done = ++completed;
            if (done % 100 == 0 || done == total) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start_time).count();
                double eta = (elapsed / done) * (total - done);
                std::lock_guard<std::mutex> lock(print_mtx);
                printf("\r  Progress: %d/%d (%.1f%%) | elapsed %.1fs | ETA %.1fs    ",
                       done, total, 100.0 * done / total, elapsed, eta);
                fflush(stdout);
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads) t.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    printf("\n\nCompleted %d stocks in %.1f seconds (%.3f s/stock avg)\n",
           total - (int)failed, total_time, total_time / (total - (int)failed));
    printf("Failed: %d stocks\n\n", (int)failed);

    // Write results
    std::ofstream out(output_path);
    out << "stock_code,data_points,date_start,date_end,"
        << "best_C,best_gamma,"
        << "is_accuracy,is_precision,is_recall,"
        << "oos_trades,oos_win_rate,oos_total_return_pct,oos_avg_return_pct,oos_max_drawdown_pct,"
        << "oos_accuracy,oos_precision,oos_recall\n";

    int cnt_oos_pos = 0, cnt_oos_neg = 0, cnt_no_trades = 0;
    double sum_oos_ret = 0;
    double sum_win_rate = 0;
    int cnt_with_trades = 0;

    for (int i = 0; i < total; i++) {
        if (!success[i]) continue;
        auto& r = results[i];
        out << r.stock_code << ","
            << r.data_points << ","
            << r.date_start << ","
            << r.date_end << ","
            << std::fixed << std::setprecision(4)
            << r.best_C << ","
            << r.best_gamma << ","
            << std::setprecision(2)
            << r.is_accuracy << ","
            << r.is_precision << ","
            << r.is_recall << ","
            << r.oos_trades << ","
            << r.oos_win_rate << ","
            << r.oos_total_return_pct << ","
            << r.oos_avg_return_pct << ","
            << r.oos_max_drawdown_pct << ","
            << r.oos_accuracy << ","
            << r.oos_precision << ","
            << r.oos_recall << "\n";

        if (r.oos_trades == 0) {
            cnt_no_trades++;
        } else {
            cnt_with_trades++;
            sum_oos_ret += r.oos_total_return_pct;
            sum_win_rate += r.oos_win_rate;
            if (r.oos_total_return_pct > 0) cnt_oos_pos++;
            else cnt_oos_neg++;
        }
    }
    out.close();

    printf("=== Batch Results Summary ===\n");
    printf("Stocks processed: %d (failed: %d)\n", total - (int)failed, (int)failed);
    printf("Stocks with trades: %d | No trades: %d\n", cnt_with_trades, cnt_no_trades);
    printf("OOS positive return: %d (%.1f%%)\n", cnt_oos_pos,
           cnt_with_trades > 0 ? 100.0 * cnt_oos_pos / cnt_with_trades : 0);
    printf("OOS negative return: %d (%.1f%%)\n", cnt_oos_neg,
           cnt_with_trades > 0 ? 100.0 * cnt_oos_neg / cnt_with_trades : 0);
    printf("Average OOS return: %.2f%%\n",
           cnt_with_trades > 0 ? sum_oos_ret / cnt_with_trades : 0);
    printf("Average OOS win rate: %.2f%%\n",
           cnt_with_trades > 0 ? sum_win_rate / cnt_with_trades : 0);
    printf("Results saved to: %s\n", output_path.c_str());
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage:\n");
        printf("  Single:  %s <stock.csv> <split_ratio>\n", argv[0]);
        printf("  Batch:   %s --batch <split_ratio> [--output results.csv] [--threads N]\n", argv[0]);
        printf("\nExamples:\n");
        printf("  %s 600610.csv 0.6\n", argv[0]);
        printf("  %s --batch 0.6 --threads 16\n", argv[0]);
        return 1;
    }

    std::string arg1 = argv[1];

    if (arg1 == "--batch") {
        double split_ratio = std::stod(argv[2]);
        std::string output = "svm_batch_results.csv";
        int threads = 8;

        for (int i = 3; i < argc; i++) {
            std::string a = argv[i];
            if (a == "--output" && i + 1 < argc) output = argv[++i];
            else if (a == "--threads" && i + 1 < argc) threads = std::stoi(argv[++i]);
        }

        batch_process(split_ratio, output, threads);
    } else {
        double split_ratio = std::stod(argv[2]);
        StockResult result;
        if (process_stock(arg1, split_ratio, result, true)) {
            printf("\n=== Summary ===\n");
            printf("Return: %.2f%% | Win rate: %.1f%% | Trades: %d | Max DD: %.2f%%\n",
                   result.oos_total_return_pct, result.oos_win_rate,
                   result.oos_trades, result.oos_max_drawdown_pct);
        } else {
            fprintf(stderr, "Failed to process stock.\n");
            return 1;
        }
    }

    return 0;
}
