/*
 * EMA四均线交叉策略回测 + 暴力网格搜索 (A股只做多)
 *
 * 完全对标 dual_ma_optuna_ema_stock.py 的逻辑：
 *   - 4条EMA: span = length0, length0+p1, length0+p1+p2, length0+p1+p2+p3
 *   - 任意两条金叉→买入, 任意两条死叉→卖出
 *   - 信号shift(1): 当日收盘产生信号 → 次日开盘执行
 *   - 百分比收益率 (适合不复权数据)
 *   - 只做多，不做空
 *
 * 参数搜索范围:
 *   length0 ∈ [3, 15]   (13种)
 *   p1      ∈ [2, 15]   (14种)
 *   p2      ∈ [2, 15]   (14种)
 *   p3      ∈ [2, 15]   (14种)
 *   总计: 13 × 14 × 14 × 14 = 35,672 种组合
 *
 * 用法:
 *   单只: ./ema_strategy 600610.csv 0.6
 *   批量: ./ema_strategy --batch 0.6 [--output results.csv] [--threads 8]
 *
 * 编译:
 *   g++ -O3 -std=c++17 -pthread -o ema_strategy ema_strategy.cpp
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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// Data structures
// ============================================================

struct Bar {
    std::string date;
    double open;
    double close;
};

struct OptResult {
    std::string stock_code;
    int data_points;
    std::string date_start;
    std::string date_end;
    int length0, p1, p2, p3;
    double in_sample_return_pct;
    double out_of_sample_return_pct;
    double full_return_pct;
};

struct BestParams {
    int length0, p1, p2, p3;
    double best_return;
};

// ============================================================
// CSV parsing — only reads date, open, close columns
// ============================================================

bool load_csv(const std::string& path, std::vector<Bar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string line;
    if (!std::getline(file, line)) return false;

    // Parse header to find column indices
    int col_date = -1, col_open = -1, col_close = -1;
    {
        std::istringstream hss(line);
        std::string token;
        int idx = 0;
        while (std::getline(hss, token, ',')) {
            if (token == "date") col_date = idx;
            else if (token == "open") col_open = idx;
            else if (token == "close") col_close = idx;
            idx++;
        }
    }
    if (col_date < 0 || col_open < 0 || col_close < 0) return false;

    int max_col = std::max({col_date, col_open, col_close});

    bars.clear();
    bars.reserve(4096);

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<std::string> fields;
        fields.reserve(max_col + 2);
        std::istringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        if ((int)fields.size() <= max_col) continue;

        Bar bar;
        bar.date = fields[col_date];
        try {
            bar.open = std::stod(fields[col_open]);
            bar.close = std::stod(fields[col_close]);
        } catch (...) {
            continue;
        }
        if (bar.open <= 0.0 || bar.close <= 0.0) continue;
        bars.push_back(std::move(bar));
    }
    return !bars.empty();
}

// ============================================================
// EMA calculation
// ============================================================

void compute_ema(const double* close, int n, int span, double* out) {
    if (n == 0) return;
    double multiplier = 2.0 / (span + 1.0);
    out[0] = close[0];
    for (int i = 1; i < n; i++) {
        out[i] = (close[i] - out[i - 1]) * multiplier + out[i - 1];
    }
}

// ============================================================
// Strategy backtest (long-only, percentage returns)
// Uses char* instead of bool* to avoid vector<bool> issues
// ============================================================

double backtest(const double* close, const double* openp, int n,
                int length0, int p1, int p2, int p3,
                double* ema1, double* ema2, double* ema3, double* ema4,
                char* buy_sig, char* sell_sig) {
    if (n < 2) return 0.0;

    compute_ema(close, n, length0, ema1);
    compute_ema(close, n, length0 + p1, ema2);
    compute_ema(close, n, length0 + p1 + p2, ema3);
    compute_ema(close, n, length0 + p1 + p2 + p3, ema4);

    std::memset(buy_sig, 0, n);
    std::memset(sell_sig, 0, n);

    const double* emas[4] = {ema1, ema2, ema3, ema4};

    // Signal at bar i → execute at bar i+1 (shift(1) = future-function correction)
    for (int i = 1; i < n - 1; i++) {
        char buy_raw = 0, sell_raw = 0;
        for (int a = 0; a < 4 && (!buy_raw || !sell_raw); a++) {
            for (int b = a + 1; b < 4 && (!buy_raw || !sell_raw); b++) {
                if (!buy_raw && emas[a][i - 1] < emas[b][i - 1] && emas[a][i] >= emas[b][i])
                    buy_raw = 1;
                if (!sell_raw && emas[a][i - 1] > emas[b][i - 1] && emas[a][i] <= emas[b][i])
                    sell_raw = 1;
            }
        }
        buy_sig[i + 1] = buy_raw;
        sell_sig[i + 1] = sell_raw;
    }

    int position = 0;
    double entry_price = 0.0;
    double cum_pct = 0.0;

    for (int i = 1; i < n; i++) {
        double daily_pnl = 0.0;
        if (position == 1) {
            daily_pnl = (close[i] - close[i - 1]) / close[i - 1] * 100.0;
        }
        if (buy_sig[i] && position == 0) {
            entry_price = openp[i];
            position = 1;
            daily_pnl = (close[i] - openp[i]) / openp[i] * 100.0;
        } else if (sell_sig[i] && position == 1) {
            daily_pnl = (openp[i] - close[i - 1]) / close[i - 1] * 100.0;
            position = 0;
        }
        cum_pct += daily_pnl;
    }

    // Forced close at end (replicate Python logic)
    if (position == 1) {
        cum_pct += (close[n - 1] - entry_price) / entry_price * 100.0;
    }

    return cum_pct;
}

// ============================================================
// Grid search: find best params on training set
// ============================================================

BestParams grid_search(const double* close, const double* openp, int n) {
    BestParams best{3, 2, 2, 2, -1e18};

    std::vector<double> ema1(n), ema2(n), ema3(n), ema4(n);
    std::vector<char> buy_sig(n), sell_sig(n);

    for (int l0 = 3; l0 <= 15; l0++) {
        for (int pp1 = 2; pp1 <= 15; pp1++) {
            for (int pp2 = 2; pp2 <= 15; pp2++) {
                for (int pp3 = 2; pp3 <= 15; pp3++) {
                    double ret = backtest(
                        close, openp, n, l0, pp1, pp2, pp3,
                        ema1.data(), ema2.data(), ema3.data(), ema4.data(),
                        buy_sig.data(), sell_sig.data());
                    if (ret > best.best_return) {
                        best = {l0, pp1, pp2, pp3, ret};
                    }
                }
            }
        }
    }
    return best;
}

// ============================================================
// Process one stock
// ============================================================

bool process_stock(const std::string& filepath, double split_ratio, OptResult& result) {
    std::vector<Bar> bars;
    if (!load_csv(filepath, bars)) return false;

    int n = (int)bars.size();
    if (n < 60) return false;

    std::vector<double> close(n), openp(n);
    for (int i = 0; i < n; i++) {
        close[i] = bars[i].close;
        openp[i] = bars[i].open;
    }

    int split_idx = (int)(n * split_ratio);
    int n_test = n - split_idx;

    // Grid search on training set
    BestParams best = grid_search(close.data(), openp.data(), split_idx);

    // Evaluate on test set
    std::vector<double> ema1(n), ema2(n), ema3(n), ema4(n);
    std::vector<char> buy_sig(n), sell_sig(n);

    double oos_return = backtest(
        close.data() + split_idx, openp.data() + split_idx, n_test,
        best.length0, best.p1, best.p2, best.p3,
        ema1.data(), ema2.data(), ema3.data(), ema4.data(),
        buy_sig.data(), sell_sig.data());

    // Full dataset return
    double full_return = backtest(
        close.data(), openp.data(), n,
        best.length0, best.p1, best.p2, best.p3,
        ema1.data(), ema2.data(), ema3.data(), ema4.data(),
        buy_sig.data(), sell_sig.data());

    result.stock_code = fs::path(filepath).stem().string();
    result.data_points = n;
    result.date_start = bars.front().date;
    result.date_end = bars.back().date;
    result.length0 = best.length0;
    result.p1 = best.p1;
    result.p2 = best.p2;
    result.p3 = best.p3;
    result.in_sample_return_pct = std::round(best.best_return * 100.0) / 100.0;
    result.out_of_sample_return_pct = std::round(oos_return * 100.0) / 100.0;
    result.full_return_pct = std::round(full_return * 100.0) / 100.0;

    return true;
}

// ============================================================
// Single stock mode
// ============================================================

void run_single(const std::string& filepath, double split_ratio) {
    std::cout << "Processing: " << filepath << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    OptResult result;
    if (!process_stock(filepath, split_ratio, result)) {
        std::cerr << "Failed to process " << filepath << std::endl;
        return;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "========================================" << std::endl;
    std::cout << "Stock: " << result.stock_code << std::endl;
    std::cout << "Data points: " << result.data_points
              << " (" << result.date_start << " ~ " << result.date_end << ")" << std::endl;
    std::cout << "Best params: length0=" << result.length0
              << " p1=" << result.p1
              << " p2=" << result.p2
              << " p3=" << result.p3 << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "In-sample return:     " << result.in_sample_return_pct << "%" << std::endl;
    std::cout << "Out-of-sample return: " << result.out_of_sample_return_pct << "%" << std::endl;
    std::cout << "Full return:          " << result.full_return_pct << "%" << std::endl;
    std::cout << "Time: " << elapsed << "s" << std::endl;
    std::cout << "(35,672 parameter combinations evaluated)" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// Batch mode (multi-threaded)
// ============================================================

void run_batch(const std::string& data_dir, double split_ratio,
               const std::string& output_csv, int n_threads) {
    // Collect all CSV files
    std::vector<std::string> csv_files;
    for (auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::string stem = entry.path().stem().string();
            if (ext == ".csv" && stem.find("batch_") != 0 && stem.find("results") == std::string::npos) {
                csv_files.push_back(entry.path().string());
            }
        }
    }
    std::sort(csv_files.begin(), csv_files.end());

    int total = (int)csv_files.size();
    std::cout << "========================================" << std::endl;
    std::cout << "Batch mode" << std::endl;
    std::cout << "  Data dir:   " << data_dir << std::endl;
    std::cout << "  Stocks:     " << total << std::endl;
    std::cout << "  Split:      " << split_ratio << std::endl;
    std::cout << "  Threads:    " << n_threads << std::endl;
    std::cout << "  Grid:       35,672 combos per stock" << std::endl;
    std::cout << "  Output:     " << output_csv << std::endl;
    std::cout << "========================================" << std::endl;

    auto t0 = std::chrono::steady_clock::now();

    std::vector<OptResult> results(total);
    std::vector<char> success(total, 0);  // char instead of bool
    std::atomic<int> progress{0};
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};
    std::mutex print_mtx;

    auto worker = [&](int start, int end) {
        for (int idx = start; idx < end; idx++) {
            bool ok = process_stock(csv_files[idx], split_ratio, results[idx]);
            success[idx] = ok ? 1 : 0;

            int done = ++progress;
            if (ok) {
                succeeded++;
                std::lock_guard<std::mutex> lock(print_mtx);
                std::cout << "[" << done << "/" << total << "] "
                          << results[idx].stock_code
                          << std::fixed << std::setprecision(1)
                          << " IS=" << results[idx].in_sample_return_pct << "%"
                          << " OOS=" << results[idx].out_of_sample_return_pct << "%"
                          << " Full=" << results[idx].full_return_pct << "%"
                          << " (l0=" << results[idx].length0
                          << " p1=" << results[idx].p1
                          << " p2=" << results[idx].p2
                          << " p3=" << results[idx].p3 << ")"
                          << std::endl;
            } else {
                failed++;
                std::lock_guard<std::mutex> lock(print_mtx);
                std::cout << "[" << done << "/" << total << "] "
                          << fs::path(csv_files[idx]).stem().string()
                          << " SKIPPED" << std::endl;
            }
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    int chunk = (total + n_threads - 1) / n_threads;
    for (int t = 0; t < n_threads; t++) {
        int start = t * chunk;
        int end = std::min(start + chunk, total);
        if (start < end) {
            threads.emplace_back(worker, start, end);
        }
    }
    for (auto& t : threads) t.join();

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Write CSV results
    std::string output_path = data_dir + "/" + output_csv;
    std::ofstream out(output_path);
    out << "stock_code,data_points,date_start,date_end,length0,p1,p2,p3,"
        << "in_sample_return_pct,out_of_sample_return_pct,full_return_pct" << std::endl;

    std::vector<OptResult> valid_results;
    for (int i = 0; i < total; i++) {
        if (success[i]) {
            auto& r = results[i];
            out << r.stock_code << "," << r.data_points << ","
                << r.date_start << "," << r.date_end << ","
                << r.length0 << "," << r.p1 << "," << r.p2 << "," << r.p3 << ","
                << std::fixed << std::setprecision(2)
                << r.in_sample_return_pct << ","
                << r.out_of_sample_return_pct << ","
                << r.full_return_pct << std::endl;
            valid_results.push_back(r);
        }
    }
    out.close();

    // Summary statistics
    int n_valid = (int)valid_results.size();
    if (n_valid == 0) {
        std::cout << "No valid results." << std::endl;
        return;
    }

    double is_sum = 0, oos_sum = 0;
    int oos_positive = 0, truly_effective = 0, overfitting = 0;
    for (auto& r : valid_results) {
        is_sum += r.in_sample_return_pct;
        oos_sum += r.out_of_sample_return_pct;
        if (r.out_of_sample_return_pct > 0) oos_positive++;
        if (r.in_sample_return_pct > 0 && r.out_of_sample_return_pct > 0) truly_effective++;
        if (r.in_sample_return_pct > 0 && r.out_of_sample_return_pct < 0) overfitting++;
    }

    // Sort by OOS return for top 10
    std::sort(valid_results.begin(), valid_results.end(),
              [](const OptResult& a, const OptResult& b) {
                  return a.out_of_sample_return_pct > b.out_of_sample_return_pct;
              });

    std::cout << "\n========================================" << std::endl;
    std::cout << "BATCH COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Processed:     " << succeeded.load() << "/" << total
              << " (skipped " << failed.load() << ")" << std::endl;
    std::cout << std::fixed;
    std::cout << "Time:          " << std::setprecision(1)
              << elapsed << "s (" << elapsed / 60.0 << " min)" << std::endl;
    std::cout << "Speed:         " << std::setprecision(2) << elapsed / n_valid
              << "s per stock" << std::endl;
    std::cout << "Results saved: " << output_path << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "IS avg return:     " << std::setprecision(2) << is_sum / n_valid << "%" << std::endl;
    std::cout << "OOS avg return:    " << oos_sum / n_valid << "%" << std::endl;
    std::cout << "OOS positive:      " << oos_positive << "/" << n_valid
              << " (" << std::setprecision(1) << 100.0 * oos_positive / n_valid << "%)" << std::endl;
    std::cout << "Truly effective:   " << truly_effective << "/" << n_valid
              << " (" << 100.0 * truly_effective / n_valid << "%)"
              << "  (IS>0 AND OOS>0)" << std::endl;
    std::cout << "Overfitting:       " << overfitting << "/" << n_valid
              << " (" << 100.0 * overfitting / n_valid << "%)"
              << "  (IS>0 but OOS<0)" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Top 10 by OOS return:" << std::endl;
    std::cout << std::setw(10) << "Stock"
              << std::setw(10) << "IS%"
              << std::setw(10) << "OOS%"
              << std::setw(10) << "Full%"
              << "  Params" << std::endl;
    for (int i = 0; i < std::min(10, n_valid); i++) {
        auto& r = valid_results[i];
        std::cout << std::setw(10) << r.stock_code
                  << std::setw(10) << std::setprecision(1) << r.in_sample_return_pct
                  << std::setw(10) << r.out_of_sample_return_pct
                  << std::setw(10) << r.full_return_pct
                  << "  l0=" << r.length0 << " p1=" << r.p1
                  << " p2=" << r.p2 << " p3=" << r.p3
                  << std::endl;
    }
    std::cout << "========================================" << std::endl;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "EMA Crossover Strategy — Brute-force Grid Search (C++)\n\n"
                  << "Usage:\n"
                  << "  Single:  " << argv[0] << " <stock.csv> <split_ratio>\n"
                  << "  Batch:   " << argv[0] << " --batch <split_ratio> [--output results.csv] [--threads N]\n\n"
                  << "Examples:\n"
                  << "  " << argv[0] << " 600610.csv 0.6\n"
                  << "  " << argv[0] << " --batch 0.6 --threads 8\n";
        return 1;
    }

    if (std::string(argv[1]) == "--batch") {
        double split_ratio = std::stod(argv[2]);
        std::string output_csv = "batch_results_cpp.csv";
        int n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads < 1) n_threads = 4;

        for (int i = 3; i < argc; i++) {
            if (std::string(argv[i]) == "--output" && i + 1 < argc) {
                output_csv = argv[++i];
            } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
                n_threads = std::stoi(argv[++i]);
            }
        }

        // Data dir = directory of this executable (or current dir)
        std::string data_dir = fs::path(argv[0]).parent_path().string();
        if (data_dir.empty()) data_dir = ".";

        run_batch(data_dir, split_ratio, output_csv, n_threads);
    } else {
        double split_ratio = std::stod(argv[2]);
        run_single(argv[1], split_ratio);
    }

    return 0;
}
