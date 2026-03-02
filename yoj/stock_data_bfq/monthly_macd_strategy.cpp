/*
 * 月线 MACD 水下金叉 / 水上死叉 策略 (A股只做多)
 *
 * 策略逻辑:
 *   1. 将日线数据聚合成月K线 (按自然月)
 *   2. 计算月线 MACD(12,26,9): DIF = EMA12 - EMA26, DEA = EMA9(DIF), MACD柱 = 2*(DIF-DEA)
 *   3. 买入信号: DIF < 0 且 DIF 上穿 DEA (水下金叉) → 次月首个交易日开盘买入
 *   4. 卖出信号: DIF > 0 且 DIF 下穿 DEA (水上死叉) → 次月首个交易日开盘卖出
 *   5. 固定规则，无需参数优化
 *
 * 用法:
 *   单只: ./monthly_macd_strategy 600900.csv
 *   批量: ./monthly_macd_strategy --batch [--output results.csv] [--threads 8]
 *
 * 编译:
 *   g++ -O3 -std=c++17 -pthread -o monthly_macd_strategy monthly_macd_strategy.cpp
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
#include <unordered_map>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// MACD parameters (fixed)
// ============================================================
static const int MACD_FAST   = 12;
static const int MACD_SLOW   = 26;
static const int MACD_SIGNAL = 9;

// ============================================================
// Data structures
// ============================================================

struct DailyBar {
    std::string date;   // YYYY-MM-DD
    double open;
    double close;
    double high;
    double low;
    double volume;
};

struct MonthlyBar {
    std::string year_month;   // YYYY-MM
    std::string first_date;   // 该月首个交易日
    std::string last_date;    // 该月最后交易日
    double open;
    double close;
    double high;
    double low;
    double volume;
    int first_daily_idx;      // 该月第一根日K在日线数组中的位置
    int last_daily_idx;       // 该月最后日K在日线数组中的位置
};

struct TradeRecord {
    std::string buy_date;
    std::string sell_date;
    double buy_price;
    double sell_price;
    double return_pct;
    std::string signal_month;  // 产生信号的月份
};

struct StockResult {
    std::string stock_code;
    int daily_bars;
    int monthly_bars;
    std::string date_start;
    std::string date_end;
    int total_trades;
    int winning_trades;
    double win_rate;
    double total_return_pct;
    double avg_return_pct;
    double max_return_pct;
    double min_return_pct;
    double buy_hold_return_pct;   // 买入持有收益 (基准)
    std::vector<TradeRecord> trades;
};

// ============================================================
// CSV parsing — reads date, open, close, high, low, volume
// ============================================================

bool load_csv(const std::string& path, std::vector<DailyBar>& bars) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    std::string line;
    if (!std::getline(file, line)) return false;

    // Parse header to find column indices
    int col_date = -1, col_open = -1, col_close = -1;
    int col_high = -1, col_low = -1, col_volume = -1;
    {
        std::istringstream hss(line);
        std::string token;
        int idx = 0;
        while (std::getline(hss, token, ',')) {
            if (token == "date") col_date = idx;
            else if (token == "open") col_open = idx;
            else if (token == "close") col_close = idx;
            else if (token == "high") col_high = idx;
            else if (token == "low") col_low = idx;
            else if (token == "volume") col_volume = idx;
            idx++;
        }
    }
    if (col_date < 0 || col_open < 0 || col_close < 0 ||
        col_high < 0 || col_low < 0 || col_volume < 0) return false;

    int max_col = std::max({col_date, col_open, col_close, col_high, col_low, col_volume});

    bars.clear();
    bars.reserve(8192);

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

        DailyBar bar;
        bar.date = fields[col_date];
        try {
            bar.open   = std::stod(fields[col_open]);
            bar.close  = std::stod(fields[col_close]);
            bar.high   = std::stod(fields[col_high]);
            bar.low    = std::stod(fields[col_low]);
            bar.volume = std::stod(fields[col_volume]);
        } catch (...) {
            continue;
        }
        if (bar.open <= 0.0 || bar.close <= 0.0) continue;
        bars.push_back(std::move(bar));
    }
    return !bars.empty();
}

// ============================================================
// Aggregate daily bars → monthly bars
// ============================================================

void aggregate_monthly(const std::vector<DailyBar>& daily, std::vector<MonthlyBar>& monthly) {
    monthly.clear();
    if (daily.empty()) return;

    monthly.reserve(daily.size() / 20);  // ~20 trading days per month

    std::string cur_ym;  // current year-month
    MonthlyBar cur;

    for (int i = 0; i < (int)daily.size(); i++) {
        // Extract YYYY-MM from date (YYYY-MM-DD)
        std::string ym = daily[i].date.substr(0, 7);

        if (ym != cur_ym) {
            // Save previous month
            if (!cur_ym.empty()) {
                monthly.push_back(cur);
            }
            // Start new month
            cur_ym = ym;
            cur.year_month = ym;
            cur.first_date = daily[i].date;
            cur.last_date  = daily[i].date;
            cur.open       = daily[i].open;
            cur.close      = daily[i].close;
            cur.high       = daily[i].high;
            cur.low        = daily[i].low;
            cur.volume     = daily[i].volume;
            cur.first_daily_idx = i;
            cur.last_daily_idx  = i;
        } else {
            // Extend current month
            cur.last_date = daily[i].date;
            cur.close     = daily[i].close;         // close = last day's close
            cur.high      = std::max(cur.high, daily[i].high);
            cur.low       = std::min(cur.low, daily[i].low);
            cur.volume   += daily[i].volume;
            cur.last_daily_idx = i;
        }
    }
    // Don't forget the last month
    if (!cur_ym.empty()) {
        monthly.push_back(cur);
    }
}

// ============================================================
// EMA calculation
// ============================================================

void compute_ema(const double* data, int n, int span, double* out) {
    if (n == 0) return;
    double multiplier = 2.0 / (span + 1.0);
    out[0] = data[0];
    for (int i = 1; i < n; i++) {
        out[i] = (data[i] - out[i - 1]) * multiplier + out[i - 1];
    }
}

// ============================================================
// MACD calculation: returns DIF, DEA, MACD_hist arrays
// ============================================================

void compute_macd(const double* close, int n,
                  int fast, int slow, int signal,
                  std::vector<double>& dif,
                  std::vector<double>& dea,
                  std::vector<double>& macd_hist) {
    dif.resize(n);
    dea.resize(n);
    macd_hist.resize(n);

    std::vector<double> ema_fast(n), ema_slow(n);
    compute_ema(close, n, fast, ema_fast.data());
    compute_ema(close, n, slow, ema_slow.data());

    for (int i = 0; i < n; i++) {
        dif[i] = ema_fast[i] - ema_slow[i];
    }

    compute_ema(dif.data(), n, signal, dea.data());

    for (int i = 0; i < n; i++) {
        macd_hist[i] = 2.0 * (dif[i] - dea[i]);
    }
}

// ============================================================
// Strategy backtest
// ============================================================

bool process_stock(const std::string& filepath, StockResult& result) {
    std::vector<DailyBar> daily;
    if (!load_csv(filepath, daily)) return false;

    int n_daily = (int)daily.size();
    if (n_daily < 60) return false;  // need enough data

    // Aggregate to monthly
    std::vector<MonthlyBar> monthly;
    aggregate_monthly(daily, monthly);

    int n_monthly = (int)monthly.size();
    if (n_monthly < MACD_SLOW + MACD_SIGNAL + 2) return false;  // need enough monthly bars for MACD

    // Extract monthly close prices
    std::vector<double> m_close(n_monthly);
    for (int i = 0; i < n_monthly; i++) {
        m_close[i] = monthly[i].close;
    }

    // Compute MACD on monthly data
    std::vector<double> dif, dea, macd_hist;
    compute_macd(m_close.data(), n_monthly, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
                 dif, dea, macd_hist);

    // Generate signals and execute trades
    // Signal at month i → execute at first trading day of month i+1
    std::vector<TradeRecord> trades;
    int position = 0;  // 0 = flat, 1 = long
    double buy_price = 0.0;
    std::string buy_date;
    std::string buy_signal_month;

    // Start from month 1 (need previous month for crossover detection)
    for (int i = 1; i < n_monthly; i++) {
        bool golden_cross = (dif[i - 1] <= dea[i - 1]) && (dif[i] > dea[i]);
        bool death_cross  = (dif[i - 1] >= dea[i - 1]) && (dif[i] < dea[i]);

        bool underwater = dif[i] < 0;   // 水下: DIF < 0
        bool abovewater = dif[i] > 0;   // 水上: DIF > 0

        bool buy_signal  = golden_cross && underwater;  // 水下金叉
        bool sell_signal = death_cross && abovewater;   // 水上死叉

        // Execute at the next month's first trading day
        if (buy_signal && position == 0 && i + 1 < n_monthly) {
            // Buy at next month's first trading day open
            int exec_daily_idx = monthly[i + 1].first_daily_idx;
            buy_price = daily[exec_daily_idx].open;
            buy_date  = daily[exec_daily_idx].date;
            buy_signal_month = monthly[i].year_month;
            position = 1;
        } else if (sell_signal && position == 1 && i + 1 < n_monthly) {
            // Sell at next month's first trading day open
            int exec_daily_idx = monthly[i + 1].first_daily_idx;
            double sell_price = daily[exec_daily_idx].open;
            std::string sell_date = daily[exec_daily_idx].date;

            double ret_pct = (sell_price - buy_price) / buy_price * 100.0;

            TradeRecord tr;
            tr.buy_date     = buy_date;
            tr.sell_date    = sell_date;
            tr.buy_price    = buy_price;
            tr.sell_price   = sell_price;
            tr.return_pct   = ret_pct;
            tr.signal_month = buy_signal_month;
            trades.push_back(tr);

            position = 0;
        }
    }

    // Force close at last available price if still holding
    if (position == 1) {
        double sell_price = daily.back().close;
        std::string sell_date = daily.back().date;
        double ret_pct = (sell_price - buy_price) / buy_price * 100.0;

        TradeRecord tr;
        tr.buy_date     = buy_date;
        tr.sell_date    = sell_date + "(forced)";
        tr.buy_price    = buy_price;
        tr.sell_price   = sell_price;
        tr.return_pct   = ret_pct;
        tr.signal_month = buy_signal_month;
        trades.push_back(tr);
    }

    // Compute statistics
    int n_trades = (int)trades.size();
    int n_winning = 0;
    double total_ret = 0.0;
    double max_ret = -1e18, min_ret = 1e18;

    for (auto& t : trades) {
        total_ret += t.return_pct;
        if (t.return_pct > 0) n_winning++;
        max_ret = std::max(max_ret, t.return_pct);
        min_ret = std::min(min_ret, t.return_pct);
    }

    if (n_trades == 0) {
        max_ret = 0; min_ret = 0;
    }

    // Buy & hold return (baseline)
    double buy_hold_ret = (daily.back().close - daily.front().open) / daily.front().open * 100.0;

    result.stock_code  = fs::path(filepath).stem().string();
    result.daily_bars  = n_daily;
    result.monthly_bars = n_monthly;
    result.date_start  = daily.front().date;
    result.date_end    = daily.back().date;
    result.total_trades = n_trades;
    result.winning_trades = n_winning;
    result.win_rate     = n_trades > 0 ? (100.0 * n_winning / n_trades) : 0.0;
    result.total_return_pct = std::round(total_ret * 100.0) / 100.0;
    result.avg_return_pct   = n_trades > 0 ? std::round(total_ret / n_trades * 100.0) / 100.0 : 0.0;
    result.max_return_pct   = std::round(max_ret * 100.0) / 100.0;
    result.min_return_pct   = std::round(min_ret * 100.0) / 100.0;
    result.buy_hold_return_pct = std::round(buy_hold_ret * 100.0) / 100.0;
    result.trades = std::move(trades);

    return true;
}

// ============================================================
// Single stock mode
// ============================================================

void run_single(const std::string& filepath) {
    std::cout << "Processing: " << filepath << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    StockResult result;
    if (!process_stock(filepath, result)) {
        std::cerr << "Failed to process " << filepath << std::endl;
        return;
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "========================================" << std::endl;
    std::cout << "Stock: " << result.stock_code << std::endl;
    std::cout << "Daily bars:   " << result.daily_bars << std::endl;
    std::cout << "Monthly bars: " << result.monthly_bars << std::endl;
    std::cout << "Period: " << result.date_start << " ~ " << result.date_end << std::endl;
    std::cout << "MACD params:  MACD(" << MACD_FAST << "," << MACD_SLOW << "," << MACD_SIGNAL << ")" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total trades:     " << result.total_trades << std::endl;
    std::cout << "Winning trades:   " << result.winning_trades << std::endl;
    std::cout << "Win rate:         " << result.win_rate << "%" << std::endl;
    std::cout << "Total return:     " << result.total_return_pct << "%" << std::endl;
    std::cout << "Avg return/trade: " << result.avg_return_pct << "%" << std::endl;
    std::cout << "Max return:       " << result.max_return_pct << "%" << std::endl;
    std::cout << "Min return:       " << result.min_return_pct << "%" << std::endl;
    std::cout << "Buy & Hold:       " << result.buy_hold_return_pct << "%" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    if (!result.trades.empty()) {
        std::cout << "Trade details:" << std::endl;
        std::cout << std::setw(4) << "#"
                  << std::setw(12) << "Buy Date"
                  << std::setw(12) << "Sell Date"
                  << std::setw(10) << "Buy Px"
                  << std::setw(10) << "Sell Px"
                  << std::setw(10) << "Return%"
                  << "  Signal" << std::endl;
        int idx = 0;
        for (auto& t : result.trades) {
            std::cout << std::setw(4) << ++idx
                      << std::setw(12) << t.buy_date
                      << std::setw(12) << t.sell_date
                      << std::setw(10) << t.buy_price
                      << std::setw(10) << t.sell_price
                      << std::setw(10) << t.return_pct
                      << "  " << t.signal_month << std::endl;
        }
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Time: " << elapsed << "s" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============================================================
// Batch mode (multi-threaded)
// ============================================================

void run_batch(const std::string& data_dir, const std::string& output_csv, int n_threads) {
    // Collect all CSV files
    std::vector<std::string> csv_files;
    for (auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::string ext  = entry.path().extension().string();
            std::string stem = entry.path().stem().string();
            // Skip result/batch/top files
            if (ext == ".csv" &&
                stem.find("batch_") != 0 &&
                stem.find("results") == std::string::npos &&
                stem.find("top") != 0 &&
                stem.find("svm_") != 0 &&
                stem.find("monthly_") != 0 &&
                stem != "all_stock") {
                csv_files.push_back(entry.path().string());
            }
        }
    }
    std::sort(csv_files.begin(), csv_files.end());

    int total = (int)csv_files.size();
    std::cout << "========================================" << std::endl;
    std::cout << "Monthly MACD Strategy — Batch Mode" << std::endl;
    std::cout << "  Data dir:   " << data_dir << std::endl;
    std::cout << "  Stocks:     " << total << std::endl;
    std::cout << "  MACD:       (" << MACD_FAST << "," << MACD_SLOW << "," << MACD_SIGNAL << ")" << std::endl;
    std::cout << "  Threads:    " << n_threads << std::endl;
    std::cout << "  Output:     " << output_csv << std::endl;
    std::cout << "========================================" << std::endl;

    auto t0 = std::chrono::steady_clock::now();

    std::vector<StockResult> results(total);
    std::vector<char> success(total, 0);
    std::atomic<int> progress{0};
    std::atomic<int> succeeded{0};
    std::atomic<int> failed{0};
    std::mutex print_mtx;

    auto worker = [&](int start, int end) {
        for (int idx = start; idx < end; idx++) {
            bool ok = process_stock(csv_files[idx], results[idx]);
            success[idx] = ok ? 1 : 0;

            int done = ++progress;
            if (ok) {
                succeeded++;
                std::lock_guard<std::mutex> lock(print_mtx);
                std::cout << "[" << done << "/" << total << "] "
                          << results[idx].stock_code
                          << std::fixed << std::setprecision(1)
                          << " trades=" << results[idx].total_trades
                          << " ret=" << results[idx].total_return_pct << "%"
                          << " wr=" << results[idx].win_rate << "%"
                          << " bh=" << results[idx].buy_hold_return_pct << "%"
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
    out << "stock_code,daily_bars,monthly_bars,date_start,date_end,"
        << "total_trades,winning_trades,win_rate,"
        << "total_return_pct,avg_return_pct,max_return_pct,min_return_pct,"
        << "buy_hold_return_pct" << std::endl;

    std::vector<StockResult> valid_results;
    for (int i = 0; i < total; i++) {
        if (success[i]) {
            auto& r = results[i];
            out << r.stock_code << ","
                << r.daily_bars << "," << r.monthly_bars << ","
                << r.date_start << "," << r.date_end << ","
                << r.total_trades << "," << r.winning_trades << ","
                << std::fixed << std::setprecision(2)
                << r.win_rate << ","
                << r.total_return_pct << "," << r.avg_return_pct << ","
                << r.max_return_pct << "," << r.min_return_pct << ","
                << r.buy_hold_return_pct << std::endl;
            valid_results.push_back(std::move(r));
        }
    }
    out.close();

    // Summary statistics
    int n_valid = (int)valid_results.size();
    if (n_valid == 0) {
        std::cout << "No valid results." << std::endl;
        return;
    }

    double ret_sum = 0, bh_sum = 0;
    int trades_total = 0, winning_total = 0;
    int ret_positive = 0, beat_bh = 0, no_trades = 0;

    for (auto& r : valid_results) {
        ret_sum += r.total_return_pct;
        bh_sum  += r.buy_hold_return_pct;
        trades_total  += r.total_trades;
        winning_total += r.winning_trades;
        if (r.total_return_pct > 0) ret_positive++;
        if (r.total_return_pct > r.buy_hold_return_pct) beat_bh++;
        if (r.total_trades == 0) no_trades++;
    }

    // Sort by total return for top 10
    std::sort(valid_results.begin(), valid_results.end(),
              [](const StockResult& a, const StockResult& b) {
                  return a.total_return_pct > b.total_return_pct;
              });

    std::cout << "\n========================================" << std::endl;
    std::cout << "BATCH COMPLETE" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Processed:       " << succeeded.load() << "/" << total
              << " (skipped " << failed.load() << ")" << std::endl;
    std::cout << std::fixed;
    std::cout << "Time:            " << std::setprecision(1)
              << elapsed << "s (" << elapsed / 60.0 << " min)" << std::endl;
    std::cout << "No trades:       " << no_trades << "/" << n_valid << std::endl;
    std::cout << "Total trades:    " << trades_total
              << " (avg " << std::setprecision(1) << (double)trades_total / n_valid << "/stock)" << std::endl;
    std::cout << "Results saved:   " << output_path << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "Avg return:        " << ret_sum / n_valid << "%" << std::endl;
    std::cout << "Avg buy & hold:    " << bh_sum / n_valid << "%" << std::endl;
    std::cout << "Positive return:   " << ret_positive << "/" << n_valid
              << " (" << std::setprecision(1) << 100.0 * ret_positive / n_valid << "%)" << std::endl;
    std::cout << "Beat buy & hold:   " << beat_bh << "/" << n_valid
              << " (" << 100.0 * beat_bh / n_valid << "%)" << std::endl;
    std::cout << "Avg win rate:      " << std::setprecision(2);
    {
        double wr_sum = 0; int wr_cnt = 0;
        for (auto& r : valid_results) {
            if (r.total_trades > 0) { wr_sum += r.win_rate; wr_cnt++; }
        }
        if (wr_cnt > 0) std::cout << wr_sum / wr_cnt << "%" << std::endl;
        else std::cout << "N/A" << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Top 10 by total return:" << std::endl;
    std::cout << std::setw(10) << "Stock"
              << std::setw(8) << "Trades"
              << std::setw(8) << "WinR%"
              << std::setw(10) << "Return%"
              << std::setw(10) << "BuyHold%"
              << std::setw(10) << "AvgRet%"
              << std::endl;
    for (int i = 0; i < std::min(10, n_valid); i++) {
        auto& r = valid_results[i];
        std::cout << std::setw(10) << r.stock_code
                  << std::setw(8) << r.total_trades
                  << std::setw(8) << std::setprecision(1) << r.win_rate
                  << std::setw(10) << r.total_return_pct
                  << std::setw(10) << r.buy_hold_return_pct
                  << std::setw(10) << r.avg_return_pct
                  << std::endl;
    }
    std::cout << "========================================" << std::endl;

    // ========================================
    // Generate Top 100 with stock names
    // ========================================
    // Try multiple candidate paths for all_stock.csv
    std::string name_file;
    {
        std::vector<std::string> candidates = {
            data_dir + "/all_stock.csv",
            data_dir + "/../all_stock.csv",
            fs::canonical(data_dir).parent_path().string() + "/all_stock.csv"
        };
        for (auto& c : candidates) {
            if (fs::exists(c)) { name_file = c; break; }
        }
    }
    std::ifstream nf(name_file);
    std::unordered_map<std::string, std::string> code_to_name;
    if (nf.is_open()) {
        std::string nline;
        while (std::getline(nf, nline)) {
            // Handle BOM
            if (!nline.empty() && (unsigned char)nline[0] == 0xEF) {
                if (nline.size() >= 3) nline = nline.substr(3);
            }
            if (nline.empty()) continue;
            auto pos = nline.find(',');
            if (pos != std::string::npos) {
                std::string code = nline.substr(0, pos);
                std::string name = nline.substr(pos + 1);
                // Remove trailing whitespace/newline
                while (!name.empty() && (name.back() == '\r' || name.back() == '\n' || name.back() == ' '))
                    name.pop_back();
                if (code_to_name.find(code) == code_to_name.end()) {
                    code_to_name[code] = name;
                }
            }
        }
        nf.close();
    }

    // Write top100
    std::string top100_path = data_dir + "/monthly_macd_top100.csv";
    std::ofstream top_out(top100_path);
    top_out << "rank,stock_code,stock_name,total_trades,winning_trades,win_rate,"
            << "total_return_pct,avg_return_pct,max_return_pct,min_return_pct,"
            << "buy_hold_return_pct,date_start,date_end" << std::endl;

    for (int i = 0; i < std::min(100, n_valid); i++) {
        auto& r = valid_results[i];
        std::string name = "N/A";
        auto it = code_to_name.find(r.stock_code);
        if (it != code_to_name.end()) name = it->second;

        top_out << (i + 1) << ","
                << r.stock_code << "," << name << ","
                << r.total_trades << "," << r.winning_trades << ","
                << std::fixed << std::setprecision(2)
                << r.win_rate << ","
                << r.total_return_pct << "," << r.avg_return_pct << ","
                << r.max_return_pct << "," << r.min_return_pct << ","
                << r.buy_hold_return_pct << ","
                << r.date_start << "," << r.date_end << std::endl;
    }
    top_out.close();
    std::cout << "Top 100 saved:   " << top100_path << std::endl;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Monthly MACD Strategy — 水下金叉/水上死叉 (C++)\n\n"
                  << "Usage:\n"
                  << "  Single:  " << argv[0] << " <stock.csv>\n"
                  << "  Batch:   " << argv[0] << " --batch [--output results.csv] [--threads N]\n\n"
                  << "Examples:\n"
                  << "  " << argv[0] << " 600900.csv\n"
                  << "  " << argv[0] << " --batch --threads 16\n";
        return 1;
    }

    if (std::string(argv[1]) == "--batch") {
        std::string output_csv = "monthly_macd_results.csv";
        int n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads < 1) n_threads = 4;

        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "--output" && i + 1 < argc) {
                output_csv = argv[++i];
            } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
                n_threads = std::stoi(argv[++i]);
            }
        }

        // Data dir = directory of this executable (or current dir)
        std::string data_dir = fs::path(argv[0]).parent_path().string();
        if (data_dir.empty()) data_dir = ".";

        run_batch(data_dir, output_csv, n_threads);
    } else {
        run_single(argv[1]);
    }

    return 0;
}
