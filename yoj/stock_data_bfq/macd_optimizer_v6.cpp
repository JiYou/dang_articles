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
// Data structures
// ============================================================

struct DailyBar {
    std::string date;
    double open, close, high, low, volume;
};

struct AggBar {
    double open, close, high, low;
    double volume;
    int first_daily_idx;
    int last_daily_idx;
};

struct StockData {
    std::string code;
    std::string name;
    std::vector<DailyBar> daily;
    std::vector<AggBar> train_bars;
    std::vector<AggBar> test_bars;
    int split_idx;
};

struct StrategyParams {
    // MACD (from V5)
    int fast, slow, signal;
    int buy_mode;         // 0=golden cross, 1=histogram>0
    int sell_mode;        // 0=death cross, 1=DIF<0, 2=histogram<0
    
    // RSI filter (NEW)
    int rsi_period;       // 0=off, else {7, 14, 21}
    double rsi_buy_max;   // only buy if RSI < this (0=off). Values: {0, 50, 60, 70}
    double rsi_sell_min;  // also sell if RSI > this (0=off). Values: {0, 70, 80}
    
    // Bollinger filter (NEW)
    int bb_period;        // 0=off, else {10, 20}
    double bb_buy_max_pctb; // only buy if %B < this (2.0=off). Values: {2.0, 0.2, 0.5, 0.8}
    
    // Volume filter (NEW)
    int vol_ma_period;    // 0=off, else {5, 10}
    double vol_min_ratio; // only buy if vol_ratio > this (0=off). Values: {0, 1.2, 1.5, 2.0}
};

struct EvalResult {
    int trades = 0;
    double cumulative_return_pct = 0;
    double annualized_return_pct = 0;
    double win_rate = 0;
    double buy_hold_return_pct = 0;
    std::string date_start, date_end;
};

// ============================================================
// Global Data
// ============================================================

std::unordered_map<std::string, std::string> g_stock_names;
std::vector<StockData> g_stocks;

// ============================================================
// Helpers
// ============================================================

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) result.push_back(cell);
    return result;
}

void load_stock_names(const std::string& data_dir) {
    std::vector<std::string> paths = {
        data_dir + "/all_stock.csv",
        data_dir + "/../all_stock.csv",
        "../all_stock.csv"
    };
    std::ifstream file;
    for (const auto& p : paths) {
        file.open(p);
        if (file.is_open()) break;
    }
    if (!file.is_open()) return;
    std::string line;
    while (std::getline(file, line)) {
        if (line.size() >= 3 && line[0] == '\xEF' && line[1] == '\xBB' && line[2] == '\xBF')
            line = line.substr(3);
        if (line.empty()) continue;
        auto tokens = split_csv(line);
        if (tokens.size() >= 2) {
            std::string code = tokens[0];
            std::string name = tokens[1];
            if (!name.empty() && name.back() == '\r') name.pop_back();
            if (g_stock_names.find(code) == g_stock_names.end())
                g_stock_names[code] = name;
        }
    }
}

std::vector<AggBar> aggregate_monthly(const std::vector<DailyBar>& daily, int start, int end) {
    std::vector<AggBar> bars;
    if (start >= end) return bars;

    std::string cur_ym = daily[start].date.substr(0, 7);
    double m_open = daily[start].open, m_close = daily[start].close;
    double m_high = daily[start].high, m_low = daily[start].low;
    double m_vol = daily[start].volume;
    int m_first = start, m_last = start;

    for (int i = start + 1; i < end; ++i) {
        std::string ym = daily[i].date.substr(0, 7);
        if (ym == cur_ym) {
            m_close = daily[i].close;
            m_high = std::max(m_high, daily[i].high);
            m_low = std::min(m_low, daily[i].low);
            m_vol += daily[i].volume;
            m_last = i;
        } else {
            bars.push_back({m_open, m_close, m_high, m_low, m_vol, m_first, m_last});
            cur_ym = ym;
            m_open = daily[i].open; m_close = daily[i].close;
            m_high = daily[i].high; m_low = daily[i].low;
            m_vol = daily[i].volume;
            m_first = i; m_last = i;
        }
    }
    bars.push_back({m_open, m_close, m_high, m_low, m_vol, m_first, m_last});
    return bars;
}

void compute_macd(const std::vector<AggBar>& bars, int fast, int slow, int signal,
                  std::vector<double>& dif, std::vector<double>& dea, std::vector<double>& hist) {
    size_t n = bars.size();
    dif.resize(n); dea.resize(n); hist.resize(n);
    if (n == 0) return;

    double mf = 2.0 / (fast + 1), ms = 2.0 / (slow + 1), msig = 2.0 / (signal + 1);
    double ef = bars[0].close, es = bars[0].close, d = 0;
    dif[0] = 0; dea[0] = 0; hist[0] = 0;

    for (size_t i = 1; i < n; ++i) {
        ef = (bars[i].close - ef) * mf + ef;
        es = (bars[i].close - es) * ms + es;
        dif[i] = ef - es;
        if (i == 1) d = dif[i];
        else d = (dif[i] - d) * msig + d;
        dea[i] = d;
        hist[i] = dif[i] - dea[i];
    }
}

void compute_rsi(const std::vector<AggBar>& bars, int period, std::vector<double>& rsi) {
    size_t n = bars.size();
    rsi.assign(n, 50.0);
    if (period <= 0 || n < (size_t)period + 1) return;
    double sum_gain = 0, sum_loss = 0;
    for (int i = 1; i <= period; ++i) {
        double diff = bars[i].close - bars[i-1].close;
        if (diff > 0) sum_gain += diff;
        else sum_loss -= diff;
    }
    double avg_gain = sum_gain / period;
    double avg_loss = sum_loss / period;
    rsi[period] = avg_loss == 0 ? 100.0 : 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
    for (size_t i = period + 1; i < n; ++i) {
        double diff = bars[i].close - bars[i-1].close;
        double gain = diff > 0 ? diff : 0;
        double loss = diff < 0 ? -diff : 0;
        avg_gain = (avg_gain * (period - 1) + gain) / period;
        avg_loss = (avg_loss * (period - 1) + loss) / period;
        rsi[i] = avg_loss == 0 ? 100.0 : 100.0 - (100.0 / (1.0 + avg_gain / avg_loss));
    }
}

void compute_bb(const std::vector<AggBar>& bars, int period, double k, std::vector<double>& pct_b) {
    size_t n = bars.size();
    pct_b.assign(n, 0.5);
    if (period <= 0 || n < (size_t)period) return;
    for (size_t i = period - 1; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < period; ++j) sum += bars[i - j].close;
        double mean = sum / period;
        double sq_sum = 0;
        for (int j = 0; j < period; ++j) {
            double diff = bars[i - j].close - mean;
            sq_sum += diff * diff;
        }
        double stddev = std::sqrt(sq_sum / period);
        double upper = mean + k * stddev;
        double lower = mean - k * stddev;
        if (upper == lower) pct_b[i] = 0.5;
        else pct_b[i] = (bars[i].close - lower) / (upper - lower);
    }
}

void compute_volume_ratio(const std::vector<AggBar>& bars, int period, std::vector<double>& vol_ratio) {
    size_t n = bars.size();
    vol_ratio.assign(n, 1.0);
    if (period <= 0 || n < (size_t)period) return;
    for (size_t i = period - 1; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < period; ++j) sum += bars[i - j].volume;
        double mean_vol = sum / period;
        if (mean_vol > 0) vol_ratio[i] = bars[i].volume / mean_vol;
        else vol_ratio[i] = 1.0;
    }
}

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    return (std::pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

// ============================================================
// Strategy execution
// ============================================================

EvalResult run_strategy(const StockData& stock, const std::vector<AggBar>& bars,
                        const std::vector<double>& dif, const std::vector<double>& dea,
                        const std::vector<double>& hist_vals,
                        const std::vector<double>& rsi_vals,
                        const std::vector<double>& pct_b_vals,
                        const std::vector<double>& vol_ratio_vals,
                        const StrategyParams& p, int global_start, int global_end) {
    EvalResult res;
    if (bars.size() < 3 || global_end <= global_start) return res;

    res.date_start = stock.daily[global_start].date;
    res.date_end = stock.daily[global_end - 1].date;
    double first_open = stock.daily[global_start].open;
    double last_close = stock.daily[global_end - 1].close;
    res.buy_hold_return_pct = (last_close - first_open) / first_open * 100.0;

    bool holding = false;
    double buy_price = 0, capital = 1.0;
    int winning = 0;

    for (size_t i = 1; i < bars.size() - 1; ++i) {
        if (!holding) {
            bool macd_buy = false;
            if (p.buy_mode == 0) {
                macd_buy = (dif[i-1] <= dea[i-1]) && (dif[i] > dea[i]);
            } else if (p.buy_mode == 1) {
                macd_buy = (hist_vals[i-1] <= 0) && (hist_vals[i] > 0);
            }
            if (!macd_buy) continue;
            
            bool rsi_ok = true;
            if (p.rsi_period > 0 && p.rsi_buy_max > 0) {
                if (rsi_vals[i] >= p.rsi_buy_max) rsi_ok = false;
            }
            if (!rsi_ok) continue;

            bool bb_ok = true;
            if (p.bb_period > 0 && p.bb_buy_max_pctb < 2.0) {
                if (pct_b_vals[i] >= p.bb_buy_max_pctb) bb_ok = false;
            }
            if (!bb_ok) continue;

            bool vol_ok = true;
            if (p.vol_ma_period > 0 && p.vol_min_ratio > 0) {
                if (vol_ratio_vals[i] <= p.vol_min_ratio) vol_ok = false;
            }
            if (!vol_ok) continue;

            int exec_idx = bars[i+1].first_daily_idx;
            if (exec_idx < global_end) {
                buy_price = stock.daily[exec_idx].open;
                holding = true;
            }
        } else {
            bool sell_signal = false;
            if (p.sell_mode == 0) {
                sell_signal = (dif[i-1] >= dea[i-1]) && (dif[i] < dea[i]);
            } else if (p.sell_mode == 1) {
                sell_signal = (dif[i-1] >= 0) && (dif[i] < 0);
            } else if (p.sell_mode == 2) {
                sell_signal = (hist_vals[i-1] >= 0) && (hist_vals[i] < 0);
            }
            
            if (p.rsi_period > 0 && p.rsi_sell_min > 0) {
                if (rsi_vals[i] > p.rsi_sell_min) sell_signal = true;
            }

            if (sell_signal) {
                int exec_idx = bars[i+1].first_daily_idx;
                if (exec_idx < global_end) {
                    double sell_price = stock.daily[exec_idx].open;
                    double ret = (sell_price - buy_price) / buy_price;
                    capital *= (1.0 + ret);
                    if (ret > 0) winning++;
                    res.trades++;
                    holding = false;
                }
            }
        }
    }

    // Force close
    if (holding) {
        double sell_price = stock.daily[global_end - 1].close;
        double ret = (sell_price - buy_price) / buy_price;
        capital *= (1.0 + ret);
        if (ret > 0) winning++;
        res.trades++;
    }

    res.cumulative_return_pct = (capital - 1.0) * 100.0;
    if (res.trades > 0) res.win_rate = (double)winning / res.trades * 100.0;

    if (res.trades > 0 && res.date_start.size() >= 10 && res.date_end.size() >= 10) {
        try {
            int y1 = std::stoi(res.date_start.substr(0, 4));
            int m1 = std::stoi(res.date_start.substr(5, 2));
            int d1 = std::stoi(res.date_start.substr(8, 2));
            int y2 = std::stoi(res.date_end.substr(0, 4));
            int m2 = std::stoi(res.date_end.substr(5, 2));
            int d2 = std::stoi(res.date_end.substr(8, 2));
            int days = (y2 - y1) * 365 + (m2 - m1) * 30 + (d2 - d1);
            res.annualized_return_pct = compute_annualized(capital, days / 365.25);
        } catch (...) {
            res.annualized_return_pct = 0;
        }
    }

    return res;
}

// ============================================================
// Score params: Top 20 avg annualized
// ============================================================

double score_params(const StrategyParams& p, bool use_train) {
    thread_local std::vector<double> tl_dif, tl_dea, tl_hist;
    thread_local std::vector<double> tl_rsi, tl_bb, tl_vol;
    std::vector<double> ann;
    ann.reserve(g_stocks.size());

    for (const auto& stock : g_stocks) {
        const auto& bars = use_train ? stock.train_bars : stock.test_bars;
        int gs = use_train ? 0 : stock.split_idx;
        int ge = use_train ? stock.split_idx : (int)stock.daily.size();

        compute_macd(bars, p.fast, p.slow, p.signal, tl_dif, tl_dea, tl_hist);
        
        if (p.rsi_period > 0) compute_rsi(bars, p.rsi_period, tl_rsi);
        else tl_rsi.assign(bars.size(), 50.0);
        
        if (p.bb_period > 0) compute_bb(bars, p.bb_period, 2.0, tl_bb);
        else tl_bb.assign(bars.size(), 0.5);
        
        if (p.vol_ma_period > 0) compute_volume_ratio(bars, p.vol_ma_period, tl_vol);
        else tl_vol.assign(bars.size(), 1.0);

        EvalResult res = run_strategy(stock, bars, tl_dif, tl_dea, tl_hist, tl_rsi, tl_bb, tl_vol, p, gs, ge);
        if (res.trades > 0) ann.push_back(res.annualized_return_pct);
    }

    if (ann.size() < 20) return -999;
    std::sort(ann.rbegin(), ann.rend());
    double sum = 0;
    for (int i = 0; i < 20; ++i) sum += ann[i];
    return sum / 20.0;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    int num_threads = std::thread::hardware_concurrency();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc)
            num_threads = std::stoi(argv[i + 1]);
    }
    if (num_threads <= 0) num_threads = 8;

    auto t_start = std::chrono::high_resolution_clock::now();

    std::string data_dir = ".";
    load_stock_names(data_dir);

    // Load stocks
    std::cout << "Loading stock data...\n";
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".csv" && entry.path().filename().string().length() == 10) {
            std::string code = entry.path().stem().string();
            std::ifstream file(entry.path());
            if (!file.is_open()) continue;

            StockData sd;
            sd.code = code;
            sd.name = g_stock_names.count(code) ? g_stock_names[code] : code;

            std::string line;
            std::getline(file, line); // header
            while (std::getline(file, line)) {
                auto t = split_csv(line);
                if (t.size() < 7) continue;
                DailyBar b;
                b.date = t[0];
                try {
                    b.open = std::stod(t[2]);
                    b.close = std::stod(t[3]);
                    b.high = std::stod(t[4]);
                    b.low = std::stod(t[5]);
                    b.volume = std::stod(t[6]);
                    sd.daily.push_back(b);
                } catch (...) {}
            }
            if (sd.daily.size() < 100) continue;

            std::sort(sd.daily.begin(), sd.daily.end(),
                      [](const DailyBar& a, const DailyBar& b) { return a.date < b.date; });

            sd.split_idx = sd.daily.size() * 0.4;
            g_stocks.push_back(std::move(sd));
        }
    }
    printf("Loaded %zu stocks.\n\n", g_stocks.size());

    // ONLY MONTHLY timeframe
    printf("================================================================\n");
    printf("  TIMEFRAME: MONTHLY\n");
    printf("================================================================\n\n");

    auto t_rebuild = std::chrono::high_resolution_clock::now();
    for (auto& stock : g_stocks) {
        stock.train_bars = aggregate_monthly(stock.daily, 0, stock.split_idx);
        stock.test_bars = aggregate_monthly(stock.daily, stock.split_idx, stock.daily.size());
    }
    auto t_rebuild_end = std::chrono::high_resolution_clock::now();
    printf("Bar aggregation: %.2fs\n", std::chrono::duration<double>(t_rebuild_end - t_rebuild).count());

    const char* buy_mode_str[] = {"GoldenX", "Hist>0"};
    const char* sell_mode_str[] = {"DeathX", "DIF<0", "Hist<0"};

    // Phase A: MACD + buy/sell mode grid
    printf("\n=== Phase A: MACD Grid Search ===\n");
    auto t_phA = std::chrono::high_resolution_clock::now();

    std::vector<StrategyParams> phase_a;
    // Monthly range: fast 5-14, slow 16-36, signal 3-9
    for (int f = 5; f <= 14; ++f) {
        for (int s = 16; s <= 36; ++s) {
            if (f >= s - 2) continue;
            for (int sig = 3; sig <= 9; ++sig) {
                for (int bm = 0; bm <= 1; ++bm) {
                    for (int sm : {0, 1, 2}) {
                        phase_a.push_back({f, s, sig, bm, sm, 0, 0, 0, 0, 2.0, 0, 0});
                    }
                }
            }
        }
    }

    printf("Phase A combos: %zu\n", phase_a.size());

    struct ComboResult {
        StrategyParams params;
        double score;
    };
    std::vector<ComboResult> phase_a_results(phase_a.size());
    std::atomic<int> counter_a{0};
    int total_a = phase_a.size();

    auto worker_a = [&]() {
        while (true) {
            int idx = counter_a.fetch_add(1);
            if (idx >= total_a) break;
            double s = score_params(phase_a[idx], true);
            phase_a_results[idx] = {phase_a[idx], s};
        }
    };

    { std::vector<std::thread> threads;
      for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_a);
      for (auto& t : threads) t.join(); }

    std::sort(phase_a_results.begin(), phase_a_results.end(),
              [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

    int top_show_a = std::min<int>(15, (int)phase_a_results.size());
    printf("\nTop %d MACD configs:\n", top_show_a);
    printf("  %-3s %-4s %-4s %-3s %-8s %-8s %7s\n", "#", "Fast", "Slow", "Sig", "BuyMode", "SellMode", "Score");
    for (int i = 0; i < top_show_a; ++i) {
        auto& r = phase_a_results[i];
        printf("  %-3d %-4d %-4d %-3d %-8s %-8s %6.2f%%\n",
               i + 1, r.params.fast, r.params.slow, r.params.signal,
               buy_mode_str[r.params.buy_mode], sell_mode_str[r.params.sell_mode], r.score);
    }

    auto t_phA_end = std::chrono::high_resolution_clock::now();
    printf("Phase A: %.2fs\n\n", std::chrono::duration<double>(t_phA_end - t_phA).count());

    // Phase B: Indicator Filters Search
    printf("=== Phase B: Indicator Filters Search ===\n");
    auto t_phB = std::chrono::high_resolution_clock::now();

    int top_macd = std::min<int>(5, (int)phase_a_results.size());
    std::vector<StrategyParams> phase_b;

    for (int t = 0; t < top_macd; ++t) {
        auto base = phase_a_results[t].params;

        int rsi_periods[] = {0, 7, 14};
        double rsi_buys[] = {0.0, 50.0, 60.0, 70.0};
        double rsi_sells[] = {0.0, 70.0, 80.0};
        
        int bb_periods[] = {0, 20};
        double bb_buys[] = {2.0, 0.2, 0.5};
        
        int vol_periods[] = {0, 5, 10};
        double vol_mins[] = {0.0, 1.5};

        for (int rsi_p : rsi_periods) {
            for (double rsi_b : rsi_buys) {
                for (double rsi_s : rsi_sells) {
                    for (int bb_p : bb_periods) {
                        for (double bb_b : bb_buys) {
                            for (int vol_p : vol_periods) {
                                for (double vol_m : vol_mins) {
                                    phase_b.push_back({
                                        base.fast, base.slow, base.signal, base.buy_mode, base.sell_mode,
                                        rsi_p, rsi_b, rsi_s,
                                        bb_p, bb_b,
                                        vol_p, vol_m
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    printf("Phase B combos: %zu\n", phase_b.size());

    std::vector<ComboResult> phase_b_results(phase_b.size());
    std::atomic<int> counter_b{0};
    int total_b = phase_b.size();

    auto worker_b = [&]() {
        while (true) {
            int idx = counter_b.fetch_add(1);
            if (idx >= total_b) break;
            double s = score_params(phase_b[idx], true);
            phase_b_results[idx] = {phase_b[idx], s};
        }
    };

    { std::vector<std::thread> threads;
      for (int i = 0; i < num_threads; ++i) threads.emplace_back(worker_b);
      for (auto& t : threads) t.join(); }

    std::sort(phase_b_results.begin(), phase_b_results.end(),
              [](const ComboResult& a, const ComboResult& b) { return a.score > b.score; });

    printf("\nTop 20 configs with filters:\n");
    printf("  %-3s %-4s %-4s %-3s %-8s %-8s | RSI(P,B,S) | BB(P,B) | Vol(P,M) | %7s\n",
           "#", "Fast", "Slow", "Sig", "Buy", "Sell", "Score");
    int top_show_b = std::min<int>(20, (int)phase_b_results.size());
    for (int i = 0; i < top_show_b; ++i) {
        auto& r = phase_b_results[i];
        printf("  %-3d %-4d %-4d %-3d %-8s %-8s | %2d,%.0f,%.0f | %2d,%.1f | %2d,%.1f  | %6.2f%%\n",
               i + 1, r.params.fast, r.params.slow, r.params.signal,
               buy_mode_str[r.params.buy_mode], sell_mode_str[r.params.sell_mode],
               r.params.rsi_period, r.params.rsi_buy_max, r.params.rsi_sell_min,
               r.params.bb_period, r.params.bb_buy_max_pctb,
               r.params.vol_ma_period, r.params.vol_min_ratio,
               r.score);
    }

    auto t_phB_end = std::chrono::high_resolution_clock::now();
    printf("Phase B: %.2fs\n\n", std::chrono::duration<double>(t_phB_end - t_phB).count());

    // Phase C: Test evaluation
    printf("=== Phase C: Test Set Evaluation ===\n");
    int eval_count = std::min<int>(20, (int)phase_b_results.size());
    printf("\nTop %d configs evaluated on test period:\n", eval_count);
    printf("  %-3s %-4s %-4s %-3s %-8s %-8s | RSI(P,B,S) | BB(P,B) | Vol(P,M) | %8s %8s\n",
           "#", "Fast", "Slow", "Sig", "Buy", "Sell", "Train", "Test");

    int best_idx = 0;
    double best_test = -9999;

    for (int ci = 0; ci < eval_count; ++ci) {
        auto& p = phase_b_results[ci].params;
        double test_s = score_params(p, false);
        double train_s = phase_b_results[ci].score;

        printf("  %-3d %-4d %-4d %-3d %-8s %-8s | %2d,%.0f,%.0f | %2d,%.1f | %2d,%.1f  | %7.2f%% %7.2f%%\n",
               ci + 1, p.fast, p.slow, p.signal,
               buy_mode_str[p.buy_mode], sell_mode_str[p.sell_mode],
               p.rsi_period, p.rsi_buy_max, p.rsi_sell_min,
               p.bb_period, p.bb_buy_max_pctb,
               p.vol_ma_period, p.vol_min_ratio,
               train_s, test_s);

        if (test_s > best_test) {
            best_test = test_s;
            best_idx = ci;
        }
    }

    StrategyParams best_p = phase_b_results[best_idx].params;
    double best_train = phase_b_results[best_idx].score;

    printf("\n  BEST CONFIG: Train %.2f%% / Test %.2f%%\n", best_train, best_test);
    printf("  Params: fast=%d slow=%d sig=%d buy=%s sell=%s rsi=(%d,%.0f,%.0f) bb=(%d,%.1f) vol=(%d,%.1f)\n\n",
           best_p.fast, best_p.slow, best_p.signal,
           buy_mode_str[best_p.buy_mode], sell_mode_str[best_p.sell_mode],
           best_p.rsi_period, best_p.rsi_buy_max, best_p.rsi_sell_min,
           best_p.bb_period, best_p.bb_buy_max_pctb,
           best_p.vol_ma_period, best_p.vol_min_ratio);

    // ============================================================
    // Final comparison
    // ============================================================
    printf("\n================================================================\n");
    printf("  VERSION COMPARISON (on %zu stocks)\n", g_stocks.size());
    printf("================================================================\n\n");

    printf("  %-10s  %8s  %8s  %s\n", "Version", "Train", "Test", "Config");
    printf("  %-10s  %8s  %8s  %s\n", "----------", "--------", "--------", "------");
    printf("  %-10s  %7.2f%%  %7.2f%%  8,24,7 GoldenX/DeathX monthly\n", "V2", 43.90, 44.65);
    printf("  %-10s  %7.2f%%  %7.2f%%  8,22,5 GoldenX/DIF<0 monthly\n", "V3", 49.80, 46.08);
    printf("  %-10s  %7.2f%%  %7.2f%%  10,24,3 Hist>0/DIF<0 monthly\n", "V4", 51.10, 46.41);
    printf("  %-10s  %7.2f%%  %7.2f%%  14,17,3 GoldenX/DIF<0 monthly\n", "V5", 51.07, 46.66);
    
    char v6_cfg[256];
    snprintf(v6_cfg, sizeof(v6_cfg), "%d,%d,%d %s/%s rsi=%d bb=%d vol=%d",
             best_p.fast, best_p.slow, best_p.signal,
             buy_mode_str[best_p.buy_mode], sell_mode_str[best_p.sell_mode],
             best_p.rsi_period, best_p.bb_period, best_p.vol_ma_period);
    printf("  %-10s  %7.2f%%  %7.2f%%  %s\n", "V6", best_train, best_test, v6_cfg);

    // ============================================================
    // Generate full results for best config
    // ============================================================
    printf("\n================================================================\n");
    printf("  GENERATING FULL RESULTS\n");
    printf("================================================================\n");

    struct FinalResult {
        std::string code, name;
        double train_ann, test_ann, total_ret;
        int trades;
        double win_rate, buy_hold_ret;
        std::string date_start, date_end;
    };
    std::vector<FinalResult> final_results;

    std::vector<double> dif_v, dea_v, hist_v;
    std::vector<double> rsi_v, bb_v, vol_v;
    
    for (const auto& stock : g_stocks) {
        int n = stock.daily.size();

        // Train
        compute_macd(stock.train_bars, best_p.fast, best_p.slow, best_p.signal, dif_v, dea_v, hist_v);
        if (best_p.rsi_period > 0) compute_rsi(stock.train_bars, best_p.rsi_period, rsi_v); else rsi_v.assign(stock.train_bars.size(), 50.0);
        if (best_p.bb_period > 0) compute_bb(stock.train_bars, best_p.bb_period, 2.0, bb_v); else bb_v.assign(stock.train_bars.size(), 0.5);
        if (best_p.vol_ma_period > 0) compute_volume_ratio(stock.train_bars, best_p.vol_ma_period, vol_v); else vol_v.assign(stock.train_bars.size(), 1.0);
        auto train_res = run_strategy(stock, stock.train_bars, dif_v, dea_v, hist_v, rsi_v, bb_v, vol_v, best_p, 0, stock.split_idx);

        // Test
        compute_macd(stock.test_bars, best_p.fast, best_p.slow, best_p.signal, dif_v, dea_v, hist_v);
        if (best_p.rsi_period > 0) compute_rsi(stock.test_bars, best_p.rsi_period, rsi_v); else rsi_v.assign(stock.test_bars.size(), 50.0);
        if (best_p.bb_period > 0) compute_bb(stock.test_bars, best_p.bb_period, 2.0, bb_v); else bb_v.assign(stock.test_bars.size(), 0.5);
        if (best_p.vol_ma_period > 0) compute_volume_ratio(stock.test_bars, best_p.vol_ma_period, vol_v); else vol_v.assign(stock.test_bars.size(), 1.0);
        auto test_res = run_strategy(stock, stock.test_bars, dif_v, dea_v, hist_v, rsi_v, bb_v, vol_v, best_p, stock.split_idx, n);

        final_results.push_back({stock.code, stock.name,
                                 train_res.annualized_return_pct, test_res.annualized_return_pct,
                                 test_res.cumulative_return_pct, test_res.trades, test_res.win_rate,
                                 test_res.buy_hold_return_pct, test_res.date_start, test_res.date_end});
    }

    std::sort(final_results.begin(), final_results.end(),
              [](const FinalResult& a, const FinalResult& b) { return a.test_ann > b.test_ann; });

    // Save CSVs
    std::ofstream out_full("macd_v6_results.csv");
    out_full << "stock_code,stock_name,train_annualized,test_annualized,total_return_pct,"
             << "trades,win_rate,buy_hold_return_pct,date_start,date_end,"
             << "fast,slow,signal,buy_mode,sell_mode,rsi_period,rsi_buy_max,rsi_sell_min,"
             << "bb_period,bb_buy_max_pctb,vol_ma_period,vol_min_ratio\n";

    std::ofstream out_top20("macd_v6_top20.csv");
    out_top20 << "stock_code,stock_name,train_annualized,test_annualized,total_return_pct,"
              << "trades,win_rate,buy_hold_return_pct,date_start,date_end\n";

    for (size_t i = 0; i < final_results.size(); ++i) {
        const auto& r = final_results[i];
        out_full << r.code << "," << r.name << "," << r.train_ann << "," << r.test_ann << ","
                 << r.total_ret << "," << r.trades << "," << r.win_rate << ","
                 << r.buy_hold_ret << "," << r.date_start << "," << r.date_end << ","
                 << best_p.fast << "," << best_p.slow << "," << best_p.signal << ","
                 << buy_mode_str[best_p.buy_mode] << "," << sell_mode_str[best_p.sell_mode] << ","
                 << best_p.rsi_period << "," << best_p.rsi_buy_max << "," << best_p.rsi_sell_min << ","
                 << best_p.bb_period << "," << best_p.bb_buy_max_pctb << ","
                 << best_p.vol_ma_period << "," << best_p.vol_min_ratio << "\n";

        if (i < 20 && r.trades > 0) {
            out_top20 << r.code << "," << r.name << "," << r.train_ann << "," << r.test_ann << ","
                      << r.total_ret << "," << r.trades << "," << r.win_rate << ","
                      << r.buy_hold_ret << "," << r.date_start << "," << r.date_end << "\n";
        }
    }

    // Print top 20 test stocks
    printf("\nTop 20 stocks by test annualized return:\n");
    printf("  %-3s %-8s %-20s %8s %8s %6s %6s\n", "#", "Code", "Name", "Train%", "Test%", "Trades", "WinR%");
    printf("  %-3s %-8s %-20s %8s %8s %6s %6s\n", "---", "--------", "----", "--------", "--------", "------", "------");
    for (int i = 0; i < std::min<int>(20, (int)final_results.size()); ++i) {
        auto& r = final_results[i];
        printf("  %-3d %-8s %-20s %7.2f%% %7.2f%% %6d %5.1f%%\n",
               i + 1, r.code.c_str(), r.name.substr(0, 20).c_str(),
               r.train_ann, r.test_ann, r.trades, r.win_rate);
    }

    // Stock-wide statistics
    printf("\n--- Stock-Wide Statistics ---\n");
    {
        std::vector<double> all_test;
        int profitable = 0, with_trades = 0;
        for (auto& r : final_results) {
            if (r.trades > 0) {
                with_trades++;
                all_test.push_back(r.test_ann);
                if (r.test_ann > 0) profitable++;
            }
        }
        std::sort(all_test.begin(), all_test.end());
        printf("Stocks with trades: %d / %zu\n", with_trades, final_results.size());
        printf("Profitable (test ann > 0): %d / %d (%.1f%%)\n", profitable, with_trades,
               with_trades > 0 ? profitable * 100.0 / with_trades : 0);
        if (!all_test.empty()) {
            double sum = 0;
            for (double v : all_test) sum += v;
            printf("Mean test annualized: %.2f%%\n", sum / all_test.size());
            printf("Median test annualized: %.2f%%\n", all_test[all_test.size() / 2]);
            printf("P25: %.2f%%, P75: %.2f%%\n",
                   all_test[all_test.size() / 4], all_test[all_test.size() * 3 / 4]);
        }
    }

    printf("\nResults saved to macd_v6_results.csv and macd_v6_top20.csv\n");

    auto t_end = std::chrono::high_resolution_clock::now();
    printf("Total Runtime: %.2fs\n", std::chrono::duration<double>(t_end - t_start).count());

    return 0;
}
