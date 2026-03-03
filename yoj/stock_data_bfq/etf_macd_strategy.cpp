/*
 * ETF MACD Strategy Runner V2
 * Uses V4 best params (10,24,3) with histogram buy + DIF<0 sell
 * Also runs V2 baseline (8,24,7) and V3 best (8,22,5) for comparison
 * 
 * ETF CSV format: date,open,high,low,close,volume
 * Compiles: g++ -O3 -std=c++17 -pthread -o etf_macd_strategy etf_macd_strategy.cpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

struct DailyBar {
    std::string date;
    double open, close, high, low, volume;
};

struct MonthlyBarFlat {
    double close;
    double volume;
    int first_daily_idx;
    int last_daily_idx;
};

struct StrategyConfig {
    std::string label;
    int fast, slow, signal;
    int buy_mode;   // 0=golden cross, 1=histogram>0
    int sell_mode;   // 0=death cross, 1=DIF<0
};

struct EvalResult {
    int trades = 0;
    double cumulative_return_pct = 0;
    double annualized_return_pct = 0;
    double win_rate = 0;
    double buy_hold_return_pct = 0;
    std::string date_start, date_end;
    std::vector<std::string> trade_log;
};

// ETF name mapping
std::unordered_map<std::string, std::string> g_etf_names = {
    // Indices
    {"sh000001", "上证指数"}, {"sh000016", "上证50"}, {"sh000300", "沪深300"},
    {"sh000852", "中证1000"}, {"sh000905", "中证500"},
    {"sz399001", "深证成指"}, {"sz399006", "创业板指"},
    // Broad ETFs
    {"sh510050", "上证50ETF"}, {"sh510300", "300ETF"}, {"sh510500", "中证500ETF"},
    {"sh510880", "红利ETF"}, {"sh510090", "180ETF"},
    // Sector ETFs
    {"sh512000", "券商ETF"}, {"sh512010", "医药ETF"}, {"sh512070", "非银ETF"},
    {"sh512170", "医疗ETF"}, {"sh512200", "房地产ETF"}, {"sh512290", "生物医药ETF"},
    {"sh512400", "有色金属ETF"}, {"sh512480", "半导体ETF"}, {"sh512660", "军工ETF"},
    {"sh512690", "酒ETF"}, {"sh512720", "计算机ETF"}, {"sh512760", "芯片ETF"},
    {"sh512800", "银行ETF"}, {"sh512880", "证券ETF"}, {"sh512980", "传媒ETF"},
    {"sz159819", "人工智能ETF"}, {"sz159825", "农业ETF"}, {"sz159869", "碳中和ETF"},
    {"sz159870", "化工ETF"}, {"sz159875", "新能源车ETF"}, {"sz159915", "创业板ETF"},
    {"sz159919", "300ETF(深)"}, {"sz159928", "消费ETF"}, {"sz159941", "纳指ETF(深)"},
    {"sz159949", "创业板50ETF"}, {"sz159952", "创新药ETF"}, {"sz159995", "芯片ETF(深)"},
    // Cross-border/Commodity ETFs
    {"sh513050", "中概互联ETF"}, {"sh513060", "恒生医疗ETF"}, {"sh513080", "法国CAC40ETF"},
    {"sh513100", "纳指ETF"}, {"sh513130", "恒生科技ETF"}, {"sh513180", "恒生互联ETF"},
    {"sh513330", "美债ETF"}, {"sh513500", "标普500ETF"}, {"sh513520", "日经ETF"},
    {"sh513660", "恒生ETF"}, {"sh513730", "东南亚科技ETF"}, {"sh513880", "日经225ETF"},
    {"sh513890", "韩国ETF"}, {"sh518380", "黄金股ETF"}, {"sh518800", "黄金ETF"},
    {"sh518880", "黄金ETF(华安)"}, {"sh560080", "中药ETF"}, {"sh561120", "饮料ETF"},
    {"sh561560", "央企红利ETF"}, {"sh562510", "上证科创板ETF"}, {"sh588000", "科创50ETF"},
    {"sh588200", "科创芯片ETF"},
    // Extra indices
    {"sh000688", "科创50指数"}, {"sz399303", "国证2000"}, {"sz399673", "创业板50指数"},
    {"sz399986", "中证新能"}, {"sz399989", "中证医疗"},
};

std::vector<MonthlyBarFlat> aggregate_monthly(const std::vector<DailyBar>& daily, int start, int end) {
    std::vector<MonthlyBarFlat> monthly;
    if (start >= end) return monthly;

    std::string cur_ym = daily[start].date.substr(0, 7);
    double cur_close = daily[start].close;
    double cur_vol = daily[start].volume;
    int cur_first = start, cur_last = start;

    for (int i = start + 1; i < end; ++i) {
        std::string ym = daily[i].date.substr(0, 7);
        if (ym == cur_ym) {
            cur_close = daily[i].close;
            cur_vol += daily[i].volume;
            cur_last = i;
        } else {
            monthly.push_back({cur_close, cur_vol, cur_first, cur_last});
            cur_ym = ym;
            cur_close = daily[i].close;
            cur_vol = daily[i].volume;
            cur_first = i;
            cur_last = i;
        }
    }
    monthly.push_back({cur_close, cur_vol, cur_first, cur_last});
    return monthly;
}

void compute_macd(const std::vector<MonthlyBarFlat>& bars, int fast, int slow, int signal,
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

double compute_annualized(double ratio, double years) {
    if (years <= 0.082) return 0;
    if (ratio <= 0) return -100.0;
    return (std::pow(ratio, 1.0 / years) - 1.0) * 100.0;
}

EvalResult run_strategy(const std::vector<DailyBar>& daily, const std::vector<MonthlyBarFlat>& monthly,
                        const std::vector<double>& dif, const std::vector<double>& dea,
                        const std::vector<double>& hist_vals,
                        const StrategyConfig& cfg, int global_start, int global_end) {
    EvalResult res;
    if (monthly.size() < 3 || global_end <= global_start) return res;

    res.date_start = daily[global_start].date;
    res.date_end = daily[global_end - 1].date;
    double first_open = daily[global_start].open;
    double last_close = daily[global_end - 1].close;
    res.buy_hold_return_pct = (last_close - first_open) / first_open * 100.0;

    bool holding = false;
    double buy_price = 0, capital = 1.0;
    int winning = 0;
    std::string buy_date;

    for (size_t i = 1; i < monthly.size() - 1; ++i) {
        if (!holding) {
            bool buy_signal = false;
            if (cfg.buy_mode == 0) {
                buy_signal = (dif[i-1] <= dea[i-1]) && (dif[i] > dea[i]);
            } else if (cfg.buy_mode == 1) {
                buy_signal = (hist_vals[i-1] <= 0) && (hist_vals[i] > 0);
            }
            if (!buy_signal) continue;

            int exec_idx = monthly[i+1].first_daily_idx;
            if (exec_idx < global_end) {
                buy_price = daily[exec_idx].open;
                buy_date = daily[exec_idx].date;
                holding = true;
            }
        } else {
            bool sell_signal = false;
            if (cfg.sell_mode == 0) {
                sell_signal = (dif[i-1] >= dea[i-1]) && (dif[i] < dea[i]);
            } else if (cfg.sell_mode == 1) {
                sell_signal = (dif[i-1] >= 0) && (dif[i] < 0);
            }
            if (sell_signal) {
                int exec_idx = monthly[i+1].first_daily_idx;
                if (exec_idx < global_end) {
                    double sell_price = daily[exec_idx].open;
                    double ret = (sell_price - buy_price) / buy_price;
                    capital *= (1.0 + ret);
                    if (ret > 0) winning++;
                    res.trades++;
                    char buf[256];
                    snprintf(buf, sizeof(buf), "  Buy: %s @ %.4f -> Sell: %s @ %.4f  Return: %.2f%%",
                             buy_date.c_str(), buy_price, daily[exec_idx].date.c_str(), sell_price, ret * 100);
                    res.trade_log.push_back(buf);
                    holding = false;
                }
            }
        }
    }

    // Force close
    if (holding) {
        double sell_price = daily[global_end - 1].close;
        double ret = (sell_price - buy_price) / buy_price;
        capital *= (1.0 + ret);
        if (ret > 0) winning++;
        res.trades++;
        char buf[256];
        snprintf(buf, sizeof(buf), "  Buy: %s @ %.4f -> Close: %s @ %.4f  Return: %.2f%% (forced)",
                 buy_date.c_str(), buy_price, daily[global_end - 1].date.c_str(), sell_price, ret * 100);
        res.trade_log.push_back(buf);
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

int main() {
    std::string etf_dir = "/ceph/dang_articles/yoj/market_data";

    // Three strategy configs to compare
    std::vector<StrategyConfig> configs = {
        {"V2(8,24,7 GX/DX)",   8, 24, 7, 0, 0},  // V2 baseline: golden cross / death cross
        {"V3(8,22,5 GX/DIF<0)", 8, 22, 5, 0, 1},  // V3 best: golden cross / DIF<0
        {"V4(10,24,3 H/DIF<0)", 10, 24, 3, 1, 1},  // V4 best: histogram>0 / DIF<0
    };

    struct ETFData {
        std::string code, name;
        std::vector<DailyBar> daily;
    };

    // Load all ETFs
    std::vector<ETFData> etfs;
    for (const auto& entry : fs::directory_iterator(etf_dir)) {
        if (entry.path().extension() != ".csv") continue;
        std::string filename = entry.path().stem().string();

        std::ifstream file(entry.path());
        if (!file.is_open()) continue;

        ETFData ed;
        ed.code = filename;
        ed.name = g_etf_names.count(filename) ? g_etf_names[filename] : filename;

        std::string line;
        std::getline(file, line); // header
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string date, s_open, s_high, s_low, s_close, s_vol;
            std::getline(ss, date, ',');
            std::getline(ss, s_open, ',');
            std::getline(ss, s_high, ',');
            std::getline(ss, s_low, ',');
            std::getline(ss, s_close, ',');
            std::getline(ss, s_vol, ',');
            try {
                DailyBar b;
                b.date = date;
                b.open = std::stod(s_open);
                b.high = std::stod(s_high);
                b.low = std::stod(s_low);
                b.close = std::stod(s_close);
                b.volume = std::stod(s_vol);
                ed.daily.push_back(b);
            } catch (...) {}
        }

        if (ed.daily.size() < 60) continue;
        std::sort(ed.daily.begin(), ed.daily.end(), [](const DailyBar& a, const DailyBar& b) {
            return a.date < b.date;
        });
        etfs.push_back(std::move(ed));
    }

    printf("Loaded %zu ETFs/indices.\n\n", etfs.size());

    // Run each config and collect results
    for (const auto& cfg : configs) {
        std::cout << "============================================================\n";
        printf("Strategy: %s\n", cfg.label.c_str());
        printf("  Buy: %s, Sell: %s\n",
               cfg.buy_mode == 0 ? "Golden Cross" : "Histogram > 0",
               cfg.sell_mode == 0 ? "Death Cross" : "DIF < 0");
        std::cout << "============================================================\n\n";

        struct Result {
            std::string code, name;
            double full_ann, train_ann, test_ann, full_ret, bh_ret;
            int trades;
            double win_rate;
            int daily_count;
            std::vector<std::string> trade_log;
        };
        std::vector<Result> results;

        std::vector<double> dif, dea, hist_vals;
        for (const auto& etf : etfs) {
            int n = etf.daily.size();
            int split = n * 0.4;

            // Full period
            auto full_m = aggregate_monthly(etf.daily, 0, n);
            compute_macd(full_m, cfg.fast, cfg.slow, cfg.signal, dif, dea, hist_vals);
            auto full_res = run_strategy(etf.daily, full_m, dif, dea, hist_vals, cfg, 0, n);

            // Train
            auto train_m = aggregate_monthly(etf.daily, 0, split);
            compute_macd(train_m, cfg.fast, cfg.slow, cfg.signal, dif, dea, hist_vals);
            auto train_res = run_strategy(etf.daily, train_m, dif, dea, hist_vals, cfg, 0, split);

            // Test
            auto test_m = aggregate_monthly(etf.daily, split, n);
            compute_macd(test_m, cfg.fast, cfg.slow, cfg.signal, dif, dea, hist_vals);
            auto test_res = run_strategy(etf.daily, test_m, dif, dea, hist_vals, cfg, split, n);

            results.push_back({etf.code, etf.name, full_res.annualized_return_pct,
                               train_res.annualized_return_pct, test_res.annualized_return_pct,
                               full_res.cumulative_return_pct, full_res.buy_hold_return_pct,
                               full_res.trades, full_res.win_rate, (int)etf.daily.size(),
                               full_res.trade_log});
        }

        std::sort(results.begin(), results.end(),
                  [](const Result& a, const Result& b) { return a.full_ann > b.full_ann; });

        printf("%-16s %-20s %6s  %10s %10s %10s  %10s %10s  %6s %6s\n",
               "Code", "Name", "Days", "Full_Ann%", "Train_Ann%", "Test_Ann%",
               "Full_Ret%", "B&H_Ret%", "Trades", "WinR%");
        printf("%-16s %-20s %6s  %10s %10s %10s  %10s %10s  %6s %6s\n",
               "----", "----", "----", "---------", "----------", "---------",
               "---------", "--------", "------", "-----");

        for (const auto& r : results) {
            printf("%-16s %-20s %6d  %10.2f %10.2f %10.2f  %10.2f %10.2f  %6d %6.1f\n",
                   r.code.c_str(), r.name.c_str(), r.daily_count,
                   r.full_ann, r.train_ann, r.test_ann,
                   r.full_ret, r.bh_ret, r.trades, r.win_rate);
        }

        // Save CSV for V4 config (the main one)
        if (cfg.label.find("V4") != std::string::npos) {
            std::ofstream csv("etf_macd_v4_results.csv");
            csv << "code,name,daily_bars,full_annualized,train_annualized,test_annualized,"
                << "full_cumulative_ret,buy_hold_ret,trades,win_rate\n";
            for (const auto& r : results) {
                csv << r.code << "," << r.name << "," << r.daily_count << ","
                    << r.full_ann << "," << r.train_ann << "," << r.test_ann << ","
                    << r.full_ret << "," << r.bh_ret << ","
                    << r.trades << "," << r.win_rate << "\n";
            }
            csv.close();
            printf("\nV4 ETF results saved to etf_macd_v4_results.csv\n");

            // Print trade details for top 5
            std::cout << "\nTop 5 ETF Trade Details:\n";
            int count = 0;
            for (const auto& r : results) {
                if (count >= 5) break;
                if (r.trades == 0) continue;
                printf("\n%s %s (Ann: %.2f%%, Total: %.2f%%)\n",
                       r.code.c_str(), r.name.c_str(), r.full_ann, r.full_ret);
                for (const auto& log : r.trade_log) {
                    std::cout << log << "\n";
                }
                count++;
            }
        }

        std::cout << "\n\n";
    }

    return 0;
}
