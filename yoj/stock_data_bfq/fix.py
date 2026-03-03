import re
with open("etf_macd_optimizer_v9.cpp", "r") as f:
    code = f.read()

code = code.replace('if(walk_forward_results.empty()) cout << "\\nWalk-Forward Windows:\\n";\\n        cout << "  Window " << walk_forward_results.size() + 1', 
'''if(walk_forward_results.empty()) cout << "\\nWalk-Forward Windows:\\n";
        cout << "  Window " << walk_forward_results.size() + 1''')

with open("etf_macd_optimizer_v9.cpp", "w") as f:
    f.write(code)
