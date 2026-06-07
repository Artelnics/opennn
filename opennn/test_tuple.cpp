#include <tuple>
#include <vector>
#include <algorithm>

int main() {
    std::vector<std::tuple<int, std::string>> v = {{1, "a"}, {2, "b"}};
    auto it = std::find_if(v.begin(), v.end(), 
                          [](const auto& e) { return std::get<1>(e) == "b"; });
    
    if (it != v.end()) {
        // This is the CORRECT syntax:
        int val = std::get<0>(*it);  // dereference first, then get<>
        
        // This does NOT work (no such member function):
        // int val = it->get<0>();
    }
}
