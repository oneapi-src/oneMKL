#include <sstream>
#include <array>

namespace global {

/* for logging results with InputTestController */
std::stringstream log{};
std::array<char, 1024> buffer{};
std::string pad{};

} // namespace global
