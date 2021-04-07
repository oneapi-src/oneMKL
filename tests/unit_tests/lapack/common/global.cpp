#include <sstream>
#include <array>

/* for logging results with InputTestController */
namespace global {
std::stringstream log{};
std::array<char, 1024> buffer{};
} // namespace global
