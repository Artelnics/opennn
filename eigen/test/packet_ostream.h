#ifndef TEST_PACKET_OSTREAM
#define TEST_PACKET_OSTREAM

#include <type_traits>
#include <ostream>

// Include this header to be able to print Packets while debugging.

template <typename Packet,
          typename EnableIf = std::enable_if_t<Eigen::internal::unpacket_traits<Packet>::vectorizable> >
std::ostream& operator<<(std::ostream& os, const Packet& packet) {
  using Scalar = typename Eigen::internal::unpacket_traits<Packet>::type;
  Scalar v[Eigen::internal::unpacket_traits<Packet>::size];
  Eigen::internal::pstoreu(v, packet);
  os << "{" << v[0];
  for (int i = 1; i < Eigen::internal::unpacket_traits<Packet>::size; ++i) {
    os << "," << v[i];
  }
  os << "}";
  return os;
}

#endif  // TEST_PACKET_OSTREAM