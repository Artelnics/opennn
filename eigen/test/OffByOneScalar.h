
// A Scalar with internal representation T+1 so that zero is internally
// represented by T(1). This is used to test memory fill.
// 
template<typename T>
class OffByOneScalar {
 public:
  OffByOneScalar() : val_(1) {}
  OffByOneScalar(const OffByOneScalar& other) {
    *this = other;
  }
  OffByOneScalar& operator=(const OffByOneScalar& other) {
    val_ = other.val_;
    return *this;
  }
  
  OffByOneScalar(T val) : val_(val + 1) {}
  OffByOneScalar& operator=(T val) {
    val_ = val + 1;
  }
  
  operator T() const {
    return val_ - 1;
  }
 
 private:
  T val_;
};
