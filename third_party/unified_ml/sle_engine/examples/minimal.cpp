#include "sle/engine.hpp"

#include <iostream>

static sle::BitVector from_string(const std::string& bits) {
    sle::BitVector v(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) v.set(i, bits[i] == '1');
    return v;
}

int main() {
    sle::BooleanCascade cascade(3);
    cascade.add_gate({0, 1, 2, 0x96});

    sle::HardLogicContract contract{from_string("00000000"), from_string("00000000")};
    sle::Engine engine(cascade, sle::BitMixer{}, contract);

    auto result = engine.run({from_string("10101010"), from_string("11001100"), from_string("11110000")});
    std::cout << "SLE output: " << result.to_string() << '\n';
    std::cout << "confidence(popcount/m): " << result.density() << '\n';
    return 0;
}
