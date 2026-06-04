#include <ucao/ucao.hpp>

template struct ucao::kernel::FPMultivector<3, 3, 0>;
template ucao::kernel::FPMultivector<3, 3, 0> ucao::kernel::fp_gp<3, 3, 0>(
    const ucao::kernel::FPMultivector<3, 3, 0>&,
    const ucao::kernel::FPMultivector<3, 3, 0>&) noexcept;

template struct ucao::kernel::FPMultivector<4, 3, 1>;
template ucao::kernel::FPMultivector<4, 3, 1> ucao::kernel::fp_gp<4, 3, 1>(
    const ucao::kernel::FPMultivector<4, 3, 1>&,
    const ucao::kernel::FPMultivector<4, 3, 1>&) noexcept;

template struct ucao::pinn::CliffordFieldLayer<3, 3, 0>;
template float ucao::pinn::clifford_pinn_loss_exact<3, 3, 0>(
    const ucao::pinn::CliffordFieldLayer<3, 3, 0>&,
    const float*, const float*, int) noexcept;

template struct ucao::combat::FPRotorChain<3, 3, 0, 8>;
