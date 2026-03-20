#ifndef PROB_H_
#define PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void initdata(int i, int j, int k,
              amrex::Array4<amrex::Real> const &rho,
              amrex::Array4<amrex::Real> const &ux,
              amrex::Array4<amrex::Real> const &uy,
              amrex::Array4<amrex::Real> const &uz,
              amrex::Array4<amrex::Real> const &vor,
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_lo,
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &prob_hi,
              amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &dx,
              double nu) {
    amrex::ignore_unused(k);

    constexpr amrex::Real t = 0.0;
    constexpr amrex::Real u_max = 0.02;
    constexpr amrex::Real rho0 = 1.0;
    constexpr amrex::Real pi = amrex::Math::pi<amrex::Real>();

    const amrex::Real kx = 2.0 * pi / (prob_hi[0] - prob_lo[0]);
    const amrex::Real ky = 2.0 * pi / (prob_hi[1] - prob_lo[1]);
    const amrex::Real td = 1.0 / (nu * (kx * kx + ky * ky));

    const amrex::Real x = prob_lo[0] + (0.5 + i) * dx[0];
    const amrex::Real y = prob_lo[1] + (0.5 + j) * dx[1];
    const amrex::Real decay = std::exp(-t / td);
    const auto xphase = amrex::Math::sincos(kx * x);
    const auto yphase = amrex::Math::sincos(ky * y);
    const auto x2phase = amrex::Math::sincos(2.0 * kx * x);
    const auto y2phase = amrex::Math::sincos(2.0 * ky * y);

    ux(i, j, k) = -u_max * std::sqrt(ky / kx) * xphase.second *
                  yphase.first * decay;
    uy(i, j, k) =  u_max * std::sqrt(kx / ky) * xphase.first *
                  yphase.second * decay;
    uz(i, j, k) = 0.0;
    vor(i, j, k) = 0.0;

    const amrex::Real pressure =
        -0.25 * rho0 * u_max * u_max *
        ((ky / kx) * x2phase.second +
         (kx / ky) * y2phase.second) *
        std::exp(-2.0 * t / td);
    rho(i, j, k) = rho0 + 3.0 * pressure;
}

#endif
