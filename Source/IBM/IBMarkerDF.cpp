#include "IBM/IBMarkerDF.H"

#include <AMReX_Array4.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
// NOTE: this AMReX tree does not ship AMReX_ParallelFor.H; ParallelFor lives in
// GpuLaunch.
#include <AMReX_GpuLaunch.H>
#include <AMReX_Print.H>

#include <cmath>

using namespace amrex;

namespace {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Real
delta_function(Real xf, Real xp, Real h, int type) noexcept {
  // Matches IAMReX DiffusedIB_Parallel::deltaFunction (value/h scaling).
  const Real rr = Math::abs(xf - xp) / h;
  Real value = 0.0_rt;

  if (type == 0) { // FOUR_POINT_IB
    if (rr < 1.0_rt) {
      value = 0.125_rt * (3.0_rt - 2.0_rt * rr +
                          std::sqrt(1.0_rt + 4.0_rt * rr - 4.0_rt * rr * rr));
    } else if (rr < 2.0_rt) {
      value = 0.125_rt * (5.0_rt - 2.0_rt * rr -
                          std::sqrt(-7.0_rt + 12.0_rt * rr - 4.0_rt * rr * rr));
    } else {
      value = 0.0_rt;
    }
  } else { // THREE_POINT_IB
    if (rr < 0.5_rt) {
      value = (1.0_rt + std::sqrt(1.0_rt - 3.0_rt * rr * rr)) / 3.0_rt;
    } else if (rr < 1.5_rt) {
      value = (5.0_rt - 3.0_rt * rr -
               std::sqrt(-2.0_rt + 6.0_rt * rr - 3.0_rt * rr * rr)) /
              6.0_rt;
    } else {
      value = 0.0_rt;
    }
  }

  return value / h;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
omega_cross_r(const Real om[3], Real rx, Real ry, Real rz, Real &cx, Real &cy,
              Real &cz) noexcept {
  cx = om[1] * rz - om[2] * ry;
  cy = om[2] * rx - om[0] * rz;
  cz = om[0] * ry - om[1] * rx;
}

} // namespace

IBMarkerDF::IBMarkerDF(Geometry const &geom, DistributionMapping const &dm,
                       BoxArray const &ba, Real x0, Real y0, Real z0, Real R,
                       MarkerIBParams const &par)
    : m_geom(geom), m_x0(x0), m_y0(y0), m_z0(z0), m_R(R), m_par(par) {
  build_markers(dm, ba);
}

Long IBMarkerDF::num_markers() const {
  return m_markers ? m_markers->TotalNumberOfParticles() : 0;
}

void IBMarkerDF::build_markers(DistributionMapping const &dm,
                               BoxArray const &ba) {
  m_markers = std::make_unique<MarkerContainer>(m_geom, dm, ba);

  const auto dx = m_geom.CellSizeArray();
  const Real h = dx[0];
  const Real pi = Math::pi<Real>();

  int Ml = m_par.n_marker;
  Real dv = 0.0_rt;

#if (AMREX_SPACEDIM == 2)
  if (Ml <= 0) {
    Ml = static_cast<int>(std::ceil(2.0_rt * pi * m_R / h));
    Ml = std::max(Ml, 16);
  }
  // Approximate annulus area of thickness h.
  dv = (2.0_rt * pi * m_R * h) / static_cast<Real>(Ml);

  // theta=pi/2 -> z=0, and phi is the polar angle.
  std::vector<Real> phi_h(Ml + 1, 0.0_rt);
  std::vector<Real> theta_h(Ml + 1, 0.5_rt * pi);
  for (int i = 1; i <= Ml; ++i) {
    phi_h[i] = 2.0_rt * pi * Real(i - 1) / Real(Ml);
  }
#else
  // IAMReX-like heuristic: build markers on a thin shell at radius R.
  if (Ml <= 0) {
    const Real rd = m_par.rd;
    const Real a1 = Math::powi<3>(m_R - (rd - 0.5_rt) * h);
    const Real a2 = Math::powi<3>(m_R - (rd + 0.5_rt) * h);
    const Real denom = 3.0_rt * h * h * h / 4.0_rt / pi;
    Ml = static_cast<int>((a1 - a2) / denom);
    Ml = std::max(Ml, 32);
  }
  {
    const Real rd = m_par.rd;
    const Real a1 = Math::powi<3>(m_R - (rd - 0.5_rt) * h);
    const Real a2 = Math::powi<3>(m_R - (rd + 0.5_rt) * h);
    dv = (a1 - a2) / (3.0_rt * Real(Ml) / 4.0_rt / pi);
  }

  // Golden-spiral parameterization (ported from IAMReX), but shifted to be
  // 1-based so we can index with particle.id() directly.
  std::vector<Real> phi_h(Ml + 1, 0.0_rt);
  std::vector<Real> theta_h(Ml + 1, 0.0_rt);
  Real phi = 0.0_rt;
  for (int marker_index = 0; marker_index < Ml; ++marker_index) {
    const Real Hk = -1.0_rt + 2.0_rt * Real(marker_index) / (Real(Ml) - 1.0_rt);
    const Real theta = std::acos(Hk);
    if (marker_index == 0 || marker_index == Ml - 1) {
      phi = 0.0_rt;
    } else {
      phi = std::fmod(phi + 3.809_rt / std::sqrt(Real(Ml)) /
                                std::sqrt(1.0_rt - Hk * Hk),
                      2.0_rt * pi);
    }
    const int idx = marker_index + 1;
    phi_h[idx] = phi;
    theta_h[idx] = theta;
  }
#endif

  m_dv = dv;

  // Copy angle tables to device for UpdateLagrangianMarker.
  m_phiK_d.resize(Ml + 1);
  m_thetaK_d.resize(Ml + 1);
  Gpu::copy(Gpu::hostToDevice, phi_h.begin(), phi_h.end(), m_phiK_d.begin());
  Gpu::copy(Gpu::hostToDevice, theta_h.begin(), theta_h.end(),
            m_thetaK_d.begin());

  // Insert markers (on IOProcessor) into a single tile, then Redistribute.
  auto &ptile = m_markers->GetParticles(0)[{0, 0}];

  if (ParallelDescriptor::MyProc() == ParallelDescriptor::IOProcessorNumber()) {
    for (int i = 1; i <= Ml; ++i) {
      MarkerContainer::ParticleType p;
      p.id() = i;
      p.cpu() = ParallelDescriptor::MyProc();
      p.pos(0) = m_x0;
      p.pos(1) = m_y0;
#if (AMREX_SPACEDIM == 3)
      p.pos(2) = m_z0;
#endif

      std::array<ParticleReal, num_Real> attr{};
      std::array<int, num_Int> idarr{};
      idarr[M_ID] = 0; // single body id (index into our single-body state)

      ptile.push_back(p);
      ptile.push_back_real(attr);
      ptile.push_back_int(idarr);
    }
  }

  m_markers->Redistribute();
  update_lagrangian_marker();

  if (m_par.verbose) {
    amrex::Print() << "[IBMarkerDF] Built " << num_markers()
                   << " markers, dv=" << m_dv
                   << " delta_type=" << m_par.delta_type
                   << " loop_ns=" << m_par.loop_ns << "\n";
  }
}

void IBMarkerDF::update_lagrangian_marker() const {
  // IAMReX semantics: update positions and reset per-marker arrays to zero,
  // then Redistribute.
  const auto *phiK = m_phiK_d.data();
  const auto *thetaK = m_thetaK_d.data();
  const Real x0 = m_x0;
  const Real y0 = m_y0;
#if (AMREX_SPACEDIM == 3)
  const Real z0 = m_z0;
#endif
  const Real R = m_R;
  const int start_id = m_start_id;

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto *uP = soa.GetRealData(U_Marker).data();
    auto *vP = soa.GetRealData(V_Marker).data();
    auto *wP = soa.GetRealData(W_Marker).data();
    auto *fxP = soa.GetRealData(Fx_Marker).data();
    auto *fyP = soa.GetRealData(Fy_Marker).data();
    auto *fzP = soa.GetRealData(Fz_Marker).data();
    auto *mxP = soa.GetRealData(Mx_Marker).data();
    auto *myP = soa.GetRealData(My_Marker).data();
    auto *mzP = soa.GetRealData(Mz_Marker).data();

    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      const int m_id = particles[i].id();
      const int idx = m_id - start_id;

      // Position on the sphere/circle.
      particles[i].pos(0) =
          x0 + R * std::sin(thetaK[idx]) * std::cos(phiK[idx]);
      particles[i].pos(1) =
          y0 + R * std::sin(thetaK[idx]) * std::sin(phiK[idx]);
#if (AMREX_SPACEDIM == 3)
      particles[i].pos(2) = z0 + R * std::cos(thetaK[idx]);
#endif

      uP[i] = 0.0_rt;
      vP[i] = 0.0_rt;
      wP[i] = 0.0_rt;
      fxP[i] = 0.0_rt;
      fyP[i] = 0.0_rt;
      fzP[i] = 0.0_rt;
      mxP[i] = 0.0_rt;
      myP[i] = 0.0_rt;
      mzP[i] = 0.0_rt;
    });
  }

  m_markers->Redistribute();
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void velocity_interpolation_cic(
    P const &p, Real &Up, Real &Vp, Real &Wp, Array4<Real const> const &E,
    int EulerVIndex, GpuArray<Real, AMREX_SPACEDIM> const &plo,
    GpuArray<Real, AMREX_SPACEDIM> const &dx, int type) noexcept {
  const Real d = AMREX_D_TERM(dx[0], *dx[1], *dx[2]);

  const Real lx = (p.pos(0) - plo[0]) / dx[0];
  const Real ly = (p.pos(1) - plo[1]) / dx[1];
#if (AMREX_SPACEDIM == 3)
  const Real lz = (p.pos(2) - plo[2]) / dx[2];
#else
  const Real lz = 0.0_rt;
#endif

  const int i0 = static_cast<int>(std::floor(lx));
  const int j0 = static_cast<int>(std::floor(ly));
  const int k0 = static_cast<int>(std::floor(lz));

  Real u = 0.0_rt, v = 0.0_rt, w = 0.0_rt;

  for (int ii = -2 + type; ii < 3 - type; ++ii) {
    const Real xi = plo[0] + (Real(i0 + ii) + 0.5_rt) * dx[0];
    const Real tU = delta_function(xi, p.pos(0), dx[0], type);

    for (int jj = -2 + type; jj < 3 - type; ++jj) {
      const Real yi = plo[1] + (Real(j0 + jj) + 0.5_rt) * dx[1];
      const Real tV = delta_function(yi, p.pos(1), dx[1], type);

#if (AMREX_SPACEDIM == 3)
      for (int kk = -2 + type; kk < 3 - type; ++kk) {
        const Real zi = plo[2] + (Real(k0 + kk) + 0.5_rt) * dx[2];
        const Real tW = delta_function(zi, p.pos(2), dx[2], type);

        const Real delta_value = tU * tV * tW;
        u += delta_value * E(i0 + ii, j0 + jj, k0 + kk, EulerVIndex + 0) * d;
        v += delta_value * E(i0 + ii, j0 + jj, k0 + kk, EulerVIndex + 1) * d;
        w += delta_value * E(i0 + ii, j0 + jj, k0 + kk, EulerVIndex + 2) * d;
      }
#else
      const Real delta_value = tU * tV;
      u += delta_value * E(i0 + ii, j0 + jj, 0, EulerVIndex + 0) * d;
      v += delta_value * E(i0 + ii, j0 + jj, 0, EulerVIndex + 1) * d;
      w += 0.0_rt;
#endif
    }
  }

  Up = u;
  Vp = v;
  Wp = w;
}

void IBMarkerDF::velocity_interpolation(MultiFab const &EulerVel,
                                        int euler_vel_comp,
                                        int delta_type) const {
  // Fill boundary before interpolation (IAMReX semantics).
  MultiFab &E = const_cast<MultiFab &>(EulerVel);
  E.FillBoundary(euler_vel_comp, 3, m_geom.periodicity());

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto *uP = soa.GetRealData(U_Marker).data();
    auto *vP = soa.GetRealData(V_Marker).data();
    auto *wP = soa.GetRealData(W_Marker).data();

    auto const &Efab = E[pti].const_array();

    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      Real Up = 0.0_rt, Vp = 0.0_rt, Wp = 0.0_rt;
      velocity_interpolation_cic(particles[i], Up, Vp, Wp, Efab, euler_vel_comp,
                                 plo, dx, delta_type);
      uP[i] += Up;
      vP[i] += Vp;
      wP[i] += Wp;
    });
  }
}

void IBMarkerDF::compute_lagrangian_force(Real dt) const {
  // Matches IAMReX: F = (Ub + omega x r - U_marker) / dt
  const Real ub[3] = {m_par.ubx, m_par.uby, m_par.ubz};
  const Real om[3] = {m_par.omx, m_par.omy, m_par.omz};
  const Real x0 = m_x0;
  const Real y0 = m_y0;
#if (AMREX_SPACEDIM == 3)
  const Real z0 = m_z0;
#endif

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto const *uP = soa.GetRealData(U_Marker).data();
    auto const *vP = soa.GetRealData(V_Marker).data();
    auto const *wP = soa.GetRealData(W_Marker).data();
    auto *fxP = soa.GetRealData(Fx_Marker).data();
    auto *fyP = soa.GetRealData(Fy_Marker).data();
    auto *fzP = soa.GetRealData(Fz_Marker).data();

    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      const Real rx = particles[i].pos(0) - x0;
      const Real ry = particles[i].pos(1) - y0;
#if (AMREX_SPACEDIM == 3)
      const Real rz = particles[i].pos(2) - z0;
#else
      const Real rz = 0.0_rt;
#endif

      Real urx, ury, urz;
      omega_cross_r(om, rx, ry, rz, urx, ury, urz);

      const Real Ux = ub[0] + urx;
      const Real Uy = ub[1] + ury;
      const Real Uz = ub[2] + urz;

      fxP[i] += (Ux - uP[i]) / dt;
      fyP[i] += (Uy - vP[i]) / dt;
      fzP[i] += (Uz - wP[i]) / dt;
    });
  }
}

template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void force_spreading_cic(
    P const &p, Real fxP, Real fyP, Real fzP, Array4<Real> const &E,
    int EulerFIndex, GpuArray<Real, AMREX_SPACEDIM> const &plo,
    GpuArray<Real, AMREX_SPACEDIM> const &dx, int type) noexcept {
  const Real lx = (p.pos(0) - plo[0]) / dx[0];
  const Real ly = (p.pos(1) - plo[1]) / dx[1];
#if (AMREX_SPACEDIM == 3)
  const Real lz = (p.pos(2) - plo[2]) / dx[2];
#else
  const Real lz = 0.0_rt;
#endif

  const int i0 = static_cast<int>(std::floor(lx));
  const int j0 = static_cast<int>(std::floor(ly));
  const int k0 = static_cast<int>(std::floor(lz));

  for (int ii = -2 + type; ii < 3 - type; ++ii) {
    const Real xi = plo[0] + (Real(i0 + ii) + 0.5_rt) * dx[0];
    const Real tU = delta_function(xi, p.pos(0), dx[0], type);

    for (int jj = -2 + type; jj < 3 - type; ++jj) {
      const Real yi = plo[1] + (Real(j0 + jj) + 0.5_rt) * dx[1];
      const Real tV = delta_function(yi, p.pos(1), dx[1], type);

#if (AMREX_SPACEDIM == 3)
      for (int kk = -2 + type; kk < 3 - type; ++kk) {
        const Real zi = plo[2] + (Real(k0 + kk) + 0.5_rt) * dx[2];
        const Real tW = delta_function(zi, p.pos(2), dx[2], type);

        const Real delta_value = tU * tV * tW;
        Gpu::Atomic::AddNoRet(&E(i0 + ii, j0 + jj, k0 + kk, EulerFIndex + 0),
                              delta_value * fxP);
        Gpu::Atomic::AddNoRet(&E(i0 + ii, j0 + jj, k0 + kk, EulerFIndex + 1),
                              delta_value * fyP);
        Gpu::Atomic::AddNoRet(&E(i0 + ii, j0 + jj, k0 + kk, EulerFIndex + 2),
                              delta_value * fzP);
      }
#else
      const Real delta_value = tU * tV;
      Gpu::Atomic::AddNoRet(&E(i0 + ii, j0 + jj, 0, EulerFIndex + 0),
                            delta_value * fxP);
      Gpu::Atomic::AddNoRet(&E(i0 + ii, j0 + jj, 0, EulerFIndex + 1),
                            delta_value * fyP);
#endif
    }
  }
}

void IBMarkerDF::force_spreading(MultiFab &EulerForce, int euler_force_comp,
                                 int delta_type) const {
  const Real dv = m_dv;
  const Real x0 = m_x0;
  const Real y0 = m_y0;
#if (AMREX_SPACEDIM == 3)
  const Real z0 = m_z0;
#endif

  const auto dx = m_geom.CellSizeArray();
  const auto plo = m_geom.ProbLoArray();

  for (MarkerParIter pti(*m_markers, 0); pti.isValid(); ++pti) {
    auto *particles = pti.GetArrayOfStructs().data();
    const Long np = pti.numParticles();

    auto &soa = pti.GetStructOfArrays();
    auto *fxP = soa.GetRealData(Fx_Marker).data();
    auto *fyP = soa.GetRealData(Fy_Marker).data();
    auto *fzP = soa.GetRealData(Fz_Marker).data();
    auto *mxP = soa.GetRealData(Mx_Marker).data();
    auto *myP = soa.GetRealData(My_Marker).data();
    auto *mzP = soa.GetRealData(Mz_Marker).data();

    auto const &E = EulerForce[pti].array();

    ParallelFor(np, [=] AMREX_GPU_DEVICE(int i) noexcept {
      // IAMReX multiplies by dv before spreading.
      Real fx = fxP[i] * dv;
      Real fy = fyP[i] * dv;
      Real fz = fzP[i] * dv;

      const Real rx = particles[i].pos(0) - x0;
      const Real ry = particles[i].pos(1) - y0;
#if (AMREX_SPACEDIM == 3)
      const Real rz = particles[i].pos(2) - z0;
#else
      const Real rz = 0.0_rt;
#endif

      // Store moment about body center (r x F).
      mxP[i] = ry * fz - rz * fy;
      myP[i] = rz * fx - rx * fz;
      mzP[i] = rx * fy - ry * fx;

      force_spreading_cic(particles[i], fx, fy, fz, E, euler_force_comp, plo,
                          dx, delta_type);
    });
  }

  EulerForce.SumBoundary(euler_force_comp, 3, m_geom.periodicity());
}

void IBMarkerDF::velocity_correction(MultiFab &EulerVel,
                                     MultiFab const &EulerForce,
                                     int euler_vel_comp, int euler_force_comp,
                                     Real dt) const {
  MultiFab::Saxpy(EulerVel, dt, EulerForce, euler_force_comp, euler_vel_comp, 3,
                  0);
}

void IBMarkerDF::update_forcing(MultiFab const &ucc, MultiFab const &vcc,
                                MultiFab const &wcc, MultiFab &Fx, MultiFab &Fy,
                                MultiFab &Fz, Real dt) const {
  // Build Euler velocity scratch: (u,v,w)
  MultiFab EulerVel(ucc.boxArray(), ucc.DistributionMap(), 3, 2);
  MultiFab EulerForceIter(ucc.boxArray(), ucc.DistributionMap(), 3, 2);
  MultiFab EulerForceTotal(ucc.boxArray(), ucc.DistributionMap(), 3, 2);

  EulerVel.setVal(0.0);
  EulerForceIter.setVal(0.0);
  EulerForceTotal.setVal(0.0);

  for (MFIter mfi(EulerVel, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.fabbox();

    auto const u = ucc[mfi].const_array();
    auto const v = vcc[mfi].const_array();
    auto const w = wcc[mfi].const_array();
    auto ev = EulerVel[mfi].array();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      ev(i, j, k, 0) = u(i, j, k, 0);
      ev(i, j, k, 1) = v(i, j, k, 0);
      ev(i, j, k, 2) = w(i, j, k, 0);
    });
  }

  // Multi-direct forcing loop (IAMReX loop_ns semantics), with total-force
  // accumulation.
  const int nloop = std::max(m_par.loop_ns, 1);
  for (int it = 0; it < nloop; ++it) {
    EulerForceIter.setVal(0.0);

    update_lagrangian_marker();
    velocity_interpolation(EulerVel, 0, m_par.delta_type);
    compute_lagrangian_force(dt);
    force_spreading(EulerForceIter, 0, m_par.delta_type);

    // Velocity correction is applied in-loop, and we accumulate total Eulerian
    // force.
    velocity_correction(EulerVel, EulerForceIter, 0, 0, dt);
    MultiFab::Add(EulerForceTotal, EulerForceIter, 0, 0, 3, 0);
  }

  // Export to your existing (Fx,Fy,Fz) convention.
  for (MFIter mfi(Fx, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box &bx = mfi.fabbox();
    auto fx = Fx[mfi].array();
    auto fy = Fy[mfi].array();
    auto fz = Fz[mfi].array();
    auto ef = EulerForceTotal[mfi].const_array();

    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      fx(i, j, k, 0) = ef(i, j, k, 0);
      fy(i, j, k, 0) = ef(i, j, k, 1);
      fz(i, j, k, 0) = ef(i, j, k, 2);
    });
  }

  Fx.FillBoundary(m_geom.periodicity());
  Fy.FillBoundary(m_geom.periodicity());
  Fz.FillBoundary(m_geom.periodicity());
}
