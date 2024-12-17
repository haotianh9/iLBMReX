/*
 * A simplified single file version of the lattice boltzmann exmaple.
 *
 */

#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
// D2Q9 parameters
const unsigned int ndir = 9;
const double w0 = 4.0/9.0;  // zero weight
const double ws = 1.0/9.0;  // adjacent weight
const double wd = 1.0/36.0; // diagonal weight
const double wi[] = {w0,ws,ws,ws,ws,wd,wd,wd,wd};
const int dirx[] = {0,1,0,-1, 0,1,-1,-1, 1};
const int diry[] = {0,0,1, 0,-1,1, 1,-1,-1};

void taylor_green(amrex::Real t, amrex::MultiFab &rho, amrex::MultiFab &ux, amrex::MultiFab &uy, double rho0, double u_max, double nu, int Nx, int Ny, amrex::GpuArray<amrex::Real, 2> dx, int Nghost)
{
    // amrex::Print() << "rho0: " << rho0 << "  u_max: " << u_max << "\n";
    // loop over boxes
    // for (amrex::MFIter mfi(rho); mfi.isValid(); ++mfi)
    // {
    //     const amrex::Box &bx = mfi.validbox();

    //     const amrex::Array4<amrex::Real> &rho_local = rho.array(mfi);
    //     const amrex::Array4<amrex::Real> &ux_local = ux.array(mfi);
    //     const amrex::Array4<amrex::Real> &uy_local = uy.array(mfi);

    //     amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j)
    //                        {
    //                            // **********************************
    //                            // SET VALUES FOR EACH CELL
    //                            // **********************************
    //                         //    amrex::Print() << "dx: " << dx[0] << " , " << dx[1] << "\n";
                               
    //                            double kx = 2.0 * M_PI / Nx;
    //                            double ky = 2.0 * M_PI / Ny;
    //                            double td = 1.0 / (nu * (kx * kx + ky * ky));

    //                            double X = i + 0.5;
    //                            double Y = j + 0.5;
    //                            ux_local(i, j, 0) = -u_max * sqrt(ky / kx) * cos(kx * X) * sin(ky * Y) * exp(-1.0 * t / td);
    //                            uy_local(i, j, 0) = u_max * sqrt(kx / ky) * sin(kx * X) * cos(ky * Y) * exp(-1.0 * t / td);
    //                            double P = -0.25 * rho0 * u_max * u_max * ((ky / kx) * cos(2.0 * kx * X) + (kx / ky) * cos(2.0 * ky * Y)) * exp(-2.0 * t / td);
    //                            rho_local(i, j, 0) = rho0 + 3.0 * P; 
    //                         //    amrex::Print() << "i,j: " << i << " , " << j << "  rho: " << rho_local(i, j, 0) << "\n";
                               
    //                            });
                               
    // }
    
    const amrex::MultiArray4<amrex::Real>& rho_arrs = rho.arrays();
    const amrex::MultiArray4<amrex::Real>& ux_arrs = ux.arrays();
    const amrex::MultiArray4<amrex::Real>& uy_arrs = uy.arrays();
    const amrex::IntVect ngs(Nghost);

    amrex::ParallelFor(rho, ngs, [=] AMREX_GPU_DEVICE( int nbx, int i, int j,int k) noexcept {

        double kx = 2.0 * M_PI / Nx;
        double ky = 2.0 * M_PI / Ny;
        double td = 1.0 / (nu * (kx * kx + ky * ky));

        double X = i + 0.5;
        double Y = j + 0.5;
        ux_arrs[nbx](i,j,k) = -u_max * sqrt(ky / kx) * cos(kx * X) * sin(ky * Y) * exp(-1.0 * t / td);
        uy_arrs[nbx](i,j,k) = u_max * sqrt(kx / ky) * sin(kx * X) * cos(ky * Y) * exp(-1.0 * t / td);
        double P = -0.25 * rho0 * u_max * u_max * ((ky / kx) * cos(2.0 * kx * X) + (kx / ky) * cos(2.0 * ky * Y)) * exp(-2.0 * t / td);
        rho_arrs[nbx](i,j,k) = rho0 + 3.0 * P; 
    });
}



void init_equilibrium(amrex::MultiFab &f1,amrex::MultiFab &rho, amrex::MultiFab &ux, amrex::MultiFab &uy ,int Nghost){
    const amrex::MultiArray4<amrex::Real>& f1_arrs = f1.arrays();
    const amrex::MultiArray4<amrex::Real>& rho_arrs = rho.arrays();
    const amrex::MultiArray4<amrex::Real>& ux_arrs = ux.arrays();
    const amrex::MultiArray4<amrex::Real>& uy_arrs = uy.arrays();
    const amrex::IntVect ngs(Nghost);

    amrex::ParallelFor(f1, ngs, [=] AMREX_GPU_DEVICE( int nbx, int i, int j,int k) noexcept {

        for(unsigned int i_dir = 0; i_dir < ndir; ++i_dir)
                               {
                                double cidotu = dirx[i_dir]* ux_arrs[nbx](i,j,k) + diry[i_dir]* uy_arrs[nbx](i,j,k);

                                // amrex::Print() <<cidotu << "\t" << rho_local(i,j,0) << "\t" << ux_local(i,j,0) << "\t" << uy_local(i,j,0) << "\n";

                                f1_arrs[nbx](i,j,k,i_dir)= wi[i_dir]*rho_arrs[nbx](i,j,k)*(1.0 + 3.0*cidotu + 4.5*cidotu*cidotu - 1.5*(ux_arrs[nbx](i,j,k)*ux_arrs[nbx](i,j,k)+uy_arrs[nbx](i,j,k)*uy_arrs[nbx](i,j,k)));
                                // f1_local(i,j,k) = 0.0;
                                // amrex::Print() << "i,j,k: " << i << " , " << j <<  " , " << k << "  f1: " << f1_local(i, j, k) << "\n";
                               }
    });
}


void stream(amrex::MultiFab &f_old,amrex::MultiFab &f_new,  int Nx, int Ny,int Nghost){
  
    const amrex::MultiArray4<amrex::Real>& f_old_arrs = f_old.arrays();
    const amrex::MultiArray4<amrex::Real>& f_new_arrs = f_new.arrays();

    const amrex::IntVect ngs(Nghost);

    amrex::ParallelFor(f_old, ngs, [=] AMREX_GPU_DEVICE( int nbx, int i, int j,int k) noexcept {
        // amrex::Print() << "nbx: " << nbx << " i: " << i << " j: " << j << "\n";
        for(unsigned int i_dir = 0; i_dir < ndir; ++i_dir)
                               {
                                int imd = (Nx+i-dirx[i_dir])%Nx;
                                int jmd = (Ny+j-diry[i_dir])%Ny;
                                // if (i_dir == 1){
                                //     amrex::Print() << i_dir<<"\t" << i <<"\t" << j  <<"\t"<< imd <<"\t" <<jmd << "\n";
                                // }
                                
#if (AMREX_SPACEDIM == 2)
                                f_new_arrs[nbx](i,j,k,i_dir) = f_old_arrs[nbx](imd,jmd,k,i_dir);
#elif (AMREX_SPACEDIM == 3)
                                int kmd = (Nz+k-dirz[i_dir])%Nz;
                                f_new_arrs[nbx](i,j,k,i_dir) = f_old_arrs[nbx](imd,jmd,kmd,i_dir);
#endif
                               }
    });
}


void compute_rho_u(amrex::MultiFab &f1,amrex::MultiFab &rho, amrex::MultiFab &ux, amrex::MultiFab &uy,int Nghost){


    

    const amrex::MultiArray4<amrex::Real>& f1_arrs = f1.arrays();
    const amrex::MultiArray4<amrex::Real>& rho_arrs = rho.arrays();
    const amrex::MultiArray4<amrex::Real>& ux_arrs = ux.arrays();
    const amrex::MultiArray4<amrex::Real>& uy_arrs = uy.arrays();
    const amrex::IntVect ngs(Nghost);

    amrex::ParallelFor(f1, ngs, [=] AMREX_GPU_DEVICE( int nbx, int i, int j,int k) noexcept {
        double rho_temp=0.0;
        double ux_temp=0.0;
        double uy_temp=0.0;
                          
                       
        for(unsigned int i_dir = 0; i_dir < ndir; ++i_dir)
                    {

                    rho_temp+=f1_arrs[nbx](i,j,k,i_dir);
                    ux_temp+=dirx[i_dir]*f1_arrs[nbx](i,j,k,i_dir);
                    uy_temp+=diry[i_dir]*f1_arrs[nbx](i,j,k,i_dir);
                    
                    }
        rho_arrs[nbx](i,j,k) = rho_temp;
        ux_arrs[nbx](i,j,k) = ux_temp/rho_temp;
        uy_arrs[nbx](i,j,k) = uy_temp/rho_temp;
                                
    });


}

void collide(amrex::MultiFab &f1,amrex::MultiFab &f2,amrex::MultiFab &rho, amrex::MultiFab &ux, amrex::MultiFab &uy,double nu,int Nghost)
{
    // useful constants
    double tauinv = 2.0/(6.0*nu+1.0); // 1/tau
    double omtauinv = 1.0-tauinv;     // 1 - 1/tau

    const amrex::MultiArray4<amrex::Real>& f1_arrs = f1.arrays();
    const amrex::MultiArray4<amrex::Real>& f2_arrs = f2.arrays();
    const amrex::MultiArray4<amrex::Real>& rho_arrs = rho.arrays();
    const amrex::MultiArray4<amrex::Real>& ux_arrs = ux.arrays();
    const amrex::MultiArray4<amrex::Real>& uy_arrs = uy.arrays();
    const amrex::IntVect ngs(Nghost);

    amrex::ParallelFor(f2, ngs, [=] AMREX_GPU_DEVICE( int nbx, int i, int j,int k) noexcept {
       
                       
        for(unsigned int i_dir = 0; i_dir < ndir; ++i_dir)
                    {

                    double cidotu = dirx[i_dir]*ux_arrs[nbx](i,j,k) + diry[i_dir]*uy_arrs[nbx](i,j,k) ;
                                // calculate equilibrium
                                double feq = wi[i_dir]*rho_arrs[nbx](i,j,k)*(1.0 + 3.0*cidotu + 4.5*cidotu*cidotu - 1.5*(ux_arrs[nbx](i,j,k)*ux_arrs[nbx](i,j,k)+uy_arrs[nbx](i,j,k)*uy_arrs[nbx](i,j,k)));
                                
                                // relax to equilibrium
                                f2_arrs[nbx](i,j,k,i_dir)  = omtauinv*f1_arrs[nbx](i,j,k,i_dir) + tauinv*feq;
                    
                    }
      
                                
    });
}


int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {

        // **********************************
        // DECLARE SIMULATION PARAMETERS
        // **********************************

        int scale;
        int Nx, Ny;
        int NSTEP;
        
       
        int max_grid_size;
        int plot_int;
        // LBM parameters
        double nu = 1.0 / 6.0;
        double tau = 3.0 * nu + 0.5;
        

        
       
        // amrex::Print() << "begin reading data " << "\n";
        // **********************************
        // READ PARAMETER VALUES FROM INPUT DATA
        // **********************************
        // inputs parameters
        {
            // ParmParse is way of reading inputs from the inputs file
            // pp.get means we require the inputs file to have it
            // pp.query means we optionally need the inputs file to have it - but we must supply a default here
            amrex::ParmParse pp;

            // The domain is broken into boxes of size max_grid_size
            pp.get("max_grid_size", max_grid_size);
            amrex::Print() << "max_grid_size: " << max_grid_size << "\n";


            pp.get("scale", scale);
            amrex::Print() << "scale: " << scale << "\n";
            
            pp.get("Nx", Nx);
            Nx = Nx * scale;
            Ny = Nx;

            // Default nsteps to 10, allow us to set it to something else in the inputs file
            NSTEP = 10;
            pp.query("NSTEP", NSTEP);
            NSTEP = NSTEP * scale * scale;
            amrex::Print() << "NSTEP: " << NSTEP << "\n";
            // Default plot_int to -1, allow us to set it to something else in the inputs file
            //  If plot_int < 0 then no plot files will be written
            plot_int = -1;
            pp.query("plot_int", plot_int);
        }
        // Taylor-Green parameters
        double u_max =(double) 0.04 / (double) scale;
        double rho0 = 1.0;
        // **********************************
        // DEFINE SIMULATION SETUP AND GEOMETRY
        // **********************************

        // make BoxArray and Geometry

        amrex::BoxArray ba;
        amrex::Geometry geom;

        // define lower and upper indices
        amrex::IntVect dom_lo(0, 0);
        amrex::IntVect dom_hi(Nx - 1, Ny - 1);

        // Make a single box that is the entire domain
        amrex::Box domain(dom_lo, dom_hi);

        // Initialize the boxarray "ba" from the single box "domain"
        ba.define(domain);

        // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
        ba.maxSize(max_grid_size);

        // This defines the physical box, 
        amrex::RealBox real_box({0., 0.},
                                {Nx, Ny });

        // periodic in all direction
        amrex::Array<int, 2> is_periodic{1, 1};

        // This defines a Geometry object
        geom.define(domain, real_box, amrex::CoordSys::cartesian, is_periodic);

        // extract dx from the geometry object
        amrex::GpuArray<amrex::Real, 2> dx = geom.CellSizeArray();
        amrex::Print() << "dx: " << dx[0] << " , " << dx[1] << "\n";
        // Nghost = number of ghost cells for each array
        int Nghost = 1;

        // Ncomp = number of components for each array
        int Ncomp = 1;

        // How Boxes are distrubuted among MPI processes
        amrex::DistributionMapping dm(ba);

        // we allocate two f multifabs; one will store the old state, the other the new.

        amrex::MultiFab f_old(ba, dm, ndir, Nghost);
        amrex::MultiFab f_new(ba, dm, ndir, Nghost);
        amrex::MultiFab rho(ba, dm, Ncomp, Nghost);
        amrex::MultiFab ux(ba, dm, Ncomp, Nghost);
        amrex::MultiFab uy(ba, dm, Ncomp, Nghost);

        // time = starting time in the simulation
        amrex::Real time = 0.0;
        
        // // **********************************
        // // INITIALIZE DATA LOOP
        // // **********************************

        taylor_green(time, rho, ux, uy, rho0, u_max, nu, Nx, Ny, dx,Nghost);
        init_equilibrium(f_old, rho, ux, uy,Nghost);
        f_new.FillBoundary(geom.periodicity());
        f_old.FillBoundary(geom.periodicity());
        rho.FillBoundary(geom.periodicity());
        ux.FillBoundary(geom.periodicity());
        uy.FillBoundary(geom.periodicity());
        // **********************************
        // WRITE INITIAL PLOT FILE
        // **********************************

        // Write a plotfile of the initial data if plot_int > 0
        if (plot_int > 0)
        {
            int step = 0;
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            amrex::MultiFab plotfile_mf(ba, dm, ndir+3, Nghost, amrex::MFInfo() );
             
            amrex::MultiFab::Copy(plotfile_mf,rho,0,0,1,Nghost);
            amrex::MultiFab::Copy(plotfile_mf,ux,0,1,1,Nghost);
            amrex::MultiFab::Copy(plotfile_mf,uy,0,2,1,Nghost);
            amrex::MultiFab::Copy(plotfile_mf,f_old,0,3,ndir,Nghost);
            WriteSingleLevelPlotfile(pltfile, plotfile_mf, {"rho","ux","uy","f1","f2","f3","f4","f5","f6","f7","f8","f9"}, geom, time, 0);
            
        }
        
        // **********************************
        // MAIN TIME EVOLUTION LOOP
        // **********************************
        
        for (int step = 1; step <= NSTEP; ++step)
        {
            amrex::Print() << "step: " << step << "\n";
            stream(f_old, f_new, Nx, Ny,Nghost);
            
            compute_rho_u(f_new, rho, ux, uy,Nghost);
            amrex::MultiFab::Swap(f_new, f_old, 0, 0, ndir, Nghost);
            f_new.FillBoundary(geom.periodicity());
            f_old.FillBoundary(geom.periodicity());
            rho.FillBoundary(geom.periodicity());
            ux.FillBoundary(geom.periodicity());
            uy.FillBoundary(geom.periodicity());
            collide(f_new ,f_old,rho, ux, uy, nu,Nghost);
            amrex::MultiFab::Swap(f_new, f_old, 0, 0, ndir, Nghost);
            f_new.FillBoundary(geom.periodicity());
            f_old.FillBoundary(geom.periodicity());
            rho.FillBoundary(geom.periodicity());
            ux.FillBoundary(geom.periodicity());
            uy.FillBoundary(geom.periodicity());

            // fills physical domain boundary ghost cells for a cell-centered multifab
            // if (not geom.isAllPeriodic()) {
            //     GpuBndryFuncFab<MyExtBCFill> bf(MyExtBCFill{});
            //     PhysBCFunct<GpuBndryFuncFab<MyExtBCFill> > physbcf(geom, bc, bf);
            //     physbcf(mf, 0, mf.nComp(), mf.nGrowVector(), time, 0);
            // }
            if (plot_int > 0){
                if(step %  plot_int == 0)
                    {
                    
                        const std::string& pltfile = amrex::Concatenate("plt",step,5);
                        amrex::MultiFab plotfile_mf(ba, dm, ndir+3, Nghost, amrex::MFInfo() );
                        
                        amrex::MultiFab::Copy(plotfile_mf,rho,0,0,1,Nghost);
                        amrex::MultiFab::Copy(plotfile_mf,ux,0,1,1,Nghost);
                        amrex::MultiFab::Copy(plotfile_mf,uy,0,2,1,Nghost);
                        amrex::MultiFab::Copy(plotfile_mf,f_old,0,3,ndir,Nghost);
                        WriteSingleLevelPlotfile(pltfile, plotfile_mf, {"rho","ux","uy","f1","f2","f3","f4","f5","f6","f7","f8","f9"}, geom, step, 0);
                    }
            }
            
        }
    }
    amrex::Finalize();
    return 0;
}
