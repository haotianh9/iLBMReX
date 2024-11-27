 %main.m
 %LBM- 2-D2Q9, flow over an obstacle, Re=400, note that c2=1/3, w9=4/9,
 % w1-4=1/9, and w5-w8, 1/36
 clear
 nx=501;ny=81;
 uo=0.1;
 f=zeros(nx,ny,9);feq=zeros(nx,ny,9);utim=zeros(1001);count=zeros(1001);
 u=uo*ones(nx,ny);v=zeros(nx,ny);
 rho=2.*ones(nx,ny);x=zeros(nx);y=zeros(ny);
 Tm=zeros(nx);w(9)=zeros; Tvm=zeros(nx);
 w=[1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36 4/9];
 cx = [1 0 -1 0 1 -1 -1 1 0];
 cy= [0 1 0 -1 1 1 -1 -1 0];
 c2=1./3.;
 dx=1.0;dy=1.0;
 xl=(nx-1)/(ny-1); yl=1.0;
 x=(0:1:nx-1);
 y=(0:1:ny-1);
 alpha=0.01;
 ReH=uo*(ny-1)/alpha
 ReD=uo*10./alpha
 omega=1./(3.*alpha+0.5);
 count(1)=0;
 %setting velocity
 for j=2:ny-1
 u(1,j)=uo;
 end
 %Main Loop
 for kk=1:2090
     kk
 % Collitions
 [f]=collition(nx,ny,u,v,cx,cy,omega,f,rho,w);
 % Streaming:
 [f]=stream(f);
 % End of streaming
 %Boundary condition:
 [f]=boundary(nx,ny,f,uo,rho);
 %Obsticale
 [f]=obstc(nx,ny,f,uo,rho);
 % Calculate rho, u, v
 [rho,u,v]=ruv(nx,ny,f);
 count(kk)=kk;
 utim(kk)=rho((nx-1)/2,(ny-1)/2);
 end
 %Plotting data
 result(nx,ny,x,y,u,v,uo,rho,count,utim);
 % +++++++++++++++++++++++++++++++++++++++++++++++
 %Boudary conditions for Channel flow
 function [f]=boundary(nx,ny,f,uo,rho)
 %right hand boundary
 for j=1:ny
 f(nx,j,3)=f(nx-1,j,3);
 f(nx,j,7)=f(nx-1,j,7);
 f(nx,j,6)=f(nx-1,j,6);
 end
 %bottom, and top boundary, bounce back
 for i=1:nx
 f(i,1,2)=f(i,1,4);
 f(i,1,5)=f(i,1,7);
 f(i,1,6)=f(i,1,8);
 f(i,ny,4)=f(i,ny,2);
 f(i,ny,7)=f(i,ny,5);
 f(i,ny,8)=f(i,ny,6);
 u(i,1)=0.0; v(i,1)=0.0;
 u(i,ny)=0.0; v(i,ny)=0.0;
 end
 %Left boundary, velocity is given= uo
 for j=2:ny-1
 f(1,j,1)=f(1,j,3)+2.*rho(1,j)*uo/3.;
 f(1,j,5)=f(1,j,7)-0.5*(f(1,j,2)-f(1,j,4))+rho(1,j)*uo/6.;
 f(1,j,8)=f(1,j,6)+0.5*(f(1,j,2)-f(1,j,4))+rho(1,j)*uo/6.;
 u(1,j)=uo; v(1,j)=0.0;
 end
 % End of boundary conditions.
 end
 % +++++++++++++++++++++++++++++++++++++++
 % Collition
 function [f]=collition(nx,ny,u,v,cx,cy,omega,f,rho,w)
 for j=1:ny
      for i=1:nx
 t1=u(i,j)*u(i,j)+v(i,j)*v(i,j);
 for k=1:9
 t2=u(i,j)*cx(k)+v(i,j)*cy(k);
 feq(i,j,k)=rho(i,j)*w(k)*(1.0+3.0*t2+4.5*t2*t2-1.5*t1);
 f(i,j,k)=(1.-omega)*f(i,j,k)+omega*feq(i,j,k);
 end
 end
 end
 end
 % +++++++++++++++++++++++++++++++++++++++++++++
 %Obsticale replace at the entrance, Back Fase Flow
 function [f]=obstc(nx,ny,f,uo,rho)
 %length of obsticale= nx/5, and has sides of 10 units
 nxb=(nx-1)/5;
 nxe=nxb+10;
 nyb=((ny-1)-10)/2;
 nyb=35;
 nye=nyb+10;
 for i=nxb:nxe
     f(i,nyb,4)=f(i,nyb,2);
     f(i,nyb,7)=f(i,nyb,5);
     f(i,nyb,8)=f(i,nyb,6);
     f(i,nye,2)=f(i,nye,4);
     f(i,nye,5)=f(i,nye,7);
     f(i,nye,6)=f(i,nye,8);
 end
 %bottom, and top boundary, bounce back
 for j=nyb:nye
     f(nxb,j,3)=f(nxb,j,1);
     f(nxb,j,7)=f(nxb,j,5);
     f(nxb,j,6)=f(nxb,j,8);
     f(nxe,j,1)=f(nxe,j,3);
     f(nxe,j,5)=f(nxe,j,7);
     f(nxe,j,8)=f(nxe,j,8);
 end
 for i=nxb:nxe
 for j=nyb:nye
 u(i,j)=0.0;
 v(i,j)=0.0;
 end
 end
 % End
 end
 % ++++++++++++++++++++++++++++++++++++++++++++++++++++
 % Plots for channel flow
 function result(nx,ny,x,y,u,v,uo,rho,count,utim)
 for j=1:ny
 Tm1(j)=u(51,j)/uo;
 Tm2(j)=u(101,j)/uo;
 Tm3(j)=u(261,j)/uo;
 Tm4(j)=u(301,j)/uo;
 end
 for i=1:nx
 umx(i)=u(i,(ny-1)/2)/uo;
 vmx(i)=v(i,(ny-1)/2)/uo;
 end
figure
 plot(x/(nx-1),umx,x/(nx-1),vmx,'LineWidth',1.5)
 figure
 plot(Tm1,y,Tm2,y,Tm3,y,Tm4,y,'LineWidth',1.5)
 xlabel('U')
 ylabel('Y')
 figure
 plot(count,utim)
 %Stream function calculation
 for j=1:ny
 sx(:,j)=x(:);
 end
 for i=1:nx
 sy(i,:)=y(:);
 end
 str=zeros(nx,ny);
 for i=1:nx
 for j=2:ny
 str(i,j)=str(i,j-1)+0.5*(u(i,j)+u(i,j-1));
 end
 end
 figure
 contour(sx,sy,str)
 figure
 contour(sx,sy,u,'LineWidth',1.0)
 axis equal
 end
 % ++++++++++++++++++++++++++++++++++++++++++++
 function[rho,u,v]=ruv(nx,ny,f)
 rho=sum (f,3);
 for i=1:nx
 rho(i,ny)=f(i,ny,9)+f(i,ny,1)+f(i,ny,3)+2.*(f(i,ny,2)+f(i,ny,6)+f(i,ny,5));
 end
 %calculate velocity compnents
 u = ( sum(f(:,:,[1 5 8]),3)- sum(f(:,:,[3 6 7]),3) )./rho;
 v = ( sum(f(:,:,[2 5 6]),3)- sum(f(:,:,[4 7 8]),3) )./rho;
 end
 % +++++++++++++++++++++++++++++++++++++++++++++++++
 % Streaming:
 function [f]=stream(f)
 f(:,:,1)=circshift( squeeze(f(:,:,1)), [+1,+0] );
 f(:,:,2)=circshift( squeeze(f(:,:,2)), [+0,+1] );
 f(:,:,3)=circshift( squeeze(f(:,:,3)), [-1,+0] );
 f(:,:,4)=circshift( squeeze(f(:,:,4)), [+0,-1] );
 f(:,:,5)=circshift( squeeze(f(:,:,5)), [+1,+1] );
 f(:,:,6)=circshift( squeeze(f(:,:,6)), [-1,+1] );
 f(:,:,7)=circshift( squeeze(f(:,:,7)), [-1,-1] );
 f(:,:,8)=circshift( squeeze(f(:,:,8)), [+1,-1] );
 end
 % End of streaming
